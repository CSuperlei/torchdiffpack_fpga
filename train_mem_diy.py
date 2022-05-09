#! /usr/bin/env python
from ast import dump
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter
import math
import sys
import shutil
import tqdm
# from adjoint_mem import odesolve_adjoint_sym12  as odesolve
# from odesolver_endtime import odesolve_endtime as odesolve
from adjoint import odesolve_adjoint as odesolve

"""--------------------"""
import create_sparse
import dynamic_sparsity
import yaml

import graphviz
from torchviz import make_dot


# from torch_ode_cifar import odesolve
def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > 40:
        optim_factor = 4
    elif epoch > 30:
        optim_factor = 3
    elif epoch > 20:
        optim_factor = 2
    elif epoch > 10:
        optim_factor = 1

    return lr / math.pow(10, (optim_factor))

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'sqnxt'], default='resnet')
parser.add_argument('--method', type=str, choices=['Euler', 'RK3', 'RK4', 'RK23', 'Sym12Async', 'RK12','Dopri5'], default='RK23')
parser.add_argument('--datasetname', type=str, choices=['Mnist', 'Cifar10', 'Cifar100'], default='Cifar10')
parser.add_argument('--num_epochs', type=int, default=90)
parser.add_argument('--start_epoch', type=int, default=0)
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='./checkpoint', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--h', type=float, default=None, help='Initial Stepsize')
parser.add_argument('--t0', type=float, default=1.0, help='Initial time')
parser.add_argument('--t1', type=float, default=0.0, help='End time')
parser.add_argument('--rtol', type=float, default=1e-4, help='Releative tolerance')
parser.add_argument('--atol', type=float, default=1e-4, help='Absolute tolerance')
parser.add_argument('--print_neval', type=bool, default=False, help='Print number of evaluation or not')
parser.add_argument('--neval_max', type=int, default=50000, help='Maximum number of evaluation in integration')

parser.add_argument('--batch_size', type=int, default=256)

args = parser.parse_args()

yaml_file = './unstructured_constant_resnet18_schedule_90.yaml'
# yaml_file = './unstructured_momentum_resnet18_schedule_90.yaml'
with open(yaml_file, 'r') as stream:
    try:
        loaded_schedule = yaml.load(stream, yaml.SafeLoader)
    except yaml.YAMLError as exc:
        print(exc)

# lr_schedule = loaded_schedule['lr_schedule']
yaml_sparsity=loaded_schedule['sparsity']
yaml_topk_period=loaded_schedule['topk_period']
yaml_num_conv_layer=loaded_schedule['num_conv_layer']
yaml_bn_weight_decay=loaded_schedule['bn_weight_decay']
yaml_nesterov=loaded_schedule['nesterov']
yaml_train_batch_size=loaded_schedule['train_batch_size']
yaml_test_batch_size=loaded_schedule['test_batch_size']
yaml_weight_decay=loaded_schedule['weight_decay']
yaml_momentum=loaded_schedule['momentum']
yaml_warmup=loaded_schedule['warmup']
yaml_total_epoch=loaded_schedule['total_epoch']
yaml_pruning_type=loaded_schedule['pruning_type']
yaml_pruning_mode=loaded_schedule['pruning_mode']
yaml_sparsify_first_layer=loaded_schedule['sparsify_first_layer']
yaml_sparsify_last_layer =loaded_schedule['sparsify_last_layer']

# if args.network == 'sqnxt':
    # from models.sqnxt import SqNxt_23_1x

    # writer = SummaryWriter(
        # 'sqnxt/' + args.network + '_mem_' + args.method + '_lr_' + str(args.lr) + '_h_' + str(args.h) + '/')
if args.network == 'resnet':
    # from odels.resnet import ResNet18
    from resnet import ResNet18

    writer = SummaryWriter(
        'tensorboard_resnet/' + args.network + '_' + args.datasetname + '_' + args.method + '_lr_' + str(args.lr) + '_h_' + str(args.h) + '/')

num_epochs = int(args.num_epochs)
lr = float(args.lr)
start_epoch = int(args.start_epoch)
batch_size = int(args.batch_size)

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        '''

        :param odefunc: 就是BasicBlock
        '''
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.options = {}
        self.options.update({'method': args.method})
        self.options.update({'h': args.h})
        self.options.update({'t0': args.t0})
        self.options.update({'t1': args.t1})
        self.options.update({'rtol': args.rtol})
        self.options.update({'atol': args.atol})
        self.options.update({'print_neval': args.print_neval})
        self.options.update({'neval_max': args.neval_max})
        print(self.options)

    def forward(self, x):
        '''

        :param x: 就是y0
        :return:
        '''
        ## ode一层需要的时间
        start = time.time()
        out = odesolve(self.odefunc, x, self.options)
        end = time.time()
        # print("ODE Block Time", end - start)  ##integration layer
        return out

    @property
    def nfe(self):
        '''
        可以直接像调用属性一样调用 o.nfe 相当于调用 o.ndefunc.nfe
        :return:
        '''
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        '''
        直接可以 o.nfe = 10 进行赋值，相当于给o.defuc.nfe进行赋值
        :param value:
        :return:
        '''
        self.odefunc.nfe = value


def conv_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1 and m.bias is not None:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


# Data Preprocess
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081, )),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307, ), (0.3081, )),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar-10', transform=transform_train, train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar-10', transform=transform_test, train=False, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

if args.network == 'sqnxt':
    pass
    # net = SqNxt_23_1x(10, ODEBlock)
elif args.network == 'resnet':
    ### 把ODEBlock传入到ResNet中
    net = ResNet18(ODEBlock)
    # writer.add_graph(net, (dummy_input, ))

net.apply(conv_init)
print(net)

if is_use_cuda:
    net.cuda()  # to(device)
    # net = nn.DataParallel(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def train(epoch, LayerSparsity=None):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # sparsity_budget = LayerSparsity.get_sparsity()
    # configuration_data = {'epoch':epoch, 'batch_idx':0, 'layer':0, 'type':0, 'period':yaml_topk_period, 'sparsity':sparsity_budget, 'warmup': yaml_warmup, 'global_sparsity': yaml_sparsity, 'pruning_type': yaml_pruning_type}
    # torch.save(configuration_data, 'configuration_data_0')

    print('Training Epoch: #%d, LR: %.4f, Time: [%s]' % (epoch, lr_schedule(lr, epoch), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    for idx, (inputs, labels) in enumerate(train_loader):
        if is_use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        # sparsity_budget = LayerSparsity.get_sparsity()  ### return self.sparsity
        # configuration_data = {'epoch':epoch, 'batch_idx':0, 'layer':0, 'type':0, 'period':yaml_topk_period, 'sparsity':sparsity_budget, 'warmup': yaml_warmup, 'global_sparsity': yaml_sparsity, 'pruning_type': yaml_pruning_type}
        # torch.save(configuration_data, 'configuration_data_0')

        with torch.autograd.set_detect_anomaly(True):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

        LayerSparsity.gather_statistic() ### 不执行任何语句
        optimizer.step()
        LayerSparsity.step() ## 不执行任何语句
        # sparsity_budget = LayerSparsity.get_sparsity()
        # torch.save(configuration_data, 'configuration_data_0')

        writer.add_scalar('Train/Loss', loss.item(), epoch * 50000 + batch_size * (idx + 1))
        train_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()

        # sys.stdout.write('\r')
        sys.stdout.write('[%s] Training Epoch [%d/%d] Iter[%d/%d]\t\t Loss: %.4f Acc@top1: %.3f \t\t\t \r'
                         % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                            epoch, num_epochs, idx, len(train_dataset) // train_loader.batch_size,
                            train_loss / (batch_size * (idx + 1)), correct / total))
       
        sys.stdout.flush()

    writer.add_scalar('Train/Accuracy', correct / total, epoch)


def test(epoch, LayerSparsity=None):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # sparsity_budget = LayerSparsity.get_sparsity()
    # configuration_data = {'epoch':epoch, 'batch_idx':0, 'layer':0, 'type':0, 'period':yaml_topk_period, 'sparsity':sparsity_budget, 'warmup': yaml_warmup, 'global_sparsity': yaml_sparsity, 'pruning_type': yaml_pruning_type}
    # torch.save(configuration_data, 'configuration_data_0')

    for idx, (inputs, labels) in enumerate(test_loader):
        # if idx == 5:
        #     break
        if is_use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # sparsity_budget = LayerSparsity.get_sparsity()
        # configuration_data = {'epoch':epoch, 'batch_idx':0, 'layer':0, 'type':0, 'period':yaml_topk_period, 'sparsity':sparsity_budget, 'warmup': yaml_warmup, 'global_sparsity': yaml_sparsity, 'pruning_type': yaml_pruning_type}
        # torch.save(configuration_data, 'configuration_data_0')

        with torch.no_grad():
            outputs = net(inputs)
            
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()
        writer.add_scalar('Test/Loss', loss.item(), epoch * 50000 + test_loader.batch_size * (idx + 1))

        # sys.stdout.write('\r')
        sys.stdout.write('[%s] Testing Epoch [%d/%d] Iter[%d/%d]\t\t Loss: %.4f Acc@top1: %.3f \t\t\t \r'
                         % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                            epoch, num_epochs, idx, len(test_dataset) // test_loader.batch_size,
                            test_loss / (100 * (idx + 1)), correct / total))

        sys.stdout.flush()

    acc = correct / total
    writer.add_scalar('Test/Accuracy', acc, epoch)
    return acc


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


if __name__ == '__main__':
    best_acc = 0.0
    LayerSparsity = dynamic_sparsity.layerSparsity(optimizer, yaml_pruning_type, yaml_sparsify_first_layer, yaml_sparsify_last_layer)
    LayerSparsity.add_module(net, 1 - yaml_sparsity, yaml_pruning_mode)

    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    #     args.checkpoint = os.path.dirname(args.resume)
    #     checkpoint = torch.load(args.resume)
    #     best_acc = checkpoint['best_acc']
    #     start_epoch = checkpoint['epoch']
    #     net.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    # for _epoch in range(start_epoch, start_epoch + num_epochs):
    #     start_time = time.time()

    #     _lr = lr_schedule(args.lr, _epoch)
    #     adjust_learning_rate(optimizer, _lr)

    #     train(_epoch, LayerSparsity=LayerSparsity)
    #     print()
    #     test_acc = test(_epoch, LayerSparsity=LayerSparsity)
    #     print()
    #     print()
    #     end_time = time.time()
    #     print('Epoch #%d Cost %ds' % (_epoch, end_time - start_time))

    #     # save model
    #     is_best = test_acc > best_acc
    #     best_acc = max(test_acc, best_acc)
    #     save_checkpoint({
    #         'epoch': _epoch + 1,
    #         'state_dict': net.state_dict(),
    #         'acc': test_acc,
    #         'best_acc': best_acc,
    #         'optimizer': optimizer.state_dict(),
    #     }, is_best, checkpoint=args.checkpoint + '_' + args.datasetname + '_' + args.method + '_' + args.network)

    # print('Best Acc@1: %.4f' % (best_acc * 100))

    #####################
    load_checkpoint_str = args.checkpoint + '_' + args.datasetname + '_' + args.method + '_' + args.network
    print(load_checkpoint_str)
    checkpoint = torch.load(load_checkpoint_str + '/model_best.pth.tar')
    net.load_state_dict(checkpoint['state_dict'])
    sparsity_budget = LayerSparsity.get_sparsity()
    if yaml_pruning_type == 'unstructured':
        net.load_state_dict(create_sparse.sparse_unstructured(net.state_dict(), sparsity_budget))
    
    # with torch.autograd.profiler.pddrofile(enabled=True) as prof:
    #     top1_acc = test(yaml_total_epoch, LayerSparsity=LayerSparsity)
    
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    torch.cuda.synchronize()
    start = time.time()
    top1_acc = test(yaml_total_epoch, LayerSparsity=LayerSparsity)
    # print('\n')
    torch.cuda.synchronize()
    end = time.time()
    print('Test Time: ', end - start)

    print('\n')
    print('Sparsity Top1_acc@1: %.4f%%'% (top1_acc * 100))

    state = {
        'state_dict': net.state_dict(),
        'acc': top1_acc,
        'epoch': yaml_total_epoch,
        'optimizer': optimizer.state_dict(),

    }
    # save_checkpoint(state, is_best, 'pruned_best_model')
    # save_checkpoint(state, True, 'pruned_best_model')
    #######################
    writer.close()

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
import os
import shutil
# from TorchDiffEqPack.odesolver_mem import odesolve_adjoint as odesolve
from adjoint import odesolve_adjoint as odesolve
import create_sparse
import dynamic_sparsity
import yaml


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# def lr_schedule(lr, epoch):
#     optim_factor = 0
#     if epoch > 60:
#         optim_factor = 2
#     elif epoch > 30:
#         optim_factor = 1

#     return lr / math.pow(10, (optim_factor))

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

# from torch_ode_cifar import odesolve

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'sqnxt'], default='resnet')
parser.add_argument('--method', type=str, default='RK23')
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=0)

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='./checkpoint', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')

parser.add_argument('--resume', default='./checkpoint_Cifar10_RK23_resnet/model_best.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--datasetname', type=str, choices=['Mnist', 'Cifar10', 'Cifar100'], default='Cifar10')

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

#################################
###        sparsity          ###
################################
yaml_file = './unstructured_constant_resnet18_schedule_90.yaml'
with open(yaml_file, 'r') as stream:
    try:
        loaded_schedule = yaml.load(stream, yaml.SafeLoader)
    except yaml.YAMLError as exc:
        print(exc)

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

if args.network == 'sqnxt':
    pass
    # from models.sqnxt import SqNxt_23_1x

elif args.network == 'resnet':
    from resnet import ResNet18

num_epochs = int(args.num_epochs)
lr = float(args.lr)
start_epoch = int(args.start_epoch)
batch_size = int(args.batch_size)

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")

ODEBlcok_Time = []
class ODEBlock(nn.Module):

    def __init__(self, odefunc):
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
        #self.options.update({'t_eval': [args.t1]})
        # print(self.options)

    def forward(self, x):
        start = time.time()
        out = odesolve(self.odefunc, x, self.options)
        end = time.time()
        # print("ODE Block Time:", end - start)
        global ODEBlcok_Time
        ODEBlcok_Time.append(end - start)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
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
    # transforms.Normalize((0.1307,), (0.3081, )),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar-10', transform=transform_train, train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar-10', transform=transform_test, train=False, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=0, shuffle=False)

if args.network == 'sqnxt':
    pass
    # net = SqNxt_23_1x(10, ODEBlock)
elif args.network == 'resnet':
    net = ResNet18(ODEBlock)

net.apply(conv_init)
# print(net)

print('Total number of parameters: {}'.format(sum(p.numel() for p in net.parameters())))

if is_use_cuda:
    net.cuda()
    # net = nn.DataParallel(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        if idx != 0:
            continue

        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = net(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()

        sys.stdout.write('\r')
        sys.stdout.write('[%s] Testing Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
                         % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                            epoch, num_epochs, idx, len(test_dataset) // test_loader.batch_size,
                            test_loss / (100 * (idx + 1)), correct / total))
        
        print()
        sys.stdout.flush()

    acc = correct / total
    return acc


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
#     if not os.path.exists(checkpoint):
#         os.makedirs(checkpoint)
#     filepath = os.path.join(checkpoint, filename)
#     torch.save(state, filepath)
#     if is_best:
#         shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))




density_array = [
0.45,
0.45,
0.35,   ## ode_layer1_conv1
0.35,   ## ode_layer1_conv2
0.45,
0.45,
0.40,   ## ode_layer2_conv1
0.40,   ## ode_layer2_conv2
0.45,
0.45,
0.38,   ## ode_layer3_conv1
0.38,   ## ode_layer3_conv2
0.45,
0.45,
0.33,## ode_layer4_conv1
0.33,   ## ode_layer4_conv2


]

density_value = {
                'conv1.weight' : 1,
                'layer1_1.0.conv1.weight' : density_array[0], 
                'layer1_1.0.conv2.weight' : density_array[1],
                'layer1_2.0.odefunc.conv1.weight' : density_array[2],
                'layer1_2.0.odefunc.conv2.weight' : density_array[3],
                'layer2_1.0.conv1.weight' : density_array[4],
                'layer2_1.0.conv2.weight' : density_array[5],
                'layer2_1.0.shortcut.0.weight' : 1,
                'layer2_2.0.odefunc.conv1.weight' : density_array[6],
                'layer2_2.0.odefunc.conv2.weight' : density_array[7],
                'layer3_1.0.conv1.weight' : density_array[8],
                'layer3_1.0.conv2.weight' : density_array[9],
                'layer3_1.0.shortcut.0.weight' : 1,
                'layer3_2.0.odefunc.conv1.weight' : density_array[10],
                'layer3_2.0.odefunc.conv2.weight' : density_array[11],
                'layer4_1.0.conv1.weight' : density_array[12],
                'layer4_1.0.conv2.weight' : density_array[13],
                'layer4_1.0.shortcut.0.weight' : 1,
                'layer4_2.0.odefunc.conv1.weight' : density_array[14], 
                'layer4_2.0.odefunc.conv2.weight' : density_array[15],
                'linear.weight' : 1
            }

# def cos_similarity(cos_a, cos_b):
#     print(cos_a.size(), cos_b.size())
#     # A = cos_a[0].reshape(-1).detach().cpu().numpy()
#     A = cos_a.reshape(-1).detach().cpu().numpy()
#     # B = cos_b[0].reshape(-1).detach().cpu().numpy()
#     B = cos_b.reshape(-1).detach().cpu().numpy()
#     num = float(A.dot(B)) #?????????????????? A * B.T
#     denom = np.linalg.norm(A) * np.linalg.norm(B)
#     cos = num / denom #?????????
#     print('cos1',cos)

#     # A = cos_a[1].reshape(-1).detach().cpu().numpy()
#     # B = cos_b[1].reshape(-1).detach().cpu().numpy()
#     # num = float(A.dot(B)) #?????????????????? A * B.T
#     # denom = np.linalg.norm(A) * np.linalg.norm(B)
#     # print('cos2', num/denom)
#     # cos += num / denom #?????????
 
#     # # cos = 0.5 + 0.5 * cos #?????????(0,1)
#     if cos > 1.0:
#         cos = 1.0
#         # print(layer_name, cos)
#     return cos 


if __name__ == "__main__":
    best_acc = 0.0
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        if yaml_pruning_type is not None:
            prunning_layers = torch.load('./conv_linear_name')
            # density_value = 0.4
            for k, v in prunning_layers.items():
                prunning_layers[k] = density_value[k]
                # prunning_layers[k] = 0.4
            
            LayerSparsity = dynamic_sparsity.layerSparsity(optimizer, yaml_pruning_type, yaml_sparsify_first_layer, yaml_sparsify_last_layer)
            LayerSparsity.add_module(net, 1 - yaml_sparsity, yaml_pruning_mode, prunning_layers) ## ????????? density = 1 - sparsity
            sparsity_budget = LayerSparsity.get_sparsity()   
            if yaml_pruning_type == 'unstructured':
                net.load_state_dict(create_sparse.sparse_unstructured(net.state_dict(), sparsity_budget))
                torch.save(net.state_dict(), './checkpoint_Cifar10_RK23_resnet/sparsity_weight_ode_layer_4_005.pth.tar')

    for _epoch in range(1):
        start_time = time.time()

        # _lr = lr_schedule(args.lr, _epoch)
        # adjust_learning_rate(optimizer, _lr)

        test_acc = test(_epoch)
        print()
        print()
        end_time = time.time()
        print('Epoch #%d Cost %ds' % (_epoch, end_time - start_time))

        # save model

    print('Best Acc@1: %.4f\n' % (best_acc * 100))
    print('Test Acc@1: %.4f\n' % (test_acc * 100))

    ## Calculate ODEBlock Time
    # sum_1 = 0
    # sum_2 = 0
    # sum_3 = 0
    # sum_4 = 0
    # for i in range(len(ODEBlcok_Time)):
    #     if i % 4 == 0:
    #         sum_1 += ODEBlcok_Time[i]
    #     elif i % 4 == 1:
    #         sum_2 += ODEBlcok_Time[i]
    #     elif i % 4 == 2:
    #         sum_3 += ODEBlcok_Time[i]
    #     elif i % 4 == 3:
    #         sum_4 += ODEBlcok_Time[i]

    # print(sum_1 * 4 / len(ODEBlcok_Time), sum_2 * 4 / len(ODEBlcok_Time), sum_3 * 4 / len(ODEBlcok_Time), sum_4 * 4 / len(ODEBlcok_Time))

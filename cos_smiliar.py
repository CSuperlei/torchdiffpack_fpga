'''
Author: CSuperlei
Date: 2022-03-02 15:25:42
LastEditTime: 2022-03-22 20:00:50
Description: 
'''
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

from transformers import Wav2Vec2ForSequenceClassification
# from TorchDiffEqPack.odesolver_mem import odesolve_adjoint as odesolve
from adjoint import odesolve_adjoint as odesolve
import create_sparse
import dynamic_sparsity
import yaml

from resnet import ResNet18


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

def cos_similarity(A, B):
    num = float(A.dot(B)) #若为行向量则 A * B.T
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    cos = num / denom #余弦值
    # cos = 0.5 + 0.5 * cos #归一化(0,1)
    return cos

net = ResNet18(ODEBlock)
net.apply(conv_init)
net.cuda()
# print(net)

### 加载要稀疏的weight名字
weight_names = torch.load('./conv_linear_name')
weight_names_list = []
for k, v in weight_names.items():
    weight_names_list.append(k)

### 加载稀疏化的方法
# params = torch.load('./checkpoint_Cifar10_RK23_resnet/sparsity_weight_ode_layer_045.pth.tar')
params = torch.load('./checkpoint_Cifar10_RK23_resnet/model_best.pth.tar')
net.load_state_dict(params['state_dict'])
weight_sparsity_list = []
for k, v in net.named_parameters():  
    # print(k)  ## 层的名字
    #print(v)  ##  参数值,这里打印会有省略号，计算的时候要整体复制到变量里计算
    if k in weight_names_list:
        v = v.reshape(-1).detach().cpu().numpy()
        weight_sparsity_list.append(v)

### 加载原始没有稀疏化的方法
params = torch.load('./checkpoint_Cifar10_RK23_resnet/sparsity_weight_ode_layer_4_005.pth.tar')
net.load_state_dict(params)
weight_list = []
for k, v in net.named_parameters():  
    # print(k)  ## 层的名字
    #print(v)  ##  参数值,这里打印会有省略号，计算的时候要整体复制到变量里计算
    if k in weight_names_list:
        v = v.reshape(-1).detach().cpu().numpy()
        # print(v)
        weight_list.append(v)

cos_list = []
for i in range(len(weight_list)):
    cos_list.append(cos_similarity(weight_sparsity_list[i], weight_list[i]))
print('\n0.5 sparsity:')
for i in range(len(weight_names_list)):
    print(weight_names_list[i], cos_list[i])

# params = torch.load('/home/leic/workplace/cailei/cv_project/cv_files/torch_diff_pack_diy/checkpoint_Cifar10_RK23_resnet/model_sparsity_fixed_new.pth.tar')
# net.load_state_dict(params)
# weight_sparsity_increment_list = []
# for k, v in net.named_parameters():  
#     if k in weight_names_list:
#         v = v.reshape(-1).detach().cpu().numpy()
#         weight_sparsity_increment_list.append(v)

# cos_increment_list = []
# for i in range(len(weight_list)):
#     cos_increment_list.append(cos_similarity(weight_sparsity_increment_list[i], weight_list[i]))
# print('\nincrement sparsity:')
# for i in range(len(weight_names_list)):
#     print(weight_names_list[i], cos_increment_list[i])

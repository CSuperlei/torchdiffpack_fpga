'''
Author: CSuperlei
Date: 2022-03-18 10:40:26
LastEditTime: 2022-03-20 08:02:01
Description: 
'''
import time
import torch
import numpy as np
import torch.nn.functional as F

layer1_1_feature_map = torch.load('./checkpoint_Cifar100_RK23_resnet/layer3_1_featuremap_out.pth.tar')

fp_1 = layer1_1_feature_map[0, :, :, :]
fp_1 = F.pad(fp_1, [1,1,1,1], 'constant', 0)

layer1_2_ode_conv1_weight = None
cifar100_spar_weight = torch.load('./checkpoint_Cifar100_RK23_resnet/sparsity_weight.pth.tar')

for k, v in cifar100_spar_weight.items():
    if 'layer3_2.0.odefunc.conv1.weight' in k:
        layer1_2_ode_conv1_weight = v

print('fp_1', fp_1.size())
print('layer1_2', layer1_2_ode_conv1_weight.size())

def convolution(k, g):
    m, n = g.size()
    sum = 0
    for i in range(m - 3):
        for j in range(n - 3):
            a = g[i:i + 3, j:j+3]
            res = torch.multiply(a,k)
            sum = sum + torch.count_nonzero(res)
    return sum

op = 0
conv_shape = layer1_2_ode_conv1_weight.size()
for i in range(conv_shape[0]):
    tmp1 = layer1_2_ode_conv1_weight[i]
    for j in range(conv_shape[1]):
        k = tmp1[j]
        g = fp_1[j]
        op += convolution(k, g)

print('total_flops', op)

# for i in range(conv_shape[0]):
#     k = layer1_2_ode_conv1_weight[i]
#     g = fp_1[i]
#     # print('k', k.size())
#     # print('g', g.size())
#     op = op + convolution(k, g)

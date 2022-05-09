import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import topk

da = [
0.45,
0.45,
0.30,  ## ode layer1
0.30,  ## ode layer1
0.45,
0.45,
0.35,  ## ode laeyr2
0.35,  ## ode layer2
0.45,
0.45,
0.40, ## ode laeyr3
0.40,  ## ode layer3
0.45,
0.45,
0.30,  ## ode layer 4
0.30,  ## ode layer 4

]

density_array = {
    'layer1_1':[da[0], da[1]],
    'layer1_2':[da[2], da[3]],
    'layer2_1':[da[4], da[5]],
    'layer2_2':[da[6], da[7]],
    'layer3_1':[da[8], da[9]],
    'layer3_2':[da[10], da[11]],
    'layer4_1':[da[12], da[13]],
    'layer4_2':[da[14], da[15]],
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, layer_name=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)#nn.GroupNorm(planes//16, planes) #nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)#nn.GroupNorm(planes//16, planes)#nn.BatchNorm2d(planes)
        self.density = 0.5
        self.layer_name = layer_name
        self.nfe = 0

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)#nn.GroupNorm(self.expansion*planes//16, self.expansion*planes)#
            )

    def forward(self, x):
        self.nfe += 1
        out = F.relu(self.bn1(self.conv1(x)))
        # torch.save(out, './' + self.layer_name + '_bb1_conv1_out.pth.tar')
        # print(self.layer_name+'_conv1', density_array[self.layer_name][0])
        out = topk.drop_hw(out, density_array[self.layer_name][0])
        # torch.save(out, './' + self.layer_name + '_bb1_conv1_pru_out.pth.tar')

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = F.relu(out)

        # torch.save(out, './' + self.layer_name + '_bb1_conv2_out.pth.tar')
        # print(self.layer_name+'_conv2', density_array[self.layer_name][1])
        out = topk.drop_hw(out, density_array[self.layer_name][1])
        # torch.save(out, './' + self.layer_name + '_bb1_conv2_pru_out.pth.tar')

        return out

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, dim, layer_name):
        super(BasicBlock2, self).__init__()
        in_planes = dim
        planes = dim
        stride = 1
        self.nfe = 0
        self.density = 0.5
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) #nn.GroupNorm(planes//16, planes)#
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) #nn.GroupNorm(planes//16, planes)#
        self.layer_name = layer_name

        self.shortcut = nn.Sequential()

    def forward(self,t, x):
        self.nfe += 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # torch.save(out, './' + self.layer_name + '_bb2_conv1_out.pth.tar')
        # print(self.layer_name+'_conv1', density_array[self.layer_name][0])
        out = topk.drop_hw(out, density_array[self.layer_name][0])
        # torch.save(out, './' + self.layer_name + '_bb2_conv1_pru_out.pth.tar')

        out = self.conv2(out)
        out = self.bn2(out)
        #out += self.shortcut(x)
        out = F.relu(out)

        # torch.save(out, './' + self.layer_name + '_bb2_conv2_out.pth.tar')
        # print(self.layer_name+'_conv2', density_array[self.layer_name][1])
        out = topk.drop_hw(out, density_array[self.layer_name][1])
        # torch.save(out, './' + self.layer_name + '_bb2_conv2_pru_out.pth.tar')

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10, ODEBlock_ = None):
        super(ResNet, self).__init__()
        self._planes = 64
        self.in_planes = self._planes
        self.ODEBlock = ODEBlock_

        self.conv1 = nn.Conv2d(3, self._planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self._planes)
        self.layer1_1 = self._make_layer(self._planes, 1, stride=1, layer_name='layer1_1')
        self.layer1_2 = self._make_layer2(self._planes, num_blocks[0]-1, stride=1, layer_name="layer1_2")

        self.layer2_1 = self._make_layer(self._planes*2, 1, stride=2, layer_name="layer2_1")
        self.layer2_2 = self._make_layer2(self._planes*2, num_blocks[1]-1, stride=1, layer_name="layer2_2")

        self.layer3_1 = self._make_layer(self._planes*4, 1, stride=2, layer_name="layer3_1")
        self.layer3_2 = self._make_layer2(self._planes*4, num_blocks[2]-1, stride=1, layer_name="layer3_2")

        self.layer4_1 = self._make_layer(self._planes*8, 1, stride=2, layer_name="layer4_1")
        self.layer4_2 = self._make_layer2(self._planes*8, num_blocks[3]-1, stride=1, layer_name="layer4_2")
        self.linear = nn.Linear(self._planes*8 * block.expansion, num_classes)

        self.activation_sparsity = []

    def _make_layer(self, planes, num_blocks, stride, layer_name=None):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride, layer_name=layer_name))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _make_layer2(self, planes, num_blocks, stride, layer_name=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        #layers.append(nn.BatchNorm2d(self.in_planes))
        for stride in strides:
            layers.append(self.ODEBlock(BasicBlock2(self.in_planes, layer_name=layer_name)))
        return nn.Sequential(*layers)
    
    def cos_similarity(self, layer_name, activation_sparsity):
        A = activation_sparsity[0].reshape(-1).detach().cpu().numpy()
        B = activation_sparsity[1].reshape(-1).detach().cpu().numpy()
        num = float(A.dot(B)) #若为行向量则 A * B.T
        denom = np.linalg.norm(A) * np.linalg.norm(B)
        cos = num / denom #余弦值
        # cos = 0.5 + 0.5 * cos #归一化(0,1)
        if cos > 1.0:
            cos = 1.0
        # print(layer_name, cos)
        return cos


    def forward(self, x):
        # density = 0.65
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1_1(out)
        # self.activation_sparsity.append(out)
        # out = topk.drop_hw(out, density)
        # print('layer1_1_pru_out', out.size())
        # self.activation_sparsity.append(out)
        # self.cos_similarity('layer1_1', self.activation_sparsity)
        # self.activation_sparsity.clear()
        # print('layer1_1_out', out.size())
        # torch.save(out, './checkpoint_Cifar100_RK23_resnet/layer1_1_featuremap_out.pth.tar')

        # start = time.time()
        out = self.layer1_2(out)
        # self.activation_sparsity.append(out)
        # out = topk.drop_hw(out, density)
        # self.activation_sparsity.append(out)
        # self.cos_similarity('layer1_2', self.activation_sparsity)
        # self.activation_sparsity.clear()
        # # end = time.time()
        # print("ode_layer1_time:", end - start)
        # print('layer1_2_out', out.size())

        out = self.layer2_1(out)
        # self.activation_sparsity.append(out)
        # out = topk.drop_hw(out, density)
        # self.activation_sparsity.append(out)
        # self.cos_similarity('layer2_1', self.activation_sparsity)
        # self.activation_sparsity.clear()
        # print('layer2_1_out', out.size())

        # torch.save(out, './checkpoint_Cifar100_RK23_resnet/layer2_1_featuremap_out.pth.tar')
        # start = time.time()
        out = self.layer2_2(out)
        # self.activation_sparsity.append(out)
        # out = topk.drop_hw(out, density)
        # self.activation_sparsity.append(out)
        # self.cos_similarity('layer2_2', self.activation_sparsity)
        # self.activation_sparsity.clear()
        # # end = time.time()
        # print("ode_layer2_time:", end - start)
        # print('layer2_2_out', out.size())

        out = self.layer3_1(out)
        # self.activation_sparsity.append(out)
        # out = topk.drop_hw(out, density)
        # self.activation_sparsity.append(out)
        # self.cos_similarity('layer3_1', self.activation_sparsity)
        # self.activation_sparsity.clear()
        # print('layer3_1_out', out.size())

        # torch.save(out, './checkpoint_Cifar100_RK23_resnet/layer3_1_featuremap_out.pth.tar')
        # start = time.time()
        out = self.layer3_2(out)
        # self.activation_sparsity.append(out)
        # out = topk.drop_hw(out, density)
        # self.activation_sparsity.append(out)
        # self.cos_similarity('layer3_2', self.activation_sparsity)
        # self.activation_sparsity.clear()
        # # end = time.time()
        # print("ode_layer3_time:", end - start)
        # print('layer3_2_out', out.size())

        out = self.layer4_1(out)
        # self.activation_sparsity.append(out)
        # out = topk.drop_hw(out, density)
        # self.activation_sparsity.append(out)
        # self.cos_similarity('layer4_1', self.activation_sparsity)
        # self.activation_sparsity.clear()
        # torch.save(out, './checkpoint_Cifar100_RK23_resnet/layer4_1_featuremap_out.pth.tar')
        # print('layer4_1_out', out.size())

        # start = time.time()
        out = self.layer4_2(out)
        # self.activation_sparsity.append(out)
        # out = topk.drop_hw(out, density)
        # self.activation_sparsity.append(out)
        # self.cos_similarity('layer4_2', self.activation_sparsity)
        # self.activation_sparsity.clear()
        # # end = time.time()
        # print("ode_layer4_time:", end - start)
        # print('layer4_2_out', out.size())


        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # print('Linear_size', out.size())
        return out

def ResNet18(ODEBlock):
      return ResNet(BasicBlock, [2,2,2,2], ODEBlock_ = ODEBlock)

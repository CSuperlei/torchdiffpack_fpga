'''
Author: CSuperlei
Date: 2022-03-21 13:22:07
LastEditTime: 2022-03-21 13:44:00
Description: 
'''
import torch
import numpy as np


def cos_similarity(origin_array, pru_array):
    A = origin_array.reshape(-1).detach().cpu().numpy()
    B = pru_array.reshape(-1).detach().cpu().numpy()
    num = float(A.dot(B)) #若为行向量则 A * B.T
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    cos = num / denom #余弦值
        # cos = 0.5 + 0.5 * cos #归一化(0,1)
    if cos > 1.0:
        cos = 1.0
        # print(layer_name, cos)
    return cos

if __name__ == '__main__':
    origin = torch.load('./layer1_1_bb1_conv1_out.pth.tar')
    pru = torch.load('./layer1_1_bb1_conv1_pru_out.pth.tar')
    res = cos_similarity(origin, pru)
    print(res)

'''
Author: CSuperlei
Date: 2022-03-22 13:17:23
LastEditTime: 2022-03-25 12:13:14
Description: 
'''
""" Lotka volterra DDE """

from pylab import array, linspace, subplots
from pyparsing import rest_of_line
# from TorchDiffEqPack.odesolver import odesolve
from ode_solver import odesolve
import torch.nn as nn
import torch
import scipy as sci
import numpy as np
from matplotlib import pyplot as plt
import xlrd
import xlwt


class Func(nn.Module):
    def __init__(self):
        super(Func, self).__init__()
        self.delay = 0.2
        # self.weight = nn.Parameter(torch.ones(1))
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))
        self.weight4 = nn.Parameter(torch.ones(1))
    def forward(self, t, Y):
        x, y = Y[0], Y[1]
        xd, yd = x, y
        # a1 = self.weight * x * (1 - yd)
        a1 = x * (self.weight1 - self.weight2 * yd)
        # a2 = - self.weight * y * (1 - xd)
        a2 = - y * (self.weight3 - self.weight4 * xd)
        out = torch.cat((a1, a2), 0)
        return out


class History(nn.Module):
    def __init__(self, init_y):
        super(History, self).__init__()
        self.w = nn.Parameter(torch.from_numpy(np.array(init_y)).float())
    def forward(self, t):
        return self.w

func = Func()

time_span = np.linspace(0.0, 10.0, 1000)  # 20 orbital periods and 500 points
t_list = time_span.tolist()

# configure training options
options = {}
options.update({'method': 'RK23'})
options.update({'t0': 0.0})
options.update({'t1': 10.0})
options.update({'h': None})
options.update({'rtol': 1e-3})
options.update({'atol': 9e-4})
options.update({'print_neval': False})
options.update({'neval_max': 1000000})
options.update({'safety': None})
options.update({'t_eval':t_list})
# options.update({'dense_output':True})
options.update({'interpolation_method':'cubic'})

res_trajetory = []
for i in range(6):
    history = History([1,i+1])
    out = odesolve(func, history(0.0), options)
    out = out.data.cpu().numpy()
    print(out)
    res_trajetory.append(out)

res_trajetory = np.array(res_trajetory)

### draw pic
fig, ax = plt.subplots()
for i in range(6):
    ax.plot(res_trajetory[i][:,0], res_trajetory[i][:,1])
    
plt.savefig('./volterra_res.png')
plt.show()


wl = xlwt.Workbook('volterra.xls')
row = 0
col = 0
ws = wl.add_sheet('test') 
ws_shape = res_trajetory.shape

for i in range(ws_shape[0]):
    row = 0
    col = i * 2
    for j in range (ws_shape[1]):
        ws.write(row, col, float(res_trajetory[i][j][0]))
        ws.write(row, col + 1, float(res_trajetory[i][j][1]))
        row += 1

wl.save('./voltera_our.xls')

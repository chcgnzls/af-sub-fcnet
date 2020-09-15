import sys, os
import numpy as np
import torch
from PIL import Image

import config.conf as conf
from scripts.fcNet import build_net

x = conf.data['af_channel'].ravel()[:, None]
y = np.vstack([conf.data['r_channel'].ravel(),  conf.data['b_channel'].ravel(),  conf.data['g_channel'].ravel()]).T
x = torch.tensor(x).type(conf.dtype)
y = torch.tensor(y).type(conf.dtype)

net = build_net(conf.dtype)
solver = torch.optim.Adam(net.parameters())
loss_list = None

if loss_list is None:
  loss_list = [float(0)]

for t in range(4001):
  yhat = net(x)
  l1_reg = l2_reg = 0
  for name, param in net.named_parameters():
    if "weight" in name:
      l1_reg += param.norm(1)
      l2_reg += param.norm(2) ** 2

  loss = torch.nn.functional.mse_loss(yhat, y) + conf.reg_lambda * (conf.reg_alpha * l1_reg + (1 - conf.reg_alpha) / 2 * l2_reg)
  loss_list.append(loss.item())
  
  if t % 100 == 0:
    print("{}: {:.4f}".format(t, loss.item()))
  if t % 500 == 0:
    tmp = (y - yhat).detach().cpu().numpy()
    tmp[tmp < 0] = 0
    tmp = np.array([tmp[:, i].reshape(1440, 1920) for i in [0, 1, 2]])
    Image.fromarray(np.uint8(tmp).transpose(1, 2, 0)).save(conf.tmpdir + str(t) + conf.ftype)
    print(conf.tmpdir + str(t) + conf.ftype + ' saved!')

  loss.backward()
  solver.step()
  solver.zero_grad()

  if abs(loss_list[t] - loss_list[t + 1]) < 1e-4 and conf.tolbreak:
    print('tol break!!')
    print("%d: %.4f" % (t, loss.item()))
    tmp = (y - yhat).detach().cpu().numpy()
    tmp[tmp < 0] = 0
    tmp = np.array([tmp[:, i].reshape(1440, 1920) for i in [0, 1, 2]])
    Image.fromarray(np.uint8(tmp).transpose(1, 2, 0)).save(conf.tmpdir + str(t) + conf.ftype)
    print(conf.tmpdir + str(t) + conf.ftype + ' saved!')
    break 
    
tmp = (y - yhat).detach().cpu().numpy()
tmp[tmp < 0] = 0
tmp = np.array([tmp[:, i].reshape(1440, 1920) for i in [0, 1, 2]])
Image.fromarray(np.uint8(tmp).transpose(1, 2, 0))

z = np.zeros((1440, 1920))
r = np.stack((tmp.transpose(1, 2, 0)[:, :, 0], z, z)).transpose(1, 2, 0)
g = np.stack((z, tmp.transpose(1, 2, 0)[:, :, 1], z)).transpose(1, 2, 0)
b = np.stack((z, z, tmp.transpose(1, 2, 0)[:, :, 2])).transpose(1, 2, 0)
Image.fromarray(np.uint8(r)).save(conf.r_name + conf.ftype)
Image.fromarray(np.uint8(g)).save(conf.g_name + conf.ftype)
Image.fromarray(np.uint8(b)).save(conf.b_name + conf.ftype)
Image.fromarray(np.uint8(tmp).transpose(1, 2, 0)).save('overlay' + conf.ftype)

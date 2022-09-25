from Unet import UNET as un
from Satnet import satnet as sn

import torch
import numpy as np

m1 = un.UNet(retain_dim=True)
m2 = sn.SatNet()

x = torch.rand((1,3,256,256))

m1(x)
m2(x)

print(sum(p.numel() for p in m1.parameters() if p.requires_grad))
print(sum(p.numel() for p in m2.parameters() if p.requires_grad))

from data_handling.dataloader import load_one

ms = []
N = 2000
ind = np.random.randint(5108, size=N)
for i in range(N):
	if i % 50 == 0:
		print(str(np.round(i/N, 3) * 100) + "%")
	x, y = load_one(ind[i])
	ms.append(torch.sum(y/255))

print(np.average(ms))

from Unet import UNET as un
from Satnet import satnet as sn

import torch

m1 = un.UNet(retain_dim=True)
m2 = sn.SatNet()

x = torch.rand((1,3,256,256))

m1(x)
m2(x)

print(sum(p.numel() for p in m1.parameters() if p.requires_grad))
print(sum(p.numel() for p in m2.parameters() if p.requires_grad))

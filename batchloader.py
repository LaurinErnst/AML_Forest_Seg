from re import T
from sys import displayhook
import pandas as pd
import os
from PIL import Image
import numpy as np
from UNet import UNet
import torch

def batchloader(batchsize = None, batch = None):
    if batchsize == None and batch != None:
        directory = 'data/images/'

        for i in batch:
            f = directory + str(i) + ".jpg"
            im = Image.open(f)
            data = np.array(im.getdata()).T

            # r = data[:,0]
            # r = r.reshape(256, 256).astype(np.uint8)

            # g = data[:,1]
            # g = g.reshape(256, 256).astype(np.uint8)

            # b = data[:,2]
            # b = b.reshape(256, 256).astype(np.uint8)
            
            # im = Image.fromarray(r)
            # im.save("r.jpeg")
            # im = Image.fromarray(g)
            # im.save("g.jpeg")
            # im = Image.fromarray(b)
            # im.save("b.jpeg")
            # print(r)
            
            return data


data = batchloader(None, [1])
data = torch.tensor(data)

Test = UNet()
print(Test.forward(data))
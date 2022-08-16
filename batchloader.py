import os
from PIL import Image
import numpy as np
import torch

def batchloader(batchsize = None, batch = None):
    if batchsize == None and batch != None:
        directory = 'data/images/'
        j = 0
        for i in batch:
            
            f = directory + str(i) + ".jpg"
            im = Image.open(f)
            im_data = np.array(im.getdata()).T

            if j == 0:
                data = np.array([im_data.reshape(3, 256, 256)])
                j += 1
            else:
                im_data = im_data.reshape(1, 3, 256, 256)
                data = np.concatenate((data, im_data))
        return torch.tensor(data)

    if batchsize != None and batch == None:

        batch = np.random.randint(5108, size = batchsize)
        print(batch)
        directory = 'data/images/'
        j = 0
        for i in batch:
            
            f = directory + str(i) + ".jpg"
            im = Image.open(f)
            im_data = np.array(im.getdata()).T

            if j == 0:
                data = np.array([im_data.reshape(3, 256, 256)])
                j += 1
            else:
                im_data = im_data.reshape(1, 3, 256, 256)
                data = np.concatenate((data, im_data))
        return torch.tensor(data)

    if batchsize == None and batch == None:
        raise Exception("Weder batchsize noch batch deklariert!")
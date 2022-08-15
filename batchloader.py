from sys import displayhook
import pandas as pd
import os
from PIL import Image

# assign directory
directory = 'data/images'
i = 0
data = []
log = open("log.txt", "w")

print("starting")
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        im = Image.open(f)
        pixel_values = list(im.getdata())
        data.append([pixel_values])
        i += 1

def batchloader(batchsize = None, batch = None):
    if batchsize == None and batch != None:
        directory = 'data/images/'

        for i in batch:
            f = directory + str(i) + "jpg"
            im = Image.open(f)
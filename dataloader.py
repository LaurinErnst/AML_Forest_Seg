from cmath import pi
from copyreg import pickle
from sys import displayhook
import pandas as pd
import os
from PIL import Image
from torch import pixel_shuffle

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
        if (i % 500 == 0):
            print(i)
        log.write(str(i))
log.close()
data = pd.DataFrame(data)
json = data.to_json()

print("writing JSON file")

f = open("im_data.json", "wr")
f.write(json)
f.close()




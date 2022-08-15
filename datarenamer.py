import os
from sys import displayhook
import pandas as pd

# assign directory
im_directory = 'data/images/'
mask_directory = 'data/masks/'



df = pd.read_csv('data/meta_data.csv')

displayhook(df)

n = 5107

for i in range(n):
    im_name = df["image"][i]
    mask_name = df["mask"][i]
    os.rename(im_directory + im_name, im_directory + str(i) + ".jpg")
    os.rename(mask_directory + mask_name, mask_directory + str(i) + ".jpg")

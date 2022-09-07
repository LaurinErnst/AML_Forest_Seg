import numpy as np
import os
from PIL import Image
import numpy as np
import torch


mask_directory = "data/masks/"


f_mask = os.path.join(mask_directory + "1.jpg")
mask = Image.open(f_mask)
mask = mask.convert("L")

# extract data into numpy array
mask_data = np.array(mask.getdata()).T / 255

data_mask = np.array([mask_data.reshape(1, 256, 256)])

print(data_mask)

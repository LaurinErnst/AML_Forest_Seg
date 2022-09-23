#from data_handling.dataloader import load_one
import numpy as np
from skimage.segmentation import felzenszwalb
from scipy import ndimage
from PIL import Image
import os
import torch

def naive_seg(i):
    #load images in color and in grayscale
    image = np.array(Image.open(os.path.join("../data/images/",str(i)+".jpg")))
    img_gray = np.array(Image.open("../data/images/"+str(i)+".jpg").convert("L"))

    #smoothen the image
    img_gray_avg = ndimage.uniform_filter(img_gray,size=8,mode="mirror")

    #use felzeszwalb algorithm to for segmentation
    segments = felzenszwalb(img_gray, scale=70, sigma=0.5, min_size=50)
    mask=np.zeros((256,256))

    #loop over segments
    for j in range(len(np.unique(segments))):
        bool_array = segments==j
        single_segment=img_gray_avg[bool_array]
        single_segment_col=image[bool_array]

        #calculate average rgb values in each segment and the difference to green
        avg_red=np.average(single_segment_col[:,0])
        avg_gr=np.average(single_segment_col[:,1])
        avg_blue=np.average(single_segment_col[:,2])
        diff=min(avg_gr-avg_red,avg_gr-avg_blue)

        #calculate average brightness and standard deviation per segment
        brightness = np.average(single_segment)
        single_segment=single_segment.flatten()
        single_segment=single_segment[single_segment!=0]
        std=np.std(single_segment)

        #conditions for forest classification
        if std>3.2  and brightness<120 and diff>-20:
            mask[bool_array]=255
    mask=torch.tensor(np.reshape(np.uint8(mask),(1,256,256)))
    return mask

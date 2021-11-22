import argparse
import os
import sys
import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


def readDepth(image_dir, src_image_shape):
    '''
    image_dir: str
    src_image_shape: tuple

    raw inference images are garscale single channle image
    return h x w in int32 as depth prediction
    '''
    pil_image = cv2.imread(image_dir, cv2.IMREAD_UNCHANGED)
    src_size = (src_image_shape[1], src_image_shape[0])
    pil_image  = cv2.resize(pil_image, src_size)
    img = np.array(pil_image, np.int32, copy=False)
    return img

def cropBboxDepth(depth_img, sub_box, obj_box):
    '''
    box: [x_min, y_min, x_max, y_max]

    return two bbox depth patches
    '''
    sub_depth = depth_img[sub_box[1]:sub_box[3], sub_box[0]:sub_box[2]]
    obj_depth = depth_img[obj_box[1]:obj_box[3], obj_box[0]:obj_box[2]]
    return sub_depth, obj_depth


depth_dir = "/data/xyao/adabins_depth/depth_results/train/"
src_dir = "/data/xyao/sg_dataset/sg_train_images/"

all_files = glob.glob(os.path.join(src_dir, "*"))
for index, f in tqdm(enumerate(all_files)):
    if index > 1:
        break
    src_image = cv2.imread(f)
    sub_box = [1,1,100,100]
    obj_box = [1,1,100,100]

    fname = f.split('/')[-1]
    depth_im_name = os.path.join(depth_dir, fname[:-4]+'.png')
    depth_img = readDepth(depth_im_name, src_image.shape)
    sub_depth, obj_depth = cropBboxDepth(depth_img, sub_box, obj_box)

    plt.figure()

    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,4) 

    axarr[0].imshow(sub_depth)
    axarr[1].imshow(obj_depth)
    axarr[2].imshow(depth_img)
    axarr[3].imshow(src_image)
    plt.savefig('./d.png')
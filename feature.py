import os
import cv2
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw

import torch
import torchvision
from vision.torchvision.models.detection import faster_rcnn

# data dir
TrainLabelFile = '/media/DataDrive/datasets/VRD/sg_dataset/sg_train_annotations.json'
TrainImageDir = '/media/DataDrive/datasets/VRD/sg_dataset/sg_train_images'
saveDir = '/media/DataDrive/datasets/VRD/maskrcnn_res/feature'

# read labels info
with open(TrainLabelFile, 'r') as f:
	TrainLabel = json.load(f)

# device
device = torch.device('cuda:0')

# pre-trained mask rcnn
model = faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# run on mask rcnn to extract feature 
for current_image_dict in tqdm(TrainLabel):
    ori_img = Image.open(os.path.join(TrainImageDir, current_image_dict['filename']))
    img = np.array(ori_img).transpose(2,0,1) / 255
    img = torch.Tensor(img).unsqueeze(0)    
    img = img.cuda()
    features_dict = model(img)
    for key in features_dict.keys():
        feature = np.squeeze(results[key].detach().cpu().numpy()).transpose(1,2,0)
        feature = np.max(feature, axis=2)
        feature = (255 * feature / np.max(feature)).astype(np.int32)
        img = Image.fromarray(np.uint8(feature))
        img.save(os.path.join(saveDir, key, current_image_dict['filename']))


	

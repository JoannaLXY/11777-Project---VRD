import os
import cv2
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# draw res
def draw_bbox_image(img, bboxes, scores, conf=0.3):
    for i in range(len(scores)):
        if scores[i] >= conf:        
            bbox = bboxes[i]
            draw = ImageDraw.Draw(img)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            draw.line((x1, y1, x2, y1), width=3, fill='red')
            draw.line((x1, y1, x1, y2), width=3, fill='red')
            draw.line((x2, y2, x2, y1), width=3, fill='red')
            draw.line((x2, y2, x1, y2), width=3, fill='red')


# data dir
TrainLabelFile = '/media/DataDrive/datasets/VRD/sg_dataset/sg_train_annotations.json'
TrainImageDir = '/media/DataDrive/datasets/VRD/sg_dataset/sg_train_images'
saveDir = '/media/DataDrive/datasets/VRD/maskrcnn_res/'

# read labels info
with open(TrainLabelFile, 'r') as f:
	TrainLabel = json.load(f)

# device
device = torch.device('cuda:0')

# pre-trained mask rcnn
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

'''
file_no = 1
file_dir = os.path.join(TrainImageDir, TrainLabel[file_no]['filename'])
ori_img = Image.open(os.path.join(TrainImageDir, TrainLabel[file_no]['filename']))
img = np.array(ori_img).transpose(2,0,1) / 255. 
img = torch.Tensor(img).unsqueeze(0)
results = model(img)
bboxes = results[0]['boxes'].detach().numpy()
scores = results[0]['scores'].detach().numpy()
draw_bbox_image(ori_img, bboxes, scores)
ori_img.save(os.path.join(saveDir, 'detection', TrainLabel[file_no]['filename']))
'''
# run on mask rcnn 
for current_image_dict in tqdm(TrainLabel):
    ori_img = Image.open(os.path.join(TrainImageDir, current_image_dict['filename']))
    img = np.array(ori_img).transpose(2,0,1) / 255
    img = torch.Tensor(img).unsqueeze(0)    
    img = img.cuda()
    results = model(img)
    bboxes = results[0]['boxes'].detach().cpu().numpy()
    scores = results[0]['scores'].detach().cpu().numpy()
    draw_bbox_image(ori_img, bboxes, scores)
    ori_img.save(os.path.join(saveDir, 'detection', current_image_dict['filename']))
	

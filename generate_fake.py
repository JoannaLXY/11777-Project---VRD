import os
import random
from os.path import exists, join, isfile, dirname, abspath, split
import glob
import cv2
import json
import pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict

"""parse annotations to build a map with key = img_name"""
def parse_annos(annos):
    new_annos = {}
    for anno in annos:
        new_annos[anno["filename"]] = anno
    return new_annos

def parse_bbox(box):
    # reference: https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/utils/general.py#L153
    # input {y, x, w, h}, output (x1, y1, x2, y2)
    out = [0,0,0,0]
    out[0] = box["x"]  # top left x
    out[1] = box["y"]  # top left y
    out[2] = box["x"] + box["w"]  # bottom right x
    out[3] = box["y"] + box["h"]  # bottom right y
    return out

"""get key name from annotation"""
def get_key_name(rela, objects):
    sub, pred, obj = rela["text"]
    # get name + bbox for subject
    sub_id = rela['objects'][0]
    sub_box = parse_bbox(objects[sub_id]['bbox'])
    # ymin ymax xmin xmax
    sub_str = "_".join([str(int(sub_box[1])), str(int(sub_box[3])), str(int(sub_box[0])), str(int(sub_box[2]))])
    # get name + bbox for object
    obj_id = rela['objects'][1]
    obj_box = parse_bbox(objects[obj_id]['bbox'])
    # ymin ymax xmin xmax
    obj_str = "_".join([str(int(obj_box[1])), str(int(obj_box[3])), str(int(obj_box[0])), str(int(obj_box[2]))])
    # key name: subid + objid
    key_name = "_".join([sub, sub_str, obj, obj_str])
    return key_name

path_to_dataset = "/home/xuhuah/11777-Project-VRD/nmp/dataset/vrd/sg_dataset"
path_to_result = "/home/xuhuah/11777-Project-VRD/result.pkl"
gt_imgs = glob.glob(join(path_to_dataset, "sg_test_images", "*"))
gt_annos = json.load(open(join(path_to_dataset, "sg_test_annotations.json")))
gt_annos = parse_annos(gt_annos)

fake_outputs = {}
for img_name, anno in gt_annos.items():
    relations = anno["relationships"]
    objects = anno["objects"]
    # pick a random subset from it
    picked_relas = random.sample(relations, 10)
    pairs = defaultdict(list)
    for rela in picked_relas:
        sub, pred, obj = rela["text"]
        key_name = get_key_name(rela, objects)
        # confidence score == 1.0 for GT
        pairs[key_name].append((pred, 1.0))
    fake_outputs[img_name] = pairs

with open(path_to_result, 'wb') as f:
    pickle.dump(fake_outputs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved to %s"%(path_to_result))
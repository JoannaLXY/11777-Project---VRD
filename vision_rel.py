# script to analyze unimodality - vision
"""
For each relationship, find the box for related objects, calculate IoU & distance
count for each subject-object relationship,
what is the average IoU & distance over all cases
"""
from dataset import VRDDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torchvision.ops.boxes as bops
import cv2
from collections import defaultdict

import tqdm

def parse_bbox(box):
    # reference: https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/utils/general.py#L153
    # input {y, x, w, h}, output (x1, y1, x2, y2)
    out = [0,0,0,0]
    out[0] = box["x"]  # top left x
    out[1] = box["y"]  # top left y
    out[2] = box["x"] + box["w"]  # bottom right x
    out[3] = box["y"] + box["h"]  # bottom right y
    return out

def vis_bbox(box, img):
    vis_img = img.copy()
    vis_img = cv2.rectangle(vis_img,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 5)
    return vis_img

DATA_PATH = "/Users/Mr.King/sg_dataset"

trainset = VRDDataset(root_dir=DATA_PATH, mode="train")
testset = VRDDataset(root_dir=DATA_PATH, mode="test")

batch_size = 1
assert batch_size == 1 # only support batch size == 1
dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True,
                        collate_fn=trainset.collate_batch)

progress_bar = tqdm.tqdm(total=len(dataloader), leave=True)

target_subobjs = ["cars-line", "photo-frame", "flags-poles", "clothes-closet", "glove-hands", "tshirt-man", "rug-flowers", "cabinets-wood", "wallpaper-wall",
                  "hat-backpack", "arms-shoes", "pillows-table"]
target_rels = ["are on a", "decorated with a", "is covered in", "drive on", "contain",
               "smaller than", "looking at a", "on side of", "standing besides"]
# target_subobjs = ["the x-yellow", "boy-white", "boats-each other"]
# target_rels = []

relation_iou = defaultdict(list) # store ious for each relation
subobj_iou = defaultdict(list) # store ious for each sub-obj pair
triplet_iou = defaultdict(list) # store ious for each triplet
cnt = 0
for data in dataloader:
    cnt += 1
    img = data[0]["img"]
    anno = data[0]["anno"]
    relationships = anno["relationships"]
    objects = anno["objects"]
    for rel in relationships:
        subject, relation, object = rel["text"]
        subject_id, object_id = rel["objects"][0], rel["objects"][1]
        assert subject in objects[subject_id]["names"]
        assert object in objects[object_id]["names"]
        subject_box = parse_bbox(objects[subject_id]["bbox"])
        object_box = parse_bbox(objects[object_id]["bbox"])
        box1 = torch.tensor([subject_box], dtype=torch.float)
        box2 = torch.tensor([object_box], dtype=torch.float)
        iou = bops.box_iou(box1, box2)
        subobj_key = "%s-%s"%(subject, object)
        triplet_key = "%s-%s-%s"%(subject, relation, object)
        subobj_iou[subobj_key].append(iou)
        triplet_iou[triplet_key].append(iou)
        relation_iou[relation].append(iou)
        # save example images
        if subobj_key in target_subobjs:
            # draw bbox
            vis_img1 = vis_bbox(subject_box, img)
            vis_img1 = vis_bbox(object_box, vis_img1)
            cv2.imwrite("examples/%s_%d.png"%(subobj_key, cnt), vis_img1)
        if relation in target_rels:
            # draw bbox
            vis_img2 = vis_bbox(subject_box, img)
            vis_img2 = vis_bbox(object_box, vis_img2)
            cv2.imwrite("examples/%s_%d.png" % (relation, cnt), vis_img2)
    progress_bar.update()
progress_bar.close()

# sort on average for each relation and for each sub-obj pair
subobj_iou_mean, subobj_iou_var = {}, {}
for key in subobj_iou:
    subobj_iou_mean[key] = np.mean(subobj_iou[key])
    subobj_iou_var[key] = np.var(subobj_iou[key])
sorted_subobj_iou = dict(sorted(subobj_iou_mean.items(), key=lambda item: item[1], reverse=True))
# convert to strings
sorted_subobj_strs = []
for key in sorted_subobj_iou:
    curr_str = "%s: %f, %f\n"%(key, subobj_iou_mean[key], subobj_iou_var[key])
    sorted_subobj_strs.append(curr_str)

rel_iou_mean, rel_iou_var = {}, {}
for key in relation_iou:
    rel_iou_mean[key] = np.mean(relation_iou[key])
    rel_iou_var[key] = np.var(relation_iou[key])
sorted_relation_iou = dict(sorted(rel_iou_mean.items(), key=lambda item: item[1], reverse=True))
# convert to strings
sorted_relation_strs = []
for key in sorted_relation_iou:
    curr_str = "%s: %f, %f\n"%(key, rel_iou_mean[key], rel_iou_var[key])
    sorted_relation_strs.append(curr_str)

# for triplets
triplet_iou_mean, triplet_iou_var = {}, {}
for key in triplet_iou:
    triplet_iou_mean[key] = np.mean(triplet_iou[key])
    triplet_iou_var[key] = np.var(triplet_iou[key])
sorted_triplet_iou = dict(sorted(triplet_iou_mean.items(), key=lambda item: item[1], reverse=True))
# convert to strings
sorted_triplet_strs = []
for key in sorted_triplet_iou:
    curr_str = "%s: %f, %f\n"%(key, triplet_iou_mean[key], triplet_iou_var[key])
    sorted_triplet_strs.append(curr_str)
# write subobj / relation with highest iou
f = open("subobj.txt", "w+")
f.writelines(sorted_subobj_strs)
f.close()
f = open("rel.txt", "w+")
f.writelines(sorted_relation_strs)
f.close()
f = open("trip.txt", "w+")
f.writelines(sorted_triplet_strs)
f.close()

print("vision-based relationship analysis done.")


"""
some issues in dataset:
1. Can have multiple names matched to one object ID
"""
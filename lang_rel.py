# script to analyze unimodality - language
"""
For each image, we find the combinations of all triplets and
increase their co-occurance by 1
check which two triplets have highest co-occurance at the end
"""
import itertools

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

DATA_PATH = "/Users/Mr.King/sg_dataset"

trainset = VRDDataset(root_dir=DATA_PATH, mode="train")
testset = VRDDataset(root_dir=DATA_PATH, mode="test")

batch_size = 1
assert batch_size == 1 # only support batch size == 1
dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True,
                        collate_fn=trainset.collate_batch)

progress_bar = tqdm.tqdm(total=len(dataloader), leave=True)

cooccurance = defaultdict(int) # count co-occurance between two relationships
cnt = 0
for data in dataloader:
    cnt += 1
    img = data[0]["img"]
    anno = data[0]["anno"]
    relationships = anno["relationships"]
    curr_triplets = []
    for rel in relationships:
        triplet = "-".join(rel["text"])
        curr_triplets.append(triplet)
    # unique combine relationship triplets
    all_cooccurance = itertools.combinations(curr_triplets,2)
    for tri1, tri2 in all_cooccurance:
        tri_tri_name = "%s&%s"%(tri1, tri2)
        cooccurance[tri_tri_name] += 1

    progress_bar.update()
progress_bar.close()

# sort on average for each co-occurance
sorted_cooccurance = dict(sorted(cooccurance.items(), key=lambda item: item[1], reverse=True))
# convert to strings
sorted_cooccurance_strs = []
for key in sorted_cooccurance:
    # only store co-occurance > 50
    if sorted_cooccurance[key] < 50:
        break
    curr_str = "%s: %d\n"%(key, sorted_cooccurance[key])
    sorted_cooccurance_strs.append(curr_str)

# write subobj / relation with highest iou
f = open("triplet.txt", "w+")
f.writelines(sorted_cooccurance_strs)
f.close()

print("language-based relationship analysis done.")


"""
some issues in dataset:
1. Can have multiple names matched to one object ID
"""
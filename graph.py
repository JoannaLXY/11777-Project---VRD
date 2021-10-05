# script to analyze graph modality
from dataset import VRDDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm

DATA_PATH = "/Users/Mr.King/sg_dataset"

trainset = VRDDataset(root_dir=DATA_PATH, mode="train")
testset = VRDDataset(root_dir=DATA_PATH, mode="test")

batch_size = 1
assert batch_size == 1 # only support batch size == 1
dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True,
                        collate_fn=trainset.collate_batch)

progress_bar = tqdm.tqdm(total=len(dataloader), leave=True)

cnt = 0
for data in dataloader:
    cnt += 1
    img = data[0]["img"]
    anno = data[0]["anno"]
    progress_bar.update()
progress_bar.close()

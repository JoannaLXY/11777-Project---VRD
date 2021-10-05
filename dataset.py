# dataloader for VRD
import numpy as np
import os
from os.path import exists, join, isfile, dirname, abspath, split
import torch
import torch.utils.data as torch_data
import glob
import json
import cv2

class VRDDataset(torch_data.Dataset):
    def __init__(self, root_dir, mode="train"):
        self.mode = mode
        self.dataset_path = root_dir

        if mode == "train":
            self.imgs = glob.glob(join(root_dir, "sg_train_images", "*"))
            self.annos = json.load(open(join(root_dir, "sg_train_annotations.json")))
        else:
            self.imgs = glob.glob(join(root_dir, "sg_test_images", "*"))
            self.annos = json.load(open(join(root_dir, "sg_test_annotations.json")))
        # parse annotations to build map with key as img_name
        self.annos = self.parse_annos(self.annos)
        assert len(self.annos) == len(self.imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        filename = os.path.basename(self.imgs[idx])
        img = cv2.imread(self.imgs[idx])
        anno = self.annos[filename]
        input_data = {
            "img": img,
            "anno": anno
        }
        return input_data

    def parse_annos(self, annos):
        new_annos = {}
        for anno in annos:
            new_annos[anno["filename"]] = anno
        return new_annos

    def collate_batch(self, batch):
        return batch

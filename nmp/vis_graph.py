from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from modules import *
from eval_metrics import *
from utils import *
from DataLoader import *

import cv2
from tqdm import tqdm

import networkx as nx
import matplotlib.pyplot as plt

def vis_graph(gt_roidb, pred_roidb):
    predicates = ["on", "wear", "has", "next to", "sleep next to", "sit next to", "stand next to", "park next", "walk next to", "above", "behind", "stand behind", "sit behind", "park behind", "in the front of", "under", "stand under", "sit under", "near", "walk to", "walk", "walk past", "in", "below", "beside", "walk beside", "over", "hold", "by", "beneath", "with", "on the top of", "on the left of", "on the right of", "sit on", "ride", "carry", "look", "stand on", "use", "at", "attach to", "cover", "touch", "watch", "against", "inside", "adjacent to", "across", "contain", "drive", "drive on", "taller than", "eat", "park on", "lying on", "pull", "talk", "lean on", "fly", "face", "play with", "sleep on", "outside of", "rest on", "follow", "hit", "feed", "kick", "skate on"]
    objects = ["person", "sky", "building", "truck", "bus", "table", "shirt", "chair", "car", "train", "glasses", "tree", "boat", "hat", "trees", "grass", "pants", "road", "motorcycle", "jacket", "monitor", "wheel", "umbrella", "plate", "bike", "clock", "bag", "shoe", "laptop", "desk", "cabinet", "counter", "bench", "shoes", "tower", "bottle", "helmet", "stove", "lamp", "coat", "bed", "dog", "mountain", "horse", "plane", "roof", "skateboard", "traffic light", "bush", "phone", "airplane", "sofa", "cup", "sink", "shelf", "box", "van", "hand", "shorts", "post", "jeans", "cat", "sunglasses", "bowl", "computer", "pillow", "pizza", "basket", "elephant", "kite", "sand", "keyboard", "plant", "can", "vase", "refrigerator", "cart", "skis", "pot", "surfboard", "paper", "mouse", "trash can", "cone", "camera", "ball", "bear", "giraffe", "tie", "luggage", "faucet", "hydrant", "snowboard", "oven", "engine", "watch", "face", "street", "ramp", "suitcase"]
    pred_roidb = pred_roidb["pred_roidb"]

    for i in range(len(gt_roidb)):
        gt = gt_roidb[i]
        pred = pred_roidb[i]
        img_path = gt["image"]
        rgb_img = cv2.imread(img_path)
        G_gt = nx.DiGraph(directed=True) # graph for GT
        G_pred = nx.DiGraph(directed=True) # graph for pred
        gt_edge_labels = {}
        pred_edge_labels = {}

        # draw bounding boxes
        assert gt["sub_box_gt"].shape[0] == gt["obj_box_gt"].shape[0]
        num_pairs = gt["sub_box_gt"].shape[0]
        for j in range(num_pairs):
            # xmin, ymin, xmax, ymax
            sub_box = gt["sub_box_gt"][j]
            obj_box = gt["obj_box_gt"][j]
            sub_name = objects[int(gt["sub_gt"][j])]
            obj_name = objects[int(gt["obj_gt"][j])]
            color = list(np.random.random(size=3) * 256)
            cv2.rectangle(rgb_img, (int(uni_box[0]), int(uni_box[1])), (int(uni_box[2]), int(uni_box[3])), color, 2)
            # draw edge in GT graph
            G_gt.add_edges_from([(sub_name, obj_name)], weight=1.0)
            gt_edge_labels[(sub_name, obj_name)] = predicates[int(gt['rela_gt'][j])] + ": 1.0"
            # draw edge in pred graph
            G_pred.add_edges_from([(sub_name, obj_name)], weight=pred['pred_rela_score'][j])
            pred_edge_labels[(sub_name, obj_name)] = predicates[int(pred['pred_rela'][j])] + ": " + "{0:.2f}".format(pred['pred_rela_score'][j])
        # draw graph
        pos1 = nx.spring_layout(G_gt)
        nx.draw_networkx_edge_labels(G_gt, pos1, edge_labels=gt_edge_labels)
        nx.draw(G_gt, pos1, node_size=1500)
        plt.show()
        pos2 = nx.spring_layout(G_pred)
        nx.draw_networkx_edge_labels(G_pred, pos2, edge_labels=pred_edge_labels)
        nx.draw(G_pred, pos2, node_size=1500)
        print()

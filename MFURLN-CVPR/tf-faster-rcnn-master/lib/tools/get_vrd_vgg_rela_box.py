from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,sys
import os.path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from model.test import test_net_vg
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
import argparse
import pprint
import numpy as np
import sys, os

import tensorflow as tf
from nets.vgg16 import vgg16
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
output_dir = 'dete_pred_vrd.npz'
save_path = '/data/xyao/sg_dataset/MFURLN/faster_rcnn/vrd_detected_box.npz'
model_path = '/data/xyao/sg_dataset/MFURLN/faster_rcnn/pretrain/vrd_vgg_pretrained.ckpt'
def read_roidb(roidb_path):
	''' python3 '''
	roidb_file = np.load(roidb_path, allow_pickle=True, encoding='latin1')
	key = list(roidb_file.keys())[0]
	roidb_temp = roidb_file[key]
	roidb = roidb_temp[()]
	return roidb

num_classes = 101
roidb = read_roidb('/data/xyao/sg_dataset/MFURLN/faster_rcnn/vrd_roidb.npz')
train_roidb = roidb['train_roidb']
val_roidb = roidb['val_roidb']
test_roidb = roidb['test_roidb']
N_train = len(train_roidb)
N_temp = np.int32(N_train/2)
print(N_temp)
train_roidb_temp = train_roidb[0:N_temp]
net = vgg16()
net.create_architecture("TEST", num_classes, tag = 'default', 
		                   anchor_scales=cfg.ANCHOR_SCALES, 
		                   anchor_ratios=cfg.ANCHOR_RATIOS)
variables = tf.global_variables()

saver = tf.train.Saver()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	saver.restore(sess, model_path)
	train_detected_box = test_net_vg(sess, net, train_roidb_temp, output_dir, num_classes, max_per_image=300, thresh=0.05)
	val_detected_box = test_net_vg(sess, net, val_roidb, output_dir, num_classes, max_per_image=300, thresh=0.05)
	test_detected_box = test_net_vg(sess, net, test_roidb, output_dir, num_classes, max_per_image=300, thresh=0.05)
vrd_detected_box = {'train_detected_box': train_detected_box,
			'val_detected_box': val_detected_box,
					'test_detected_box': test_detected_box}
np.savez(save_path, vrd_detected_box = vrd_detected_box)

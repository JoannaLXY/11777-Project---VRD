from __future__ import print_function
import numpy as np
import os
import ipdb
import time
from tqdm import tqdm
from utils import read_roidb, compute_iou_each
from collections import defaultdict
import pickle

def graph_npy2roidb(roidb, pred_probs, pred_cls, mode='pred', level='image', topk=False):
	'''
	function: process the pred_probs and pred_cls to the roidb format;
			  then the metric calculation functions can deal with them
	args:
		roidb: the ground truth roidb array of dict
		topk: get the top k highest predication
		pred_probs: the prediction probs of the predicate based on the input box pair
					shape: [N_GT_set, k]
		pred_cls: the prediction class of the predicate based on the input box pair
					shape: [N_GT_set, k]
		mode: 'pred' or 'rela'
	'''
	def _output2roidb(roidb_use, output, output_score, mode='pred'):
		if mode == 'pred':
			N_total = len(roidb_use['rela_gt'])
		else:
			N_total = len(roidb_use['rela_dete'])
		pred_rela = output[:N_total]
		pred_rela_score = output_score[:N_total]
		return pred_rela, pred_rela_score
	def _instance_output2roidb(start, roidb_use, output, output_score, mode='pred'):
		if mode == 'pred':
			N_total = len(roidb_use['rela_gt'])
		else:
			N_total = len(roidb_use['rela_dete'])
		pred_rela = pred_cls[start:(start+N_total)]
		pred_rela_score = pred_probs[start:(start+N_total)]
		start += N_total
		return start, pred_rela, pred_rela_score

	pred_roidb = []
	N_data = len(roidb)
	start = 0
	if mode == 'pred':
		for i in range(N_data):
			roidb_use = roidb[i]
			if level == 'instance':
				start, pred_rela, pred_rela_score = _instance_output2roidb(start, roidb_use, pred_cls, pred_probs, mode=mode)
			else:
				pred_rela, pred_rela_score = _output2roidb(roidb_use, pred_cls[i], pred_probs[i], mode=mode)
			pred_roidb_temp = {'pred_rela': pred_rela, 'pred_rela_score': pred_rela_score,
								'sub_box_dete': roidb_use['sub_box_gt'], 'obj_box_dete': roidb_use['obj_box_gt'],
								'sub_dete': roidb_use['sub_gt'], 'obj_dete': roidb_use['obj_gt']}
			pred_roidb.append(pred_roidb_temp)
	elif mode == 'rela':
		# train set
		if N_data > 1000:
			for i in range(N_data):
				roidb_use = roidb[i]
				if level == 'instance':
					start, pred_rela, pred_rela_score = _instance_output2roidb(start, roidb_use, pred_cls, pred_probs, mode=mode)
				else:
					pred_rela, pred_rela_score = _output2roidb(roidb_use, pred_cls[i], pred_probs[i], mode=mode)
				pred_roidb_temp = {'pred_rela': pred_rela, 'pred_rela_score': pred_rela_score,
									'sub_box_dete': roidb_use['sub_box_dete'], 'obj_box_dete': roidb_use['obj_box_dete'],
									'sub_dete': roidb_use['sub_dete'], 'obj_dete': roidb_use['obj_dete']}
				pred_roidb.append(pred_roidb_temp)
		else:
			for i in range(N_data):
				roidb_use = roidb[i]
				if level == 'instance':
					start, pred_rela, pred_rela_score = _instance_output2roidb(start, roidb_use, pred_cls, pred_probs, mode=mode)
				else:
					pred_rela, pred_rela_score = _output2roidb(roidb_use, pred_cls[i], pred_probs[i], mode=mode)
				sub_score = roidb_use['sub_score']
				obj_score = roidb_use['obj_score']
				sub_obj_score = np.log(sub_score) + np.log(obj_score)
				# sub_obj_score = np.zeros_like(obj_score)
				if topk:
					pred_rela_score = list(map(lambda i: sub_obj_score + pred_rela_score[:,i], range(pred_rela_score.shape[-1])))
					pred_rela_score = np.array(pred_rela_score).T
				else:
					pred_rela_score = pred_rela_score + sub_obj_score
				pred_roidb_temp = {'pred_rela': pred_rela, 'pred_rela_score': pred_rela_score,
									'sub_box_dete': roidb_use['sub_box_dete'], 'obj_box_dete': roidb_use['obj_box_dete'],
									# 'sub_dete': roidb_use['sub_dete']-1, 'obj_dete': roidb_use['obj_dete']-1}
									'sub_dete': roidb_use['sub_dete'], 'obj_dete': roidb_use['obj_dete']}
				pred_roidb.append(pred_roidb_temp)
	roidb_temp = {}
	roidb_temp['pred_roidb'] = pred_roidb
	return roidb_temp

def compute_overlap(det_bboxes, gt_bboxes):
    """
    Compute overlap of detected and ground truth boxes.

    Inputs:
        - det_bboxes: array (2, 4), 2 x [y_min, y_max, x_min, x_max]
            The detected bounding boxes for subject and object
        - gt_bboxes: array (2, 4), 2 x [y_min, y_max, x_min, x_max]
            The ground truth bounding boxes for subject and object
    Returns:
        - overlap: non-negative float <= 1
    """
    overlaps = []
    for det_bbox, gt_bbox in zip(det_bboxes, gt_bboxes):
        overlaps.append(compute_iou_each(det_bbox, gt_bbox))
    return min(overlaps)

def roidb2list(test_roidb, pred_roidb, mode='pred', topk=False, is_zs=False, dataset='vrd'):
	N_data = len(test_roidb)
	if topk:
		if dataset == 'vrd':
			k = 70
		else:
			k = 100
	else:
		k = 1
	# k = 70 if topk else 1

	det_labels = []
	det_bboxes = []

	for i in range(N_data):
		if mode == 'pred':
			n_dete = len(test_roidb[i]['rela_gt'])
		else:
			n_dete = len(test_roidb[i]['rela_dete'])
		conf_dete = np.ones([n_dete*k, 1])
		dete_label = np.concatenate([conf_dete, \
				np.reshape(pred_roidb[i]['pred_rela_score'],[n_dete*k,1]),
				conf_dete,
				np.repeat(np.reshape(pred_roidb[i]['sub_dete'],[n_dete,1]),k,axis=0),
				np.reshape(pred_roidb[i]['pred_rela'],[n_dete*k,1]),
				np.repeat(np.reshape(pred_roidb[i]['obj_dete'],[n_dete,1]),k,axis=0)], 1)

		dete_box = np.repeat(np.concatenate([
				np.reshape(pred_roidb[i]['sub_box_dete'],[n_dete, 1, 4]),
				np.reshape(pred_roidb[i]['obj_box_dete'],[n_dete, 1, 4])], 1), k, axis=0)
		det_labels.append(dete_label)
		det_bboxes.append(dete_box)

	gt_labels = []
	gt_bboxes = []
	if is_zs:
		if dataset == 'vrd':
			zs_flag = np.load('/DATA5_DB8/data/yhu/NRI/dsr_data/dsr_zs.npy', encoding='bytes')
		else:
			zs_flag = read_roidb('/DATA5_DB8/data/yhu/VTransE/input/zeroshot_vg.npz')
	for i in range(N_data):
		if is_zs:
			if dataset == 'vrd':
				zs_index = np.where(zs_flag[i]==1)[0]
			else:
				zs_index = np.where(zs_flag[i]['zero_shot']==1)[0]
			rela_gt = test_roidb[i]['rela_gt'][zs_index]
			sub_gt = test_roidb[i]['sub_gt'][zs_index]
			obj_gt = test_roidb[i]['obj_gt'][zs_index]
			sub_box_gt = test_roidb[i]['sub_box_gt'][zs_index]
			obj_box_gt = test_roidb[i]['obj_box_gt'][zs_index]
		else:
			rela_gt = test_roidb[i]['rela_gt']
			sub_gt = test_roidb[i]['sub_gt']
			obj_gt = test_roidb[i]['obj_gt']
			sub_box_gt = test_roidb[i]['sub_box_gt']
			obj_box_gt = test_roidb[i]['obj_box_gt']
		n_gt = len(rela_gt)
		gt_label = np.concatenate([
				np.reshape(sub_gt, [n_gt,1]),
				np.reshape(rela_gt, [n_gt,1]),
				np.reshape(obj_gt, [n_gt,1])], 1)

		gt_box = np.concatenate([
				np.reshape(sub_box_gt, [n_gt, 1, 4]),
				np.reshape(obj_box_gt, [n_gt, 1, 4])], 1)
		gt_labels.append(gt_label)
		gt_bboxes.append(gt_box)
	return det_labels, det_bboxes, gt_labels, gt_bboxes

def eval_result(test_roidb, pred_roidb, N_recall, is_zs=False, mode='pred', topk=False, dataset='vrd', vis=False):
	det_labels, det_bboxes, gt_labels, gt_bboxes = \
		roidb2list(test_roidb, pred_roidb, mode=mode, topk=topk, is_zs=is_zs, dataset=dataset)
	predicates = ["on", "wear", "has", "next to", "sleep next to", "sit next to", "stand next to", "park next",
				  "walk next to", "above", "behind", "stand behind", "sit behind", "park behind", "in the front of",
				  "under", "stand under", "sit under", "near", "walk to", "walk", "walk past", "in", "below", "beside",
				  "walk beside", "over", "hold", "by", "beneath", "with", "on the top of", "on the left of",
				  "on the right of", "sit on", "ride", "carry", "look", "stand on", "use", "at", "attach to", "cover",
				  "touch", "watch", "against", "inside", "adjacent to", "across", "contain", "drive", "drive on",
				  "taller than", "eat", "park on", "lying on", "pull", "talk", "lean on", "fly", "face", "play with",
				  "sleep on", "outside of", "rest on", "follow", "hit", "feed", "kick", "skate on"]
	objects = ["person", "sky", "building", "truck", "bus", "table", "shirt", "chair", "car", "train", "glasses",
			   "tree", "boat", "hat", "trees", "grass", "pants", "road", "motorcycle", "jacket", "monitor", "wheel",
			   "umbrella", "plate", "bike", "clock", "bag", "shoe", "laptop", "desk", "cabinet", "counter", "bench",
			   "shoes", "tower", "bottle", "helmet", "stove", "lamp", "coat", "bed", "dog", "mountain", "horse",
			   "plane", "roof", "skateboard", "traffic light", "bush", "phone", "airplane", "sofa", "cup", "sink",
			   "shelf", "box", "van", "hand", "shorts", "post", "jeans", "cat", "sunglasses", "bowl", "computer",
			   "pillow", "pizza", "basket", "elephant", "kite", "sand", "keyboard", "plant", "can", "vase",
			   "refrigerator", "cart", "skis", "pot", "surfboard", "paper", "mouse", "trash can", "cone", "camera",
			   "ball", "bear", "giraffe", "tie", "luggage", "faucet", "hydrant", "snowboard", "oven", "engine", "watch",
			   "face", "street", "ramp", "suitcase"]
	output = {}
	relationships_found = 0
	n_re = N_recall
	all_relationships = sum(labels.shape[0] for labels in gt_labels)
	curr_img_id = 0
	for item in zip(det_labels, det_bboxes, gt_labels, gt_bboxes):
		curr_img_name = os.path.basename(test_roidb[curr_img_id]["image"])
		# for each image, create a folder and save one image for each gt pair of sub-obj
		pairs = defaultdict(list)
		(det_lbls, det_bxs, gt_lbls, gt_bxs) = item
		if not det_lbls.any() or not gt_lbls.any():
			continue  # omit empty detection matrices
		gt_detected = np.zeros(gt_lbls.shape[0])
		# det_score = np.sum(np.log(det_lbls[:, 0:3]), axis=1)
		det_score = det_lbls[:,1]
		inds = np.argsort(det_score)[::-1][:n_re]  # at most n_re predictions for each image
		for pair_id, (det_box, det_label, det_s) in enumerate(zip(det_bxs[inds, :], det_lbls[inds, 3:], det_score[inds])):
			# find matched pair
			paired_gt_id = find_paired_gt_id(det_box, gt_bxs, det_label, gt_lbls)
			sub, pred, obj = objects[int(gt_lbls[paired_gt_id, 0])], predicates[int(gt_lbls[paired_gt_id, 1])], \
							 objects[int(gt_lbls[paired_gt_id, 2])]
			sub_box, obj_box = gt_bxs[paired_gt_id, 0], gt_bxs[paired_gt_id, 1]
			# sanity check for predicate prediction task, sub and obj same as gt
			det_sub, det_obj = objects[int(det_label[0])], objects[int(det_label[2])]
			assert (det_sub == sub and det_obj == obj)
			# get sub_str and obj_str
			sub_str = "_".join([str(int(sub_box[1])), str(int(sub_box[3])), str(int(sub_box[0])), str(int(sub_box[2]))])
			obj_str = "_".join([str(int(obj_box[1])), str(int(obj_box[3])), str(int(obj_box[0])), str(int(obj_box[2]))])
			key_name = "_".join([sub, sub_str, obj, obj_str])
			if int(det_label[1]) <= len(predicates)-1:
				# only put valid labels
				our_label = predicates[int(det_label[1])]
				pairs[key_name].append((our_label, det_s))

			# this overlaps only calculated on pred when its label == gt label
			# because "not any(det_label-gt_label)" only return true when sub-pred-obj all same as gt_label
			# here in predicate pred case, sub,obj must be same, so only return true when pred is same as gt label
			overlaps = np.array([
				max(compute_overlap(det_box, gt_box), 0.499)
				if detected == 0 and not any(det_label - gt_label)
				else 0
				for gt_box, gt_label, detected
				in zip(gt_bxs, gt_lbls, gt_detected)
			])
			if (overlaps >= 0.5).any():
				gt_detected[np.argmax(overlaps)] = 1
				relationships_found += 1

		output[curr_img_name] = pairs
		curr_img_id += 1
	# only save results when vis=True
	if vis:
		if topk:
			# save results for top70+recall50
			with open("/home/xuhuah/11777-Project-VRD/nmp_res_rec%d_top70.pkl"%(n_re), 'wb') as f:
				pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
		else:
			# save results for top1+recall50
			with open("/home/xuhuah/11777-Project-VRD/nmp_res_rec%d_top1.pkl"%(n_re), 'wb') as f:
				pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

	return float(relationships_found / all_relationships)


def find_paired_gt_id(det_box, gt_bxs, det_label, gt_lbls):
	overlaps = np.array([
		max(compute_overlap(det_box, gt_box), 0.499)
		if (det_label[0]==gt_label[0]) and (det_label[2]==gt_label[2])
		else 0
		for gt_box, gt_label
		in zip(gt_bxs, gt_lbls)
	])
	# for predicate task, must have exact match between boxes
	assert max(overlaps) == 1.0
	found_pair_id = np.argmax(overlaps)
	return found_pair_id

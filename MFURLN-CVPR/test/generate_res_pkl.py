import os,sys
import os.path
currentdir = '/home/xyao/11777-Project-VRD/MFURLN-CVPR/test'
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import pickle
import numpy as np 
import json
from collections import defaultdict

from model.ass_fun import *

def box_to_str(bbox):
    new_bbox = [bbox[1], bbox[3], bbox[0], bbox[2]]
    return '_'.join(map(str, map(int,new_bbox)))

predict_roidb = read_roidb('/data/xyao/sg_dataset/MFURLN/output/pred/vrd_roid_predicate.npz')
gt_roidb = read_roidb('/data/xyao/sg_dataset/MFURLN/process/vrd_pred_process_roidb.npz')['test_roidb']
with open('/data/xyao/sg_dataset/json_dataset/annotations_test.json', 'r') as f:
    test_raw = json.load(f)
with open('/data/xyao/sg_dataset/json_dataset/objects.json', 'r') as f:
    objects_raw = json.load(f)
with open('/data/xyao/sg_dataset/json_dataset/predicates.json', 'r') as f:
    predicates_raw = json.load(f)


predict_roidb = predict_roidb['pred_roidb']

result = {}
assert len(gt_roidb) == len(predict_roidb)
N_data = len(gt_roidb)
for i in range(N_data):
    image_name = gt_roidb[i]['image'].split('/')[-1]
    gt_rela_list = gt_roidb[i]['rela_gt']
    N_rela = len(gt_rela_list)
    if N_rela == 0:
        continue
    result[image_name] = defaultdict(list)
    sub_gt_list = gt_roidb[i]['sub_gt']
    obj_gt_list = gt_roidb[i]['obj_gt']
    sub_bbox_gt_list = gt_roidb[i]['sub_box_gt']
    obj_bbox_gt_list = gt_roidb[i]['obj_box_gt']
    pred_rela_list = predict_roidb[i]['pred_rela']
    pred_rela_score_list = predict_roidb[i]['pred_rela_score']
    N_pred = len(pred_rela_list)
    assert N_pred == N_rela
    sort_score = -np.sort(-np.reshape(pred_rela_score_list, [1, -1]))
    if N_pred <= 50:
        thresh = -1
    else:
        thresh = sort_score[0][50]

    for j in range(N_pred):
        assert sub_gt_list[j] == predict_roidb[i]['sub_dete'][j]
        assert obj_gt_list[j] == predict_roidb[i]['obj_dete'][j]
        assert sub_bbox_gt_list[j].any() == predict_roidb[i]['sub_box_dete'][j].any()
        assert obj_bbox_gt_list[j].any() == predict_roidb[i]['obj_box_dete'][j].any()
        predict_predicate = predicates_raw[int(pred_rela_list[j])]
        predict_predicate_score = pred_rela_score_list[j]
        if predict_predicate_score <= thresh:
            print("more")
            continue

        sub_bbox = box_to_str(sub_bbox_gt_list[j]) 
        obj_bbox = box_to_str(obj_bbox_gt_list[j]) 
        key_name = objects_raw[int(sub_gt_list[j])] + '_' + sub_bbox + '_' + objects_raw[int(obj_gt_list[j])] + '_' + obj_bbox
        result[image_name][key_name].append((predict_predicate,predict_predicate_score))


path_to_result = '/data/xyao/sg_dataset/MFURLN/output/mfurln_result.pkl'
with open(path_to_result, 'wb') as f:
    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved to %s"%(path_to_result))
"""
NOTE: before start, make sure you download the dataset from
https://drive.google.com/drive/folders/19CGn9p7w1jbjjbuFqHKzBFmcE9JK852U?usp=sharing
Then change the path to your stored location
"""
import os
from os.path import exists, join, isfile, dirname, abspath, split
import glob
import cv2
import json
import pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict

predicates = ["on", "wear", "has", "next to", "sleep next to", "sit next to", "stand next to", "park next", "walk next to", "above", "behind", "stand behind", "sit behind", "park behind", "in the front of", "under", "stand under", "sit under", "near", "walk to", "walk", "walk past", "in", "below", "beside", "walk beside", "over", "hold", "by", "beneath", "with", "on the top of", "on the left of", "on the right of", "sit on", "ride", "carry", "look", "stand on", "use", "at", "attach to", "cover", "touch", "watch", "against", "inside", "adjacent to", "across", "contain", "drive", "drive on", "taller than", "eat", "park on", "lying on", "pull", "talk", "lean on", "fly", "face", "play with", "sleep on", "outside of", "rest on", "follow", "hit", "feed", "kick", "skate on"]
objects = ["person", "sky", "building", "truck", "bus", "table", "shirt", "chair", "car", "train", "glasses", "tree", "boat", "hat", "trees", "grass", "pants", "road", "motorcycle", "jacket", "monitor", "wheel", "umbrella", "plate", "bike", "clock", "bag", "shoe", "laptop", "desk", "cabinet", "counter", "bench", "shoes", "tower", "bottle", "helmet", "stove", "lamp", "coat", "bed", "dog", "mountain", "horse", "plane", "roof", "skateboard", "traffic light", "bush", "phone", "airplane", "sofa", "cup", "sink", "shelf", "box", "van", "hand", "shorts", "post", "jeans", "cat", "sunglasses", "bowl", "computer", "pillow", "pizza", "basket", "elephant", "kite", "sand", "keyboard", "plant", "can", "vase", "refrigerator", "cart", "skis", "pot", "surfboard", "paper", "mouse", "trash can", "cone", "camera", "ball", "bear", "giraffe", "tie", "luggage", "faucet", "hydrant", "snowboard", "oven", "engine", "watch", "face", "street", "ramp", "suitcase"]


"""parse annotations to build a map with key = img_name"""
def parse_annos(annos):
    new_annos = {}
    for anno in annos:
        new_annos[anno["filename"]] = anno
    return new_annos

"""convert box format to xmin, ymin, xmax, ymax"""
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

"""generate similar format as predictions file for GT"""
def generate_gts(gt_annos):
    outputs = {}
    for img_name, anno in gt_annos.items():
        relations = anno["relationships"]
        objects = anno["objects"]
        pairs = defaultdict(list)
        for rela in relations:
            sub, pred, obj = rela["text"]
            key_name = get_key_name(rela, objects)
            # confidence score == 1.0 for GT
            pairs[key_name].append((pred, 1.0))
        outputs[img_name] = pairs
    return outputs

"""generate similar format as predictions file for GT (for 'json dataset' version) """
def generate_gts_v2(gt_annos):
    result = {}
    for img, attr in gt_annos.items():
        file = {}
        for rel in attr:
            subj_cat = int(rel['subject']['category'])
            obj_cat = int(rel['object']['category'])
            subj_bbox = str(rel['subject']['bbox'][0]) + '_' + str(rel['subject']['bbox'][2]) + '_' + str(rel['subject']['bbox'][1]) + '_' + str(rel['subject']['bbox'][3])
            obj_bbox = str(rel['object']['bbox'][0]) + '_' + str(rel['object']['bbox'][2]) + '_' + str(rel['object']['bbox'][1]) + '_' + str(rel['object']['bbox'][3])
            subj_name = objects[subj_cat]
            obj_name = objects[obj_cat]
            pair_name = subj_name + '_' + subj_bbox + '_' + obj_name + '_' + obj_bbox
            pre = predicates[rel['predicate']]
            file[pair_name] = file.get(pair_name, []) + [(pre, 1.0)]

        result[img] = file

    return result

"""draw bounding box and predicates from GT and preditions"""
def draw_box_preds_in_image(img_path, gt_list, pred_list, pair_name):
    rgb_img = cv2.imread(img_path)
    items = pair_name.split("_")
    # draw bounding boxes
    # ymin ymax xmin xmax
    cv2.rectangle(rgb_img, (int(items[3]), int(items[1])), (int(items[4]), int(items[2])), (0, 255, 0), 2)
    cv2.putText(rgb_img, items[0], (int(items[3])-5, int(items[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))
    cv2.rectangle(rgb_img, (int(items[8]), int(items[6])), (int(items[9]), int(items[7])), (255, 0, 0), 2)
    cv2.putText(rgb_img, items[5], (int(items[8])-5, int(items[6])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0))
    # two white boards to write results
    left_table = np.zeros((rgb_img.shape[0], 200, 3),dtype=np.uint8)
    right_table = np.zeros((rgb_img.shape[0], 200, 3),dtype=np.uint8)
    left_table.fill(255)
    right_table.fill(255)
    y0, dy = 10, 15
    for i, predicate in enumerate(gt_list):
        text = predicate[0] + ": " + str(predicate[1])
        y = y0 + i * dy
        cv2.putText(left_table, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255))
    for i, predicate in enumerate(pred_list):
        text = predicate[0] + ": " + str(predicate[1])
        y = y0 + i * dy
        cv2.putText(right_table, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255))
    vis_img = np.hstack([left_table, rgb_img, right_table])
    return vis_img

###START OF THE MAIN PROGARM###
path_to_dataset = "/home/xuhuah/11777-Project-VRD/nmp/dataset/vrd/sg_dataset"
path_to_result = "/home/xuhuah/11777-Project-VRD/result.pkl"
output_dirs = "/home/xuhuah/vrd_vis"

# load ground truth
img_dir = join(path_to_dataset, "sg_test_images")
gt_imgs = glob.glob(join(img_dir, "*"))
gt_annos = json.load(open(join(path_to_dataset, "annotations_test.json"), 'r'))
# gt_annos = parse_annos(gt_annos)
# # should have 1000 test images
# assert len(gt_imgs) == 1000
# assert len(gt_annos) == 1000
gts = generate_gts_v2(gt_annos)



# generate similar format as prediction for GT

# load predictions
preds = pickle.load(open(path_to_result, "rb"))
# should have 1000 test results
assert len(preds.keys()) == 1000

# iterate over GT annos
for img_name, anno in tqdm(gt_annos.items()):
    os.makedirs(join(output_dirs,img_name), exist_ok=True)
    relations = anno["relationships"]
    objects = anno["objects"]
    # get prediction result and gt
    pred = preds[img_name]
    gt = gts[img_name]
    for pair_name, gt_list in gt.items():
        # if this pair NOT exists in prediction, append "N_"
        prefix = "N_"
        pred_list = []
        if pair_name in pred:
            # if this pair exists in prediction, append "Y_"
            prefix = "Y_"
            pred_list = pred[pair_name]
        vis_img = draw_box_preds_in_image(join(img_dir, img_name), gt_list, pred_list, pair_name)
        cv2.imwrite(join(output_dirs, img_name, prefix+pair_name+".png"), vis_img)

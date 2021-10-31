import json

with open('/data/xyao/sg_dataset/json_dataset/annotations_test.json', 'r') as f:
    test_raw = json.load(f)
with open('/data/xyao/sg_dataset/json_dataset/objects.json', 'r') as f:
    objects_raw = json.load(f)

# if using sub_obj as key
repeat_imgs = []
for image_name in list(test_raw.keys()):
    sample = test_raw[image_name]
    sub_obj_pair = []
    for pair in sample:
        sub = pair['subject']['category']
        obj = pair['object']['category']
        sub_obj_pair.append(objects_raw[sub]+'_'+objects_raw[obj])
    if len(sub_obj_pair) != len(set(sub_obj_pair)):
        repeat_imgs.append(image_name)
print(len(repeat_imgs))



'''
[{'predicate': 0,
  'object': {'category': 59, 'bbox': [357, 481, 819, 854]},
  'subject': {'category': 25, 'bbox': [271, 355, 788, 873]}},
 {'predicate': 2,
  'object': {'category': 25, 'bbox': [271, 355, 788, 873]},
  'subject': {'category': 59, 'bbox': [357, 481, 819, 854]}},
 {'predicate': 14,
  'object': {'category': 25, 'bbox': [271, 355, 788, 873]},
  'subject': {'category': 48, 'bbox': [537, 718, 5, 913]}}]
'''
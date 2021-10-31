
def visualize(filenames, labels, boxes, nums=0):
    if nums != 0:
        filenames = filenames[:nums]
        labels = labels[:nums]
        boxes = boxes[:nums]
    
    # 读object id list
    with open("/home/xiaochen/matranse/json_annos/obj2vec.json", 'r') as f:
        obj_id_dict = {o: obj for o, obj in enumerate(sorted(json.load(f).keys()))}


    # 读predicate id list
    with open("/home/xiaochen/matranse/json_annos/predicate.json") as f:
        pred_id_dict = json.load(f)
    #with open("/home/xiaochen/matranse/json_annos/predicates.json", 'r') as f:
    #    pred_id_dict = {p: pred for p, pred in enumerate(json.load(f))}

    # 可视化检测结果
    save_dir = "/home/xiaochen/matranse/visual_results/"
    for i, filename in enumerate(filenames):
        if not boxes[i].any() or not labels[i].any():
            continue
        
        image_path = "/home/xiaochen/matranse/sg_dataset/images/" + filename + ".jpg"
        image = cv2.imread(image_path)
     
        #print(filename)
        #print(boxes[i], type(boxes[i]))
        #print(labels[i], type(labels[i]))
     
        boxes[i] = boxes[i][0]   
        labels[i] = labels[i][0].astype(np.int32)        
        if len(labels[i]) == 3:
            labels[i] = [0, 0, 0, labels[i][0], labels[i][1], labels[i][2]]


        #print(boxes[i], type(boxes[i]))
        #print(labels[i], type(labels[i]))

        # 画boxes
        obj_box = boxes[i][1]
        cv2.rectangle(image, (obj_box[2], obj_box[0]), (obj_box[3], obj_box[1]), color=(0,0,255), thickness=2)            
        sub_box = boxes[i][0]                       
        cv2.rectangle(image, (sub_box[2], sub_box[0]), (sub_box[3], sub_box[1]), color=(255,0,0), thickness=2)            
        # 画labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        obj_name = obj_id_dict[labels[i][5]] 
        label_size = cv2.getTextSize(obj_name, font, 1, 2)
        cv2.rectangle(image, (obj_box[2], obj_box[0] - label_size[0][1]), (obj_box[2] + label_size[0][0], obj_box[0]), color=(0,0,255), thickness=-1)
        cv2.putText(image, obj_name, (obj_box[2], obj_box[0]), font, 1, (255, 255, 255)) 

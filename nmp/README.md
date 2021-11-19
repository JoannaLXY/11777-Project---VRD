# Neural Message Passing for Visual Relationship Detection

## Requirements
- Pytorch 1.8.0
- Tensorflow 1.3.0
- Python 3.6

## ~~Data Preprocessing~~ 
<span style="color:red">[YOU DON'T NEED TO DO THIS SECTION BECAUSE HANDSOME XUHUA HAS DONE THIS FOR YOU!!]</span>

### 1. Extract and save visual appearance
Here we use vrd dataset and predicate detection as example.

- `CUDA_VISIBLE_DEVICES=0 python extract_vgg_feature.py --dataset=vrd --data_type=pred`

### 2. Save the data path into roidb file

- ` python process.py --dataset=vrd --data-type=pred`

## Data & Features Download 
<span style="color:red">[PLEASE DO THIS SECTION INSTEAD!!!]</span>

Please download following 3 folders from this [link](https://drive.google.com/drive/folders/1BbRhD8lgOiWliWc0Xm_vrksiMnMbsF4r?usp=sharing).
* ```VTransE/```
* ```dataset/```
* ```data/```

**The final folder should be**

```
├── VTransE
    ├── vrd_vgg_feats
└── dataset
    ├── vrd
      ├── sg_dataset
          ├── sg_test_images
          ├── sg_train_images
          └── *.json
└── data
    └── pretrained
    ├── *.npy
    └── *.npz
└── checkpoints
├── DataLoader.py
├── eval_metrics.py
├── modules.py
├── train_vg.py
├── train_vrd.py
├── utils.py
└── visualize.py
```

## Train & Eval
Some roidb files are downloaded from original paper, so their key/value contains wrong data path. To fix this, 
please update these two [lines](https://github.com/JoannaLXY/11777-Project-VRD/blob/d637548375430d88ff19964cbf0e35fbc49acb75/nmp/DataLoader.py#L18-L19) to your corresponding folders from last section.
This function ```update_keys()``` is used to hack the roidb. 

```
# train 
CUDA_VISIBLE_DEVICES=0 python train_vrd.py --encoder=nmp --use-loc --mode=whole --feat-mode=full
# eval
CUDA_VISIBLE_DEVICES=0 python train_vrd.py --encoder=nmp --use-loc --mode=eval --feat-mode=full --restore --load-folder=exp0
```

After training for 50 epochs, you should expect to see results reported in our midterm report:
```
------------- pred --------------
recall_50: 0.5698 recall_100: 0.5698
------------- pred topk--------------
recall_50: 0.9013 recall_100: 0.9628

```

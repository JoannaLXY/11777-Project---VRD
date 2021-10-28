#### Faster-rcnn trained weight
1. Use venv instead of conda
2. Follow https://github.com/endernewton/tf-faster-rcnn to download vgg-16 pretrained weights, make cuda librares
3. Train on vrd using MFURLN-CVPR-2019-relationship-detection-method/tf-faster-rcnn-master/lib/tools/train_vrd_dete_vgg.py
4. Or, download finetuned weight from https://share.weiyun.com/55KK78Y

#### Process to generate roidb
1. Run MFURLN-CVPR-2019-relationship-detection-method/process/vrd_pred_process.py, store it in '/data/xyao/sg_dataset/MFURLN/process/vrd_pred_process_roidb.npz'
2. Run MFURLN-CVPR-2019-relationship-detection-method/process/vrd_rela_process.py, store it in '/data/xyao/sg_dataset/MFURLN/process/vrd_rela_process_roidb.npz'

#### Spec on roidb files
1. '/data/xyao/sg_dataset/MFURLN/faster_rcnn/vrd_roidb.npz' is made from MFURLN-CVPR-2019-relationship-detection-method/tf-faster-rcnn-master/lib/tools/vrd_process_dete.py
2. '/data/xyao/sg_dataset/MFURLN/faster_rcnn/vrd_detected_box.npz' is made from MFURLN-CVPR-2019-relationship-detection-method/tf-faster-rcnn-master/lib/tools/get_vrd_vgg_rela_box.py

#### Train
1. Model from MFURLN-CVPR-2019-relationship-detection-method/train/vrd_rela.py is stored in '/data/xyao/sg_dataset/MFURLN/output/vrd_roid_rela.ckpt'
2. Model from MFURLN-CVPR-2019-relationship-detection-method/train/vrd_predicate is stored in '/data/xyao/sg_dataset/MFURLN/output/pred/'

#### Infernece -TBD
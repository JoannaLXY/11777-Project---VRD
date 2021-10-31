# -*- coding: utf-8 -*-
"""Functions for training and testing a network."""

import os
import cv2
import json

from matplotlib import pyplot as plt
import numpy as np
import torch
import yaml
from scipy.io import loadmat

from src.vrd_data_loader_class import VRDDataLoader
from src.utils.metric_utils import evaluate_relationship_recall
from src.utils.file_utils import load_annotations

with open('config.yaml', 'r') as f:
    CONFIG = yaml.load(f)


class VRDTrainTester():
    """Train and test utilities on VRD."""

    def __init__(self, net, net_name, use_cuda=CONFIG['use_cuda']):
        """Initiliaze train/test instance."""
        self.net = net
        self.net_name = net_name
        self.use_cuda = use_cuda

    def train(self, optimizer, criterion, scheduler=None,
              epochs=5, batch_size=32, val_batch_size=100,
              loss_sampling_period=50):
        """Train a neural network if it does not already exist."""
        # Check if the model is already trained
        print("Performing training for " + self.net_name)
        model_path_name = CONFIG['models_path'] + self.net_name + '.pt'
        if os.path.exists(model_path_name):
            self.net.load_state_dict(torch.load(model_path_name))
            print("Found existing trained model.")
            if self.use_cuda:
                self.net.cuda()
            else:
                self.net.cpu()
            return self.net
        # Settings and loading
        self.net.train()
        self.criterion = criterion
        data_loader = self._set_data_loaders(batch_size, val_batch_size)
        if self.use_cuda:
            self.net.cuda()
            if self.criterion is not None:
                self.criterion = self.criterion.cuda()
            data_loader.cuda()
            self.val_data_loader = self.val_data_loader.cuda()
        batches = data_loader.get_batches()

        # Main training procedure
        loss_history = []
        for epoch in range(epochs):
            if scheduler is not None:
                scheduler.step()
            accum_loss = 0
            for batch_cnt, batch_start in enumerate(batches):
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + Backward + Optimize on batch data
                loss = self._compute_loss(data_loader, epoch, batch_start)
                loss.backward()
                optimizer.step()

                # Print loss statistics
                accum_loss += loss.item()
                if (batch_cnt + 1) % loss_sampling_period == 0:
                    accum_loss /= loss_sampling_period
                    val_loss = self._compute_validation_loss()
                    loss_history.append((batch_cnt + 1, accum_loss, val_loss))
                    print(
                        '[%d, %5d] loss: %.3f, validation loss: %.3f'
                        % (epoch, batch_cnt, accum_loss, val_loss)
                    )
                    accum_loss = 0
        torch.save(self.net.state_dict(), model_path_name)
        print('Finished Training')
        #if any(loss_history):
        #    self.plot_loss(loss_history)
        return self.net

    def test(self, batch_size=100, test_mode='relationship'):
        """Test a neural network."""
        print("Testing %s on VRD." % (self.net_name))
        self.net.eval()
        data_loader = self._set_test_data_loader(
            batch_size=batch_size, test_mode=test_mode).eval()
        if self.use_cuda:
            self.net.cuda()
            data_loader.cuda()

        # scores 预测结果, boxes 和 labels 是 ground truth标签
        scores, boxes, labels = {}, {}, {}

        # 挨个batch测试
        for batch in data_loader.get_batches():
            outputs = self._net_outputs(data_loader, 0, batch)
            filenames = data_loader.get_files(0, batch)
            scores.update({
                filename: np.array(score_vec)
                for filename, score_vec
                in zip(filenames, outputs.cpu().detach().numpy().tolist())
            })
            boxes.update(data_loader.get_boxes(0, batch))
            labels.update(data_loader.get_labels(0, batch))
       
        '''
        print(outputs[0], "\n")
        print(filenames[0], "\n")          # 图像名字
        print(scores[filenames[0]], "\n")  # 70个predicate的score
        print(boxes[filenames[0]], "\n")   # obj box 和 sub box
        print(labels[filenames[0]], "\n")  # sub score, 0, obj score, sub id, -1, obj id

        # 读object id list
        obj_id_list = loadmat("/home/xiaochen/matranse/matlab_annos/objectListN.mat")
        obj_id_list = obj_id_list['objectListN'][0]

        # 读predicate id list
        pred_id_list = loadmat("/home/xiaochen/matranse/matlab_annos/predicate.mat")
        pred_id_list = pred_id_list['predicate'][0]

        # 可视化检测结果
        save_dir = "/home/xiaochen/matranse/visual_results/"
        for filename in filenames:
            image_path = "/home/xiaochen/matranse/sg_dataset/images/" + filename[:filename.rfind("_")] + ".jpg"
            image = cv2.imread(image_path)
            # 画boxes
            obj_box = boxes[filename][1]
            cv2.rectangle(image, (obj_box[2], obj_box[0]), (obj_box[3], obj_box[1]), color=(0,0,255), thickness=2)            
            sub_box = boxes[filename][0]                       
            cv2.rectangle(image, (sub_box[2], sub_box[0]), (sub_box[3], sub_box[1]), color=(255,0,0), thickness=2)            
            # 画labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            obj_name = obj_id_list[labels[filename][5]][0] 
            print(obj_name)           
            label_size = cv2.getTextSize(obj_name, font, 1, 2)
            cv2.rectangle(image, (obj_box[2], obj_box[0] - label_size[0][1]), (obj_box[2] + label_size[0][0], obj_box[0]), color=(0,0,255), thickness=-1)
            cv2.putText(image, obj_name, (obj_box[2], obj_box[0]), font, 1, (255, 255, 255)) 

            sub_name = obj_id_list[labels[filename][3]][0]            
            label_size = cv2.getTextSize(sub_name, font, 1, 2)
            cv2.rectangle(image, (sub_box[2], sub_box[0] - label_size[0][1]), (sub_box[2] + label_size[0][0], sub_box[0]), color=(255,0,0), thickness=-1)
            cv2.putText(image, sub_name, (sub_box[2], sub_box[0]), font, 1, (255, 255, 255)) 
            
            # 标注predicate
            pred_idx = np.argmax(scores[filename])
            pred_name = pred_id_list[pred_idx][0]
            label_size = cv2.getTextSize(pred_name, font, 1, 2)
            cv2.rectangle(image, (0, 0), label_size[0], color=(0,0,0), thickness=-1)
            cv2.putText(image, pred_name, (0, label_size[0][1]), font, 1, (255, 255, 255)) 

            # 存图
            flag = cv2.imwrite(save_dir + filename + ".jpg", image)
        '''

        #
        debug_scores = {
            filename: np.argmax(scores[filename])
            for filename in scores
        }
        #print(debug_scores)
 
        #
        annotations = load_annotations('test')

        #
        debug_labels = {
            rel['filename'][:rel['filename'].rfind('.')]: rel['predicate_id']
            for anno in annotations
            for rel in anno['relationships']
        }

        # 所有结果
        debug_annos = {
            rel['filename'][:rel['filename'].rfind('.')]: (
                rel,
                scores[rel['filename'][:rel['filename'].rfind('.')]].tolist()
            )
            for anno in annotations
            for rel in anno['relationships']
            if rel['filename'][:rel['filename'].rfind('.')] in scores
        }

        # 输出所有预测结果
        with open(self.net_name + '.json', 'w') as fid:
            json.dump(debug_annos, fid)

        # 输出预测正确的数量
        print(sum(
            1 for name in debug_scores
            if debug_scores[name] == debug_labels[name]))

        # 记录所有预测正确的图片名字
        with open(self.net_name + '.txt', 'w') as fid:
            fid.write(json.dumps([
                name for name in debug_scores
                if debug_scores[name] == debug_labels[name]]))

        # 输出统计结果
        for mode in ['relationship', 'unseen', 'seen']:
            for keep in [1, 70]:
                print(
                    'Recall@50-100 (top-%d) %s:' % (keep, mode),
                    evaluate_relationship_recall(
                        scores, boxes, labels, keep, mode
                    )
                )

    def plot_loss(self, loss_history):
        """
        Plot training and validation loss.

        loss_history is a list of 4-element tuples, like
        (epoch, batch number, train_loss, val_loss)
        """
        train_loss = [loss for _, loss, _ in loss_history]
        validation_loss = [val_loss for _, _, val_loss in loss_history]
        min_batch = min(batch for batch, _, _ in loss_history)
        ticks = [
            '%d' % batch if batch == min_batch else ''
            for batch, _, _ in loss_history
        ]
        non_white_ticks = [t for t, tick in enumerate(ticks) if tick]
        for epoch, position in enumerate(non_white_ticks):
            ticks[position] = str(epoch + 1) if not (epoch % 3) else ''
        _, axs = plt.subplots()
        axs.plot(train_loss)
        axs.plot(validation_loss, 'orange')
        plt.xticks(range(len(train_loss)), ticks)
        plt.title(self.net_name + ' Loss Curves')
        plt.ylabel('Loss')
        plt.xlabel('Epoch - Batch Number')
        plt.legend(['Train Loss', 'Val. Loss'], loc='upper left')
        plt.savefig(
            CONFIG['figures_path'] + self.net_name + 'Loss.jpg',
            bbox_inches='tight'
        )

    def _compute_loss(self, data_loader, epoch, batch_start):
        """Compute loss for current batch."""
        loss = self.criterion(
            self._net_outputs(data_loader, epoch, batch_start),
            data_loader.get_targets(epoch, batch_start)
        )
        loss += sum(0.01 * param.norm(2) for param in self.net.parameters())
        return loss

    def _compute_validation_loss(self):
        """Compute validation loss."""
        self.net.eval()
        batches = self.val_data_loader.get_batches()
        accum_loss = sum(
            self._compute_loss(self.val_data_loader, 0, batch_start).item()
            for batch_start in batches
        )
        self.net.train()
        return accum_loss / len(batches)

    def _set_data_loaders(self, batch_size, val_batch_size):
        """Set data loaders used during training."""
        data_loader = VRDDataLoader(batch_size=batch_size)
        self.val_data_loader = VRDDataLoader(batch_size=val_batch_size).eval()
        return data_loader

    def _set_test_data_loader(self, batch_size, test_mode):
        """Set data loader used during testing."""
        return VRDDataLoader(batch_size=batch_size, test_mode=test_mode)

    def _net_outputs(self, data_loader, epoch, batch_start):
        """Get network outputs for current batch."""
        return self.net(
            data_loader.get_union_boxes_pool5_features(epoch, batch_start)
        )

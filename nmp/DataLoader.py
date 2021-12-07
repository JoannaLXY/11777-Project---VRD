import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import numpy as np
import os
import pickle
import ipdb
from PIL import Image
from utils import read_roidb, box_id, get_box_feats, get_depth_feats

def update_keys(roidb):
    N = len(roidb)
    for i in range(N):
        roidb_use = roidb[i]
        for key, value in roidb_use.items():
            if isinstance(value, str):
                value = roidb_use[key]
                new_value = value.replace('/DATA5_DB8/data/yhu/VTransE/dsr_vrd_vgg_feats/', '/home/xuhuah/11777-Project-VRD/nmp/VTransE/vrd_vgg_feats/')
                new_value = new_value.replace('/DATA5_DB8/data/yhu/VTransE/dataset/VRD/', '/home/xuhuah/11777-Project-VRD/nmp/dataset/vrd/')
                new_value = new_value.replace('../VTransE', './VTransE')
                roidb_use[key] = new_value
    return roidb

class VrdPredDataset(Dataset):
    """docstring for VrdPred"""
    def __init__(self, mode = 'train', feat_mode = 'full', prior=False, ori_vgg=False, use_loc=False):
        super(VrdPredDataset, self).__init__()
        self.num_nodes = 21
        self.num_node_types = 101
        self.num_edge_types = 71
        self.num_edges = 41 #41 #30 #91
        if mode == 'train':
            self.mode = 'train'
        else:
            self.mode = 'test'
        self.feat_mode = feat_mode
        self.prior = prior
        # ----------- senmantic feature ------------- #
        self.predicates_vec = np.load('./data/vrd_predicates_vec.npy')
        self.objects_vec = np.load('./data/vrd_objects_vec.npy')

        # ------------ original roidb feature --------#
        self.roidb_read = read_roidb('./data/vrd_pred_graph_roidb.npz')
        self.roidb = self.roidb_read[self.mode]
        # hack: change provided name
        self.roidb = update_keys(self.roidb)

        # Exclude self edges
        self.off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((self.num_nodes, self.num_nodes)) - np.eye(self.num_nodes)),
            [self.num_nodes, self.num_nodes])

        # ------------ prior probability ------------- #
        # shape: [100, 100, 70] sum of the last dimension is 1
        f = open('./data/vrd_so_prior.pkl', 'rb')
        f.seek(0)
        self.rel_so_prior = pickle.load(f, encoding='bytes')    #[100, 100, 70]

        # ------------- prior of the existance of current [sub, obj] pair ---#
        # shape: [100, 100] sum=1
        self.prior_probs = np.load('./data/vrd_prior_prob.npy', encoding='bytes')

        self.use_loc = use_loc

    def get_adj(self, roidb_use):
        bbox_coordinates = np.zeros([self.num_edges, 20])
        # add depth feature, make it depth_size x depth_size
        depth_size = 64
        matrix = np.eye(self.num_nodes)
        rel_rec = np.zeros([self.num_edges, self.num_nodes])
        rel_send = np.zeros([self.num_edges, self.num_nodes])
        sub_idx = box_id(roidb_use['sub_box_gt'], roidb_use['uni_box_gt'])
        obj_idx = box_id(roidb_use['obj_box_gt'], roidb_use['uni_box_gt'])

        # name of image
        name = roidb_use['image'].split('/')[-1][:-4]

        # check if npy file exist
        npy_path = os.path.join('.', 'data', 'depth', self.mode + '_npy', name + '.npz')
        if os.path.exists(npy_path):
        # if exist just load it
            depth = np.load(npy_path)
            sub_depth = depth['sub_depth']
            obj_depth = depth['obj_depth']
            uni_depth = depth['uni_depth']
        else:
        # else read images
            sub_depth = np.zeros([self.num_edges, depth_size, depth_size])
            obj_depth = np.zeros([self.num_edges, depth_size, depth_size])
            uni_depth = np.zeros([self.num_edges, depth_size, depth_size])
            
            # load original image and depth image
            img_path = os.path.join('.', 'dataset', 'vrd', 'sg_dataset', 'sg_' + self.mode + '_images', name + '.jpg')
            depth_path = os.path.join('.', 'data', 'depth', self.mode, name + '.png') 
            try:
                ori_img = Image.open(img_path)
            except:
                ori_img = Image.open(img_path[:-4] + '.png')
            depth_img = Image.open(depth_path)
            # resize depth image
            depth_img = depth_img.resize(ori_img.size)
            depth_img = np.array(depth_img, np.float32)

        for i in range(len(sub_idx)):
            sub_id = int(sub_idx[i])
            obj_id = int(obj_idx[i])
            rel_rec[i] = matrix[obj_id]
            rel_send[i] = matrix[sub_id]
            bbox_coordinates[i] = get_box_feats(roidb_use['uni_box_gt'][sub_id], roidb_use['uni_box_gt'][obj_id])
            # also get depth info from images if not saved yet here
            if not os.path.exists(npy_path):
                sub_depth[i], obj_depth[i], uni_depth[i] = get_depth_feats(depth_img, roidb_use['uni_box_gt'][sub_id], roidb_use['uni_box_gt'][obj_id], depth_size) 

    
        # if not saved
        if not os.path.exists(npy_path):
            # run once to save depth info, make it faster
            np.savez(npy_path[:-4], sub_depth=sub_depth, obj_depth=obj_depth, uni_depth=uni_depth)

        depth = {'sub': sub_depth, 'obj': obj_depth, 'uni': uni_depth}
 
        #print(sub_depth.shape)
        #print(obj_depth.shape)
        #print(uni_depth.shape)

        # --------- cross entropy loss ---------#
        edges = np.zeros(self.num_edges) + self.num_edge_types - 1
        edges[:len(roidb_use['rela_gt'])] = roidb_use['rela_gt']
        edges = np.array(edges, dtype=np.int64)

        node_cls = np.zeros(self.num_nodes) + self.num_node_types - 1
        node_cls[:len(roidb_use['uni_gt'])] = roidb_use['uni_gt']
        node_cls = np.array(node_cls, dtype=np.int64)

        return edges, node_cls, rel_rec, rel_send, bbox_coordinates, depth

    def train_item(self, roidb_use, depth):
        if self.feat_mode == 'full':
            # --------- node feature ------------#
            feats = np.load(roidb_use['uni_fc7'])
            w2vec = list(map(lambda x: self.objects_vec[int(x)], roidb_use['uni_gt']))
            #print(roidb_use['uni_gt'])
            #print(roidb_use['uni_box_gt'])
            w2vec = np.reshape(np.array(w2vec),[-1, 300])
            nodes = np.zeros([self.num_nodes, 4096 + 4396])
            nodes[:feats.shape[0], :4096] = feats
            nodes[:feats.shape[0], 4096: 8192] = np.reshape(depth['uni'][:len(roidb_use['uni_gt'])], [-1, 4096])
            nodes[:feats.shape[0], 4096 + 4096:] = w2vec   # [self.num_nodes, 4096+300]
        elif self.feat_mode == 'vis':
            feats = np.load(roidb_use['uni_fc7'])
            nodes = np.zeros([self.num_nodes, 4096])
            nodes[:feats.shape[0]] = feats
        elif self.feat_mode == 'sem':
            w2vec = list(map(lambda x: self.objects_vec[int(x)], roidb_use['uni_gt']))
            w2vec = np.reshape(np.array(w2vec),[-1, 300])
            nodes = np.zeros([self.num_nodes, 300])
            nodes[:w2vec.shape[0]] = w2vec

        prior_matrix = np.zeros([self.num_edges, self.num_edge_types])-0.5/self.num_edge_types
        for i in range(len(roidb_use['rela_gt'])):
            sub_cls = int(roidb_use['sub_gt'][i])
            obj_cls = int(roidb_use['obj_gt'][i])
            current_prior = self.rel_so_prior[sub_cls, obj_cls]
            # current_prior = -0.5*(current_prior+1.0/self.num_edge_types)
            current_prior = -0.5*(1.0/self.num_edge_types)
            prior_matrix[i, :(self.num_edge_types-1)] = current_prior

        # ------ region vgg feature --- initialize edge feature ---------#
        # sub_idx = box_id(roidb_use['sub_box_gt'], roidb_use['uni_box_gt'])
        # obj_idx = box_id(roidb_use['obj_box_gt'], roidb_use['uni_box_gt'])
        edge_feats = np.zeros([self.num_edges, 512])
        pred_fc7 = np.load(roidb_use['pred_fc7'])
        edge_feats[:len(roidb_use['rela_gt'])] = pred_fc7
        
        return nodes, edge_feats, prior_matrix
    

    def __getitem__(self, index):
        roidb_use = self.roidb[index]
        edges, node_cls, rel_rec, rel_send, bbox_coordinates, depth = self.get_adj(roidb_use)
        nodes, edge_feats, prior_matrix = self.train_item(roidb_use, depth)

        '''
        print("\n")
        print(nodes.shape)
        print(edge_feats.shape)
        print(prior_matrix.shape)
        print(edges.shape)
        print(node_cls.shape)
        print(rel_rec.shape)
        print(rel_send.shape)
        print(bbox_coordinates.shape)
        print(depth.keys())
        print("\n")
        '''

        #sub_depth, obj_depth = self.get_depth(roidb_use, bbox_coordinates)
        bbox_coordinates = torch.FloatTensor(bbox_coordinates)
        nodes = torch.FloatTensor(nodes)
        edges = torch.LongTensor(edges)
        node_cls = torch.LongTensor(node_cls)
        edge_feats = torch.FloatTensor(edge_feats)
        rel_rec = torch.FloatTensor(rel_rec)
        rel_send = torch.FloatTensor(rel_send)
        prior_matrix = torch.FloatTensor(prior_matrix)
        for key in depth.keys():
            depth[key] = torch.FloatTensor(depth[key])
        if self.prior:
            return nodes, edges, node_cls, edge_feats, rel_rec, rel_send, bbox_coordinates, prior_matrix, depth
        else:
            return nodes, edges, node_cls, edge_feats, rel_rec, rel_send

    def __len__(self):
        return len(self.roidb)

class VrdRelaDataset(Dataset):
    """docstring for VrdRela"""
    def __init__(self, mode = 'train', feat_mode = 'full', prior=False, ori_vgg=False, use_loc=False, use_depth=False):
        super(VrdRelaDataset, self).__init__()
        self.num_nodes = 21 #44 #21
        self.num_edges = 41 #30 #170
        self.num_node_types = 101
        self.num_edge_types = 71
        self.feat_mode = feat_mode
        self.prior = prior
        if mode == 'train':
            self.mode = 'train'
        else:
            self.mode = 'test'
            # if mode == 'test':
            self.num_nodes = 96 #63
            self.num_edges = self.num_nodes * (self.num_nodes-1)

        # ----------- senmantic feature ------------- #
        self.predicates_vec = np.load('./data/vrd_predicates_vec.npy')
        self.objects_vec = np.load('./data/vrd_objects_vec.npy')

        # ------------ original roidb feature --------#
        self.roidb_read = read_roidb('./data/vrd_rela_graph_roidb_iou_dis_{}_{}.npz'.format(0.5*10, 0.45*10))
        
        self.roidb = self.roidb_read[self.mode]

        # Exclude self edges
        self.off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((self.num_nodes, self.num_nodes)) - np.eye(self.num_nodes)),
            [self.num_nodes, self.num_nodes])

        # ------------ prior probability ------------- #
        self.prior = prior
        f = open('./data/vrd_so_prior.pkl', 'rb')
        f.seek(0)
        self.rel_so_prior = pickle.load(f, encoding='bytes')    #[100, 100, 70]

        self.use_loc = use_loc
        self.use_depth = use_depth

    def get_adj(self, roidb_use):
        bbox_coordinates = np.zeros([self.num_edges, 20])
        matrix = np.eye(self.num_nodes)
        rel_rec = np.zeros([self.num_edges, self.num_nodes])
        rel_send = np.zeros([self.num_edges, self.num_nodes])
        sub_idx = box_id(roidb_use['sub_box_dete'], roidb_use['uni_box_gt'])
        obj_idx = box_id(roidb_use['obj_box_dete'], roidb_use['uni_box_gt'])

        for i in range(len(sub_idx)):
            sub_id = int(sub_idx[i])
            obj_id = int(obj_idx[i])
            rel_rec[i] = matrix[obj_id]
            rel_send[i] = matrix[sub_id]
            bbox_coordinates[i] = get_box_feats(roidb_use['uni_box_gt'][sub_id], roidb_use['uni_box_gt'][obj_id])

        edges = np.zeros(self.num_edges) + self.num_edge_types-1
        edges[:len(roidb_use['rela_dete'])] = roidb_use['rela_dete']
        edges = np.array(edges, dtype=np.int64)

        node_cls = np.zeros(self.num_nodes) + self.num_node_types-1
        node_cls[:len(roidb_use['uni_gt'])] = roidb_use['uni_gt']
        node_cls = np.array(node_cls, dtype=np.int64)
        return edges, node_cls, rel_rec, rel_send, bbox_coordinates

    def train_item(self, roidb_use):
        # --------- node feature ------------#
        feats = np.load(roidb_use['uni_fc7'])
        
        w2vec = list(map(lambda x: self.objects_vec[int(x)], roidb_use['uni_gt']))
        w2vec = np.reshape(np.array(w2vec),[-1, 300])

        if feats.shape[0] > self.num_nodes:
            index_box = np.sort(random.sample(range(feats.shape[0]), self.num_nodes))
            feats = feats[index_box, :]
            w2vec = w2vec[index_box, :]
            if self.feat_mode == 'full':
                nodes = np.concatenate([feats, w2vec], 1)   # [self.num_nodes, 4096+300]
            elif self.feat_mode == 'vis':
                nodes = feats
            elif self.feat_mode == 'sem':
                nodes = w2vec

            # --------- edge feature ------------#
            # edge_idx = roidb_use['edge_matrix'][index_box, :]
            # edge_idx = edge_idx[:, index_box]             # [self.num_nodes, self.num_nodes]
        else:
            if self.feat_mode == 'full':
                nodes = np.zeros([self.num_nodes, 4396])
                nodes[:feats.shape[0], :4096] = feats
                nodes[:feats.shape[0], 4096:] = w2vec   # [self.num_nodes, 4096+300]
            elif self.feat_mode == 'vis':
                nodes = np.zeros([self.num_nodes, 4096])
                nodes[:feats.shape[0]] = feats
            elif self.feat_mode == 'sem':
                nodes = np.zeros([self.num_nodes, 300])
                nodes[:w2vec.shape[0]] = w2vec
            
        prior_matrix = np.zeros([self.num_edges, self.num_edge_types])-0.5/self.num_edge_types
        for i in range(len(roidb_use['rela_dete'])):
            sub_cls = int(roidb_use['sub_dete'][i])
            obj_cls = int(roidb_use['obj_dete'][i])
            current_prior = self.rel_so_prior[sub_cls, obj_cls]
            # current_prior = -0.5*(current_prior+1.0/self.num_edge_types)
            current_prior = -0.5*(1.0/self.num_edge_types)
            prior_matrix[i, :(self.num_edge_types-1)] = current_prior

        # ------ region vgg feature --- initialize edge feature ---------#
        sub_idx = box_id(roidb_use['sub_box_dete'], roidb_use['uni_box_gt'])
        obj_idx = box_id(roidb_use['obj_box_dete'], roidb_use['uni_box_gt'])
        edge_feats = np.zeros([self.num_edges, 512])

        # pred_fc7 = np.load(roidb_use['pred_fc7'])

        # edge_feats[:len(roidb_use['rela_dete'])] = pred_fc7

        # for i in range(len(sub_idx)):
        #     edge_feats[int(sub_idx[i]),int(obj_idx[i])] = pred_fc7[i]
        # edge_feats = np.reshape(edge_feats, [self.num_nodes ** 2, -1])
        # edge_feats = edge_feats[self.off_diag_idx]

        return nodes, edge_feats, prior_matrix


    def __getitem__(self, index):
        roidb_use = self.roidb[index]
        nodes, edge_feats, prior_matrix = self.train_item(roidb_use)
        edges, node_cls, rel_rec, rel_send, bbox_coordinates = self.get_adj(roidb_use)
        bbox_coordinates = torch.FloatTensor(bbox_coordinates)
        nodes = torch.FloatTensor(nodes)
        edges = torch.LongTensor(edges)
        node_cls = torch.LongTensor(node_cls)
        edge_feats = torch.FloatTensor(edge_feats)
        rel_rec = torch.FloatTensor(rel_rec)
        rel_send = torch.FloatTensor(rel_send)

        prior_matrix = torch.FloatTensor(prior_matrix)
        if self.prior:
            return nodes, edges, node_cls, edge_feats, rel_rec, rel_send, bbox_coordinates, prior_matrix
        else:
            return nodes, edges, node_cls, edge_feats, rel_rec, rel_send

    def __len__(self):
        return len(self.roidb)

class VgPredDataset(Dataset):
    """docstring for VgPred"""
    def __init__(self, mode = 'train', feat_mode = 'full', prior = False, ori_vgg=False, use_loc=False):
        super(VgPredDataset, self).__init__()
        self.num_nodes = 110 #98
        self.num_edge_types = 101
        self.num_node_types = 201
        self.num_edges = 490 #352
        if mode == 'train':
            self.mode = 'train'
        else:
            self.mode = 'test'
        self.feat_mode = feat_mode
        self.prior = prior
        # ----------- senmantic feature ------------- #
        self.predicates_vec = np.load('./data/vg_predicates_vec.npy')
        self.objects_vec = np.load('./data/vg_objects_vec.npy')

        # ------------ original roidb feature --------#
        self.roidb_read = read_roidb('./data/vg_pred_graph_roidb.npz')
        self.roidb = self.roidb_read[self.mode]

        self.rel_so_prior = np.load('./data/vg_so_prior.npy')    #[201, 201, 100]

        self.use_loc = use_loc

    def get_adj(self, roidb_use):
        bbox_coordinates = np.zeros([self.num_edges, 20])
        matrix = np.eye(self.num_nodes)
        rel_rec = np.zeros([self.num_edges, self.num_nodes])
        rel_send = np.zeros([self.num_edges, self.num_nodes])
        sub_idx = box_id(roidb_use['sub_box_gt'], roidb_use['uni_box_gt'])
        obj_idx = box_id(roidb_use['obj_box_gt'], roidb_use['uni_box_gt'])

        for i in range(len(sub_idx)):
            sub_id = int(sub_idx[i])
            obj_id = int(obj_idx[i])
            rel_rec[i] = matrix[obj_id]
            rel_send[i] = matrix[sub_id]
            bbox_coordinates[i] = get_box_feats(roidb_use['uni_box_gt'][sub_id], roidb_use['uni_box_gt'][obj_id])

        edges = np.zeros(self.num_edges) + self.num_edge_types - 1
        edges[:len(roidb_use['rela_gt'])] = roidb_use['rela_gt']
        edges = np.array(edges, dtype=np.int64)

        node_cls = np.zeros(self.num_nodes) + self.num_node_types-1
        node_cls[:len(roidb_use['uni_gt'])] = roidb_use['uni_gt']
        node_cls = np.array(node_cls, dtype=np.int64)
        return edges, node_cls, rel_rec, rel_send, bbox_coordinates

    def train_item(self, roidb_use):

        if self.feat_mode == 'full':
            # --------- node feature ------------#
            feats = np.load(roidb_use['uni_fc7'])
            w2vec = list(map(lambda x: self.objects_vec[int(x)], roidb_use['uni_gt']))
            w2vec = np.reshape(np.array(w2vec),[-1, 300])
            nodes = np.zeros([self.num_nodes, 4396])
            nodes[:feats.shape[0], :4096] = feats
            nodes[:feats.shape[0], 4096:] = w2vec   # [self.num_nodes, 4096+300]
        elif self.feat_mode == 'vis':
            feats = np.load(roidb_use['uni_fc7'])
            nodes = np.zeros([self.num_nodes, 4096])
            nodes[:feats.shape[0]] = feats
        elif self.feat_mode == 'sem':
            w2vec = list(map(lambda x: self.objects_vec[int(x)], roidb_use['uni_gt']))
            w2vec = np.reshape(np.array(w2vec),[-1, 300])
            nodes = np.zeros([self.num_nodes, 300])
            nodes[:w2vec.shape[0]] = w2vec

        
        # prior_matrix = np.zeros([self.num_edges, self.num_edge_types])
        prior_matrix = np.zeros([self.num_edges, self.num_edge_types])-0.5/self.num_edge_types
        for i in range(len(roidb_use['rela_gt'])):
            sub_cls = int(roidb_use['sub_gt'][i])
            obj_cls = int(roidb_use['obj_gt'][i])
            current_prior = self.rel_so_prior[sub_cls, obj_cls]
            current_prior = -0.5*(current_prior+1.0/self.num_edge_types)
            # current_prior = -1.0*(current_prior+1.0/self.num_edge_types)
            prior_matrix[i, :(self.num_edge_types-1)] = current_prior

        # ------ region vgg feature --- initialize edge feature ---------#
        # sub_idx = box_id(roidb_use['sub_box_gt'], roidb_use['uni_box_gt'])
        # obj_idx = box_id(roidb_use['obj_box_gt'], roidb_use['uni_box_gt'])
        edge_feats = np.zeros([self.num_edges, 512])
        pred_fc7 = np.load(roidb_use['pred_fc7'])
        edge_feats[:len(roidb_use['rela_gt'])] = pred_fc7
        return nodes, edge_feats, prior_matrix

    def __getitem__(self, index):
        roidb_use = self.roidb[index]
        nodes, edge_feats, prior_matrix = self.train_item(roidb_use)
        edges, node_cls, rel_rec, rel_send, bbox_coordinates = self.get_adj(roidb_use)
        nodes = torch.FloatTensor(nodes)
        edges = torch.LongTensor(edges)
        node_cls = torch.LongTensor(node_cls)
        edge_feats = torch.FloatTensor(edge_feats)
        rel_rec = torch.FloatTensor(rel_rec)
        rel_send = torch.FloatTensor(rel_send)
        prior_matrix = torch.FloatTensor(prior_matrix)
        bbox_coordinates = torch.FloatTensor(bbox_coordinates)
        if self.prior:
            return nodes, edges, node_cls, edge_feats, rel_rec, rel_send, bbox_coordinates, prior_matrix
        else:
            return nodes, edges, node_cls, edge_feats, rel_rec, rel_send

    def __len__(self):
        return len(self.roidb)

def load_dataset(data_set='vrd', ori_vgg=False, dataset='pred', level='image', batch_size=32, eval_batch_size=32, shuffle=False,  feat_mode='full', prior=False):
    if data_set == 'vrd':
        if dataset=='pred' and level=='image':
            load_func_name = VrdPredDataset
        elif dataset=='rela' and level=='image':
            load_func_name = VrdRelaDataset
    else:
        load_func_name = VgPredDataset

    train_data = load_func_name(mode='train', feat_mode = feat_mode, prior=True, ori_vgg=ori_vgg)
    val_data = load_func_name(mode='test', feat_mode = feat_mode, prior=True, ori_vgg=ori_vgg)
    test_data = load_func_name(mode='test', feat_mode = feat_mode, prior=True, ori_vgg=ori_vgg)

    train_loader = DataLoader(train_data, shuffle=shuffle, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=eval_batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=eval_batch_size)

    return train_loader, val_loader, test_loader

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import ipdb
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable


class Interpolate(nn.Module):
    def __init__(self, size, mode="bilinear"):
        super(Interpolate, self).__init__()
        self.interpolate = F.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interpolate(x, size=self.size, mode=self.mode)
        return x


class DepthDecoder(nn.Module):
    def __init__(self, conv_in_channel=41, conv_out_channel=1024, fc_hid=64, do_prob=0.):
        super(DepthDecoder, self).__init__()
        # typically the conv_in_channel equals to the number of edges, which is 41 by default
        self.conv_in_channel = conv_in_channel
        self.conv_out_channel = conv_out_channel
        self.fc_hid = fc_hid

        self.conv_depth = nn.Sequential(
            nn.Conv2d(self.conv_in_channel, self.conv_out_channel, kernel_size=3, padding=1),
            nn.ReLU(True), 
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc_depth = MLP(self.conv_out_channel, self.fc_hid, self.fc_hid, do_prob)
 
    def forward(self, depth):
        x = self.conv_depth(depth)
        #print(x.size())
        x = torch.flatten(x, 1)
        #print(x.size())
        x = self.fc_depth(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)

class SimpleEncoder(nn.Module):
    def __init__(self, n_hid, edge_types=71, node_types=101, do_prob=0., use_vis=True, use_spatial=True, use_sem=True, use_loc=False, use_cls=False):
        super(SimpleEncoder, self).__init__()

        self.use_vis = use_vis
        self.use_spatial = use_spatial
        self.use_sem = use_sem
        self.use_loc = use_loc
        self.use_cls = use_cls

        # self.vis_hid = int(n_hid/2)
        self.vis_hid = n_hid
        self.sem_hid = n_hid
        self.spatial_hid = n_hid
        self.loc_hid = 64
        self.cls_hid = 64

        self.fc_vis = FC(4096, self.vis_hid)
        self.fc_spatial = FC(512, self.spatial_hid)
        self.fc_sem = FC(300, self.sem_hid)
        self.fc_loc = FC(20, self.loc_hid)

        n_fusion = 0
        if self.use_vis:
            n_fusion += self.vis_hid
            if self.use_cls:
                n_fusion += self.cls_hid
        if self.use_spatial:
            n_fusion += self.spatial_hid
        if self.use_sem:
            n_fusion += self.sem_hid
        if self.use_loc:
            n_fusion += self.loc_hid
        
        # ---- sub obj concat ---------#
        self.fc_so_vis = FC(self.vis_hid*2, self.vis_hid)

        # ---- sub obj concat ---------#
        self.fc_so_sem = FC(self.sem_hid*2, self.sem_hid)

        # ---- all the feature into hidden space -------#
        self.fc_fusion = FC(n_fusion, n_hid)

        self.fc_rel = FC(n_hid, edge_types, relu=False)

        if self.use_vis:
            self.fc_cls = FC(4096, node_types, relu=False)
        else:
            self.fc_cls = FC(300, node_types, relu=False)
        self.fc_so_cls = FC(node_types*2, self.cls_hid)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, spatial_feats, rel_rec, rel_send, bbox_loc):
        inputs = inputs[:, :, :].contiguous()
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [batch_size, num_nodes, num_dims]

        if self.use_vis:
            x_v = self.fc_vis(x[:, :, :4096])   #[batch_size, num_nodes, n_hid]
            e_hid_v = self.node2edge(x_v, rel_rec, rel_send) #[batch_size, num_edges, n_hid*2]
            e_v = self.fc_so_vis(e_hid_v)   #[batch_size, num_edges, n_hid]
            edge_feats = e_v                #[batch_size, num_edges, n_hid]

            x_cls = self.fc_cls(x[:, :, :4096])
            if self.use_cls:
                e_hid_cls = self.node2edge(x_cls, rel_rec, rel_send)
                e_cls = self.fc_so_cls(e_hid_cls)
                edge_feats = torch.cat([edge_feats, e_cls], -1)

        if self.use_sem:
            if self.use_vis:
                x_s = self.fc_sem(x[:, :, 4096:])   #[batch_size, num_nodes, n_hid]
            else:
                x_s = self.fc_sem(x)
                x_cls = self.fc_cls(x)
            e_hid_s = self.node2edge(x_s, rel_rec, rel_send) #[batch_size, num_edges, n_hid*2]
            e_s = self.fc_so_sem(e_hid_s)   #[batch_size, num_edges, n_hid]
            if self.use_vis:
                edge_feats = torch.cat([edge_feats, e_s], -1)   #[batch_size, num_edges, n_hid*2]
            else:
                edge_feats = e_s

        if self.use_spatial:
            e_l = self.fc_spatial(spatial_feats)   #[batch_size, bun_edges, n_hid]
            if self.use_vis or self.use_sem:
                edge_feats = torch.cat([edge_feats, e_l], -1)   #[batch_size, num_edges, n_hid*3]
            else:
                edge_feats = e_l

        if self.use_loc:
            e_loc = self.fc_loc(bbox_loc)
            edge_feats = torch.cat([edge_feats, e_loc], -1)

        self.edge_feats_final = self.fc_fusion(edge_feats)
        output = self.fc_rel(self.edge_feats_final)
        return output, x_cls
        

class NMPEncoder(nn.Module):
    def __init__(self, n_hid, edge_types=71, node_types=101, n_iter=2, do_prob=0., use_vis=True, use_spatial=False, use_sem=True, use_loc=False, use_cls=False, uni_depth=False, obj_depth=True):
        super(NMPEncoder, self).__init__()
        self.use_vis = use_vis
        self.use_spatial = use_spatial
        self.use_sem = use_sem
        self.use_loc = use_loc
        self.use_cls = use_cls
        self.uni_depth = uni_depth
        self.obj_depth = obj_depth

        self.n_iter = n_iter

        self.vis_hid = 128
        self.sem_hid = n_hid
        self.spatial_hid = n_hid
        self.loc_hid = 64
        self.cls_hid = 64
        self.uni_depth_hid = 64
        self.obj_depth_hid = 128 

        self.mlp1 = MLP(n_hid * 2, n_hid, n_hid, do_prob)   # vanilla | diff2
        #self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)       # diff
        #self.mlp1 = MLP(n_hid * 3, n_hid, n_hid, do_prob)    # both | both2
        #self.mlp1 = MLP(n_hid * 4, n_hid, n_hid, do_prob)    # both3
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid * 3, n_hid, n_hid, do_prob)   # vanilla | diff2
        #self.mlp3 = MLP(n_hid * 2, n_hid, n_hid, do_prob)   # vector diff
        #self.mlp3 = MLP(n_hid * 4, n_hid, n_hid, do_prob)    # both | both2
        #self.mlp3 = MLP(n_hid * 5, n_hid, n_hid, do_prob)    # both3
        self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp5 = MLP(n_hid * 2, n_hid, n_hid, do_prob)

        self.mlp_e2n = MLP(n_hid * 2, n_hid, n_hid, do_prob)

        # ------- visual feature ---------#
        # self.fc_vis = FC(4096, n_hid)
        self.fc_vis = MLP(4096, self.vis_hid, self.vis_hid, do_prob)
        # ------ spatial feature ---------#
        # self.fc_spatial = FC(512, n_hid)
        self.fc_spatial = MLP(512, self.spatial_hid, self.spatial_hid, do_prob)
        # ------- semantic feature -------#
        # self.fc_sem = FC(300, n_hid)
        self.fc_sem = MLP(300, self.sem_hid, self.sem_hid, do_prob)
        # ------- location feature -------#
        # 2D feature
        self.fc_loc = MLP(20, self.loc_hid, self.loc_hid, do_prob)
        # ------- depth feature -------#
        # obj depth feature
        self.fc_obj_depth = MLP(64*64, self.obj_depth_hid, self.obj_depth_hid, do_prob)
        # union depth feature
        self.fc_uni_depth = MLP(64*64, self.uni_depth_hid, self.uni_depth_hid, do_prob)

        n_fusion = 0
        if self.use_vis:
            n_fusion += self.vis_hid
            if self.obj_depth:
                n_fusion += self.obj_depth_hid
            if self.use_cls:
                n_fusion += self.cls_hid
        if self.use_sem:
            n_fusion += self.sem_hid
        
        final_fusion = n_hid
        if self.use_loc:
            final_fusion += self.loc_hid
        if self.uni_depth:
            final_fusion += self.uni_depth_hid

        # # ---- sub obj concat ---------#
        # self.fc_so_vis = FC(n_hid*2, n_hid)

        # # ---- sub obj concat ---------#
        # self.fc_so_sem = FC(n_hid*2, n_hid)

        # ---- all the feature into hidden space -------#
        self.fc_fusion = FC(n_fusion, n_hid)

        self.fc_rel = FC(final_fusion, edge_types, relu=False)

        if self.use_vis:
            self.fc_cls = FC(4096, node_types, relu=False)
        else:
            self.fc_cls = FC(300, node_types, relu=False)
        self.fc_cls_feat = FC(node_types, self.cls_hid)

        self.dropout_prob = do_prob
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        diff = receivers - senders
        diff_inv = senders - receivers

        # vanilla
        edges = torch.cat([receivers, senders], dim=2)

        # diff
        #edges = diff

        # diff 2, using diff and inv diff
        #edges = torch.cat([diff, diff_inv], dim=2)

        # both
        #edges = torch.cat([receivers, senders, diff], dim=2)

        # both 2, put diff between them
        #edges = torch.cat([receivers, diff, senders], dim=2)

        # both 3, put two difference between them
        #edges = torch.cat([receivers, diff, diff_inv, senders], dim=2)

        return edges

    def edge2node(self, x, rel_rec, rel_send):
        new_rec_rec = rel_rec.permute(0,2,1)
        weight_rec = torch.sum(new_rec_rec, -1).float()
        weight_rec = weight_rec + (weight_rec==0).float()
        weight_rec = torch.unsqueeze(weight_rec, -1).expand(weight_rec.size(0), weight_rec.size(1), x.size(-1))
        incoming = torch.matmul(new_rec_rec, x)
        incoming = incoming / weight_rec

        new_rec_send = rel_send.permute(0,2,1)
        weight_send = torch.sum(new_rec_send, -1).float()
        weight_send = weight_send + (weight_send==0).float()
        weight_send = torch.unsqueeze(weight_send, -1).expand(weight_send.size(0), weight_send.size(1), x.size(-1))
        outgoing = torch.matmul(new_rec_send, x)
        outgoing = outgoing / weight_send

        nodes = torch.cat([incoming, outgoing], -1)
        # nodes = (incoming + outgoing) * 0.5
        # nodes = incoming + outgoing
        # nodes = outgoing
        # nodes = incoming
        return nodes

    def forward(self, inputs, spatial_feats, rel_rec, rel_send, bbox_loc, depth=None):
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        batch_size = inputs.size(0)
        n_atoms = inputs.size(1)
        n_edges = rel_rec.size(1)

        # add depth info here
        #if depth != None:
        #    sub_depth, obj_depth, uni_depth = depth['sub'], depth['obj'], depth['uni'] 
        #else:
        #    sub_depth, obj_depth, uni_depth = None, None, None

        if self.use_vis:
            x_v = self.fc_vis(x[:, :, :4096])   #[batch_size, num_nodes, n_hid]
            node_feats = x_v
            if self.obj_depth:
                x_d = self.fc_obj_depth(x[:, :, 4096: 8192])
                node_feats = torch.cat([node_feats, x_d], -1)
            if self.use_sem:
                x_s = self.fc_sem(x[:, :, 4096 + 4096:])   #[batch_size, num_nodes, n_hid]
                node_feats = torch.cat([node_feats, x_s], -1)
            x_cls = self.fc_cls(x[:, :, :4096])
            if self.use_cls:
                e_cls = self.fc_cls_feat(x_cls)
                node_feats = torch.cat([node_feats, e_cls], -1)
        else:
            x_s = self.fc_sem(x)
            node_feats = x_s
            x_cls = self.fc_cls(x)

        # 0th fusion nodes
        node_feats = self.fc_fusion(node_feats)

        # 1st nodes2edges
        if self.use_spatial:
            x_l = self.fc_spatial(spatial_feats)
            edge_feats = x_l
        else:
            edge_feats = self.mlp1(self.node2edge(node_feats, rel_rec, rel_send))

        # 2nd edges2nodes
        x = edge_feats
        x = self.mlp_e2n(self.edge2node(x, rel_rec, rel_send))
        x = self.mlp2(x)
        
        # 3rd nodes2edges
        self.node_feats = x
        x = self.node2edge(x, rel_rec, rel_send)
        # # n2e
        # x = self.mlp4(x)
        # x = self.edge2node(x, rel_rec, rel_send)
        # x = self.mlp5(x)
        # x = self.node2edge(x, rel_rec, rel_send)
        # [e_{ij}^1; e_{ij}^2]

        # 4th fusion edges
        x = torch.cat((x, edge_feats), dim=2)  # Skip connection
        self.edge_feats = self.mlp3(x)
        # e_{ij}^2
        # self.edge_feats = self.mlp4(x)        

        # final concat spatial
        if self.use_loc:
            e_loc = self.fc_loc(bbox_loc)
            self.edge_feats = torch.cat([self.edge_feats, e_loc], -1)
        
        #if self.uni_depth:
        #    e_depth = self.fc_uni_depth(torch.flatten(uni_depth, 2))
        #    self.edge_feats = torch.cat([self.edge_feats, e_depth], -1)

        # out fc feat
        output = self.fc_rel(self.edge_feats)

        return output, x_cls

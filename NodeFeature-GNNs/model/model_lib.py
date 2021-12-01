import torch
import torch.nn as nn
import numpy as np
# import torch.nn.functional as F
# from model.resnet3d_xl import Net
# from model.nonlocal_helper import Nonlocal
# from ipdb import set_trace
from utils import rownorm
from ipdb import set_trace
class RefNRIMLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0., no_bn=False):
        super(RefNRIMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.ELU(inplace=True),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_out),
            nn.ELU(inplace=True)
        )
        if no_bn:
            self.bn = None
        else:
            self.bn = nn.BatchNorm1d(n_out)

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
        orig_shape = inputs.shape
        x = inputs.view(-1, inputs.size(-1))
        x = self.bn(x)
        return x.view(orig_shape)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = self.model(inputs)
        if self.bn is not None:
            return self.batch_norm(x)
        else:
            return x


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


class VideoModelCoord(nn.Module):
    def __init__(self, opt):
        super(VideoModelCoord, self).__init__()
        self.nr_boxes = 4
        self.nr_actions = opt.num_classes
        self.nr_frames = 16
        self.coord_feature_dim = opt.coord_feature_dim
        dropout = 0.1
        no_bn = False

#######################################################################################
        self.category_embed_layer = nn.Embedding(3, opt.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)

        self.coord_category_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim+self.coord_feature_dim//2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
        )
########################################################################################

        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim//2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim//2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        edges = np.ones(4) - np.eye(4)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()),
                                          requires_grad=False)

        self.mlp1 = RefNRIMLP(self.coord_feature_dim * 2, self.coord_feature_dim, self.coord_feature_dim, dropout, no_bn=no_bn)
        self.mlp2 = RefNRIMLP(self.coord_feature_dim, self.coord_feature_dim, self.coord_feature_dim, dropout, no_bn=no_bn)

        self.tmlp1 = RefNRIMLP(self.coord_feature_dim * 2, self.coord_feature_dim, self.coord_feature_dim, dropout, no_bn=no_bn)

        self.point_spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.t_box_feature_fusion = nn.Sequential(
            nn.Linear(self.nr_frames*self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.classifier_p = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            # nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )

        if opt.fine_tune:
            self.fine_tune(opt.fine_tune)

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        #import pdb
        for k, v in weights.items():
            if not 'classifier.4' in k:
                new_weights[k.replace('module.', '')] = v
        #pdb.set_trace()
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'classifier.4' in name:

                param.requires_grad = False
                frozen_weights += 1

            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def node2edge(self, node_embeddings):
        send_embed = node_embeddings[:, self.send_edges, :]
        recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=3)

    def edge2node(self, edge_embeddings):
        if len(edge_embeddings.shape) == 4:         # edge_embeddings [N, V(V-1), T, Dc]
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)       # old_shape[1] = 12
            incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(old_shape[0], -1, old_shape[2], old_shape[3])
        else:
            incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming/(self.nr_boxes-1)

    def forward(self, global_img_input, box_categories, box_input, hand_id, video_label):
        box_categories = box_categories.cuda()

        video_label = video_label.cuda()
        box_input =box_input.cuda()
        hand_id = hand_id.cuda()

        b = hand_id.shape[0]

        N, E, T, V = b, 12, 16, 4
        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(b*self.nr_boxes*self.nr_frames, 4)

        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        box_categories = box_categories.view(b*self.nr_boxes*self.nr_frames)
        box_category_embeddings = self.category_embed_layer(box_categories)  # (b*nr_b*nr_f, coord_feature_dim//2)

        bf = self.coord_to_feature(box_input)       # [N*V*T,128]
        bf = torch.cat([bf, box_category_embeddings], dim=1)  # (b*nr_b*nr_f, coord_feature_dim + coord_feature_dim//2)
        bf = self.coord_category_fusion(bf)  # (b*nr_b*nr_f, coord_feature_dim)

        bf = bf.view(b, self.nr_boxes, self.nr_frames, self.coord_feature_dim)      # [N, V, T, D]

        x = self.node2edge(bf).reshape(N*E*T, -1)  # [N, V(V-1), T, D']
        x = self.mlp1(x)        # [N*V(V-1)*T, D]
        x = x.reshape(N, E, T, -1)
        x = self.edge2node(x)   # [N, V, T, D]
        x = x.reshape(N*V*T,-1)
        x = self.mlp2(x)        # [N, V, T, D]
        x = x.reshape(N, V, T, -1)
        # used in top branch
        point_x = torch.cat((x, bf), dim=-1)      # [N, V, T, D']

        point_x = point_x.reshape(N, V, T, -1)
        point_x = self.tmlp1(point_x)       # [N,V,T,-1]

        bf_spatial = self.point_spatial_node_fusion(point_x.view(b*self.nr_boxes*self.nr_frames, -1))
        bf_spatial = bf_spatial.view(b, self.nr_boxes, self.nr_frames, self.coord_feature_dim)
        bf_temporal_input = bf_spatial.view(b, self.nr_boxes, self.nr_frames*self.coord_feature_dim)
        box_features = self.t_box_feature_fusion(
            bf_temporal_input.view(b * self.nr_boxes, -1))  # (b*nr_boxes, coord_feature_dim)
        point_features = torch.mean(box_features.view(b, self.nr_boxes, -1), dim=1)  # (b, coord_feature_dim)

        pf = self.classifier_p(point_features)
        return pf, video_label.long()


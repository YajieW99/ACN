import torch
import torch.nn as nn
import numpy as np
import math
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
        self.longtailclassnum = 30
        dropout = 0.1
        no_bn = False

        self.soft = torch.nn.Softmax(0)

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
        self.mlp3 = RefNRIMLP(self.coord_feature_dim * 2, self.coord_feature_dim, self.coord_feature_dim, dropout, no_bn=no_bn)
        self.mlp4 = RefNRIMLP(self.coord_feature_dim * 2, self.coord_feature_dim, self.coord_feature_dim, dropout, no_bn=no_bn)

        self.box_feature_fusion_v = nn.Sequential(
            nn.Linear(self.nr_frames*self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.box_feature_fusion_p = nn.Sequential(
            nn.Linear(self.nr_frames*self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.mlp5 = nn.Sequential(
            nn.Linear(2*self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )

        self.mlp6 = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            # nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(),
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim, bias=False),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            # nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.nr_actions)
        )

        self.W_v = nn.Parameter(torch.tensor(np.zeros([self.longtailclassnum, self.nr_actions]).astype(np.float32)),requires_grad=True)
        self.W_p = nn.Parameter(torch.tensor(np.zeros([self.longtailclassnum, self.nr_actions]).astype(np.float32)),requires_grad=True)
        self.Wv = [[85,86,87,88,90,91,92,93],[131,132,133],[105],[109,111,112,113,114],[131,132,133],
              [85,86,87,88,90,91,92,93],[109,111,112,113,114],[85,86,87,88,90,91,92,93],
              [23,24,25],[131,132,133],[11,16,17,18,45,79,80,148],[85,86,87,88,90,91,92,93],
              [12,14,157,158],[23,24,25],[0,34,37,41,42,47],[11,16,17,18,45,79,80,148],[53,54,55,56,57],
              [9,10,13,15],[30,31,35,44],[52,52,],[11,16,17,18,45,79,80,148],[109],[110],
              [62,63,64,65,66,67,68,69,127],[140,142],[141],[32],[14],[116],[85,86,87,88,90,91,92,93]]

        self.Wp = [[87],[44,99,119,133,163],[6,105],[48,122,113,114,159],[43,47,100,109,117,131,161],
              [91],[56,62,97,112,115,132,160],[92,88],[25],[56,62,97,112,115,132,160],[16,17,18],[93],
              [14],[24,130],[42],[11,12,13],[56,62,97,112,115,132,160],[15,155,157],[31],
              [51],[56,62,97,112,115,132,160],[109],[110],[11,16,17,22,40,56,62,75,132,160],[140,142],
              [141],[32],[14],[116],[92,88]]

        for i in range(len(self.Wv)):
            wvcurrent = self.Wv[i]
            for j in wvcurrent:
                nn.init.constant_(self.W_v[i][j], 1/len(wvcurrent))
        for i in range(len(self.Wp)):
            wpcurrent = self.Wp[i]
            for j in wpcurrent:
                nn.init.constant_(self.W_p[i][j], 1/len(wpcurrent))

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
        if len(edge_embeddings.shape) == 4:         # edge_embeddings [N, V(V-1), T, Dc=128]
            old_shape = edge_embeddings.shape
            tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)       # old_shape[1] = 12
            incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(old_shape[0], -1, old_shape[2], old_shape[3])
        else:
            incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming/(self.nr_boxes-1)

    def forward(self, global_img_input, box_categories, box_input, hand_id, video_label, longtailclass,
                      longtaillabel, hand_vplist, flag, isval):
        # local_img_tensor is (b, nr_frames, nr_boxes, 3, h, w)
        # global_img_tensor is (b, nr_frames, 3, h, w)
        # box_input is (b, nr_frames, nr_boxes, 4)

        box_input =box_input.cuda()
        hand_id = hand_id.cuda()


        b = hand_id.shape[0]

        N, E, T, V = b, 12, 16, 4
        box_input = box_input.transpose(2, 1).contiguous()
        box_input = box_input.view(b*self.nr_boxes*self.nr_frames, 4)
        bf = self.coord_to_feature(box_input)       # [N*V*T,128]
        bf = bf.view(b, self.nr_boxes, self.nr_frames, self.coord_feature_dim)      # [N, V, T, D]

        x = self.node2edge(bf).reshape(N*E*T, -1)  # [N, V(V-1), T, D'=256]
        x = self.mlp1(x)        # [N*V(V-1)*T, D]
        x_skip = x      # 128
        x = x.reshape(N, E, T, -1)
        x = self.edge2node(x)   # [N, V, T, D]
        x = x.reshape(N*V*T,-1)
        x = self.mlp2(x)        # [N, V, T, D]
        x = x.reshape(N, V, T, -1)
        # used in top branch
        # point_x = torch.cat((x, bf), dim=-1)      # [N, V, T, D'=256]

        x = self.node2edge(x).reshape(N*E*T, -1)       # 256
        x = self.mlp3(x)                    # [xx, 128]
        x = torch.cat((x, x_skip), dim=-1)      # [N, E, T, D'=256]
        x = self.mlp4(x).reshape(N,E,T,-1).transpose(0,1).reshape(E, -1)         # [N, E, T, D]

        Dim = self.coord_feature_dim  #256
        zeros = torch.zeros(1, N * T * Dim).cuda()  # [1, 49152]
        edge_feature = torch.split(x, 4, dim=0)  

        edge_feature_0 = torch.cat((zeros, edge_feature[0]), dim=0) 
        edge_feature_1 = torch.cat((zeros, edge_feature[1]), dim=0)
        edge_feature_2 = torch.cat((zeros, edge_feature[2]), dim=0)

        edge_feature = torch.cat((edge_feature_0, edge_feature_1, edge_feature_2, zeros), dim=0) # [16, 49512]
        edge_feature = edge_feature.reshape(4, 4, N, T, Dim).permute(2, 0, 1, 3, 4).reshape(N * 4, -1)

        edge_hand = []
        edge_nohand = []

        hand_id_new = []
        nohand_index = []
        hand_index = []

        for h_index, h in enumerate(hand_id):
            if h == 5:
                edge_nohand.append(edge_feature[h_index * 4:(h_index * 4 + 4)])
                nohand_index.append(h_index)
            else:
                edge_hand.append(edge_feature[h_index * 4:(h_index * 4 + 4)])
                hand_id_new.append(h)
                hand_index.append(h_index)

        hand_id_new = torch.tensor(hand_id_new).cuda()
        hand_index = torch.tensor(hand_index).cuda()
        nohand_index = torch.tensor(nohand_index).cuda()

        if len(edge_hand) != 0:
            # [N*4, 4*T*Dim]
            edge_hand = torch.stack([edge_hand[i] for i in range(len(edge_hand))]).reshape(-1, 4*T*Dim).cuda()
            N_verb = int((edge_hand.shape[0]) / 4)
        else:
            edge_hand = None
            N_verb = 0

        if len(edge_nohand) != 0:
            # [N*4, 4*T*Dim]
            edge_nohand = torch.stack([edge_nohand[i] for i in range(len(edge_nohand))]).reshape(-1, 4*T*Dim).cuda()
            N_prep = int((edge_nohand.shape[0]) / 4)
        else:
            edge_nohand = None
            N_prep = 0

        hand_id_global = []
        if N_verb != 0:

            for i, id in enumerate(hand_id_new):
                hand_id_global.append(i * 4 + id)
            hand_id_global = torch.stack([hand_id_global[i] for i in range(len(hand_id_global))]).cuda()

            # dir verb subgraph
            verb_graph_edge = torch.zeros(N_verb * 4, 4 * T * Dim).cuda()
            verb_graph_edge[hand_id_global] = edge_hand[hand_id_global]

            verb_e = edge_hand[hand_id_global].reshape(N_verb,4,Dim*T).reshape(N_verb*4, Dim*T)
            verb_e1 = []
            for v in verb_e:
                if not v.equal(torch.zeros(Dim * T).cuda()) :
                    verb_e1.append(v)
            verb_e1 = torch.stack([verb_e1[i] for i in range(len(verb_e1))]).reshape(N_verb, 3, T * Dim).cuda()

            verb_graph_edge = verb_graph_edge.reshape(N_verb, 4, 4, T * Dim).permute(0, 2, 1, 3).reshape(N_verb * 4, 4 * T * Dim)
            # edge_hand_t
            edge_hand = edge_hand.reshape(N_verb,4,4,T*Dim).permute(0,2,1,3).reshape(N_verb * 4, 4 * T * Dim)
            verb_graph_edge[hand_id_global] = edge_hand[hand_id_global]
            verb_e = edge_hand[hand_id_global].reshape(N_verb,4,Dim*T).reshape(N_verb*4, Dim*T)
            verb_e2 = []
            for v in verb_e:
                if not v.equal(torch.zeros(Dim * T).cuda()):
                    verb_e2.append(v)
            verb_e2 = torch.stack([verb_e2[i] for i in range(len(verb_e2))]).reshape(N_verb, 3, T * Dim).cuda()
            verb_final = torch.cat((verb_e1, verb_e2), dim=1)       # [N_verb, 6, T*Dim]
            # prep subgraph
            prep_graph_edge = edge_hand - verb_graph_edge       # prep_graph_edge_t
            prep_graph_edge = prep_graph_edge.reshape(N_verb, 4, 4, -1)  # [N_verb, 4, 4, T*Dim]
            prep_e = []
            for n in range(N_verb):
                prep_e.append([])
            for n in range(N_verb):
                for i in range(4):
                    for j in range(4):
                        if not prep_graph_edge[n,i,j].equal(torch.zeros(T*Dim).cuda()):
                            prep_e[n].append(prep_graph_edge[n,i,j])
            for n in range(N_verb):
                prep_e[n] = torch.stack([prep_e[n][i] for i in range(len(prep_e[n]))])
            prep_final = torch.stack([prep_e[n] for n in range(len(prep_e))]).cuda()
            verb_final = self.box_feature_fusion_v(verb_final.reshape(N_verb * 6, -1))       # 128
            verb_final = torch.mean(verb_final.view(N_verb, 6, -1), dim=1)
            prep_final = self.box_feature_fusion_p(prep_final.reshape(N_verb * 6, -1))
            prep_final = torch.mean(prep_final.view(N_verb, 6, -1), dim=1)

            box_features = torch.cat((verb_final, prep_final),dim=-1)                # [N_verb, 256]
            box_features = self.mlp5(box_features)                                  # [N_verb, 128]
            cls_hand = self.classifier(box_features)                                  # [N_verb, 174]

            if len(cls_hand.shape) == 1:
                cls_hand = cls_hand.unsqueeze(0)

            video_label_hand = []
            for id in hand_index:
                video_label_hand.append(video_label[id])
            video_label_hand = torch.stack([video_label_hand[i] for i in range(len(video_label_hand))])

        else:
            cls_hand = None
            video_label_hand = None

        if N_prep != 0:
            edge_nohand = edge_nohand.reshape(N_prep, 4, 4, T * Dim)
            nohand_e = []

            for n in range(N_prep):
                nohand_e.append([])
            for n in range(N_prep):
                for i in range(4):
                    for j in range(4):
                        if not edge_nohand[n, i, j].equal(torch.zeros(T * Dim).cuda()):
                            nohand_e[n].append(edge_nohand[n, i, j])
            for n in range(N_prep):
                nohand_e[n] = torch.stack([nohand_e[n][i] for i in range(len(nohand_e[n]))])
            nohand_final = torch.stack([nohand_e[n] for n in range(len(nohand_e))]).cuda()
            nohand_final = nohand_final.reshape(N_prep*12,T*Dim)
            nohand_final = self.box_feature_fusion_p(nohand_final)
            nohand_final = torch.mean(nohand_final.view(N_prep, 12, -1), dim=1)
            nohand_final = self.mlp6(nohand_final)

            cls_nohand = self.classifier(nohand_final)

            if len(cls_nohand.shape) == 1:
                cls_nohand = cls_nohand.unsqueeze(0)

            video_label_nohand = []
            for id in nohand_index:
                video_label_nohand.append(video_label[id])

            if len(video_label_nohand) != 0:
                video_label_nohand = torch.stack([video_label_nohand[i] for i in range(len(video_label_nohand))])
            else:
                video_label_nohand = None
        else:
            cls_nohand = None
            video_label_nohand = None


        cls = torch.zeros(N, 174).cuda()
        label = torch.zeros(N, 1).cuda()
        if cls_hand is not None:
            for i in range(hand_index.shape[0]):
                cls[hand_index[i]] = cls_hand[i]
                label[hand_index[i]] = video_label_hand[i]
        if cls_nohand is not None:
            for i in range(nohand_index.shape[0]):
                cls[nohand_index[i]] = cls_nohand[i]
                label[nohand_index[i]] = video_label_nohand[i]
        label = label.squeeze()
        cert = torch.nn.CrossEntropyLoss()

        if not isval:
        # compositional branch
            verb_final_exp = []
            prep_final_exp = []
            v_inlongtail = []
            v_inlongtailindex = []
            p_inlongtail = []
            p_inlongtailindex = []
            k=0
            for i in range(len(hand_vplist)):
                v = hand_vplist[i,0]
                p = hand_vplist[i,1]
                if v in longtailclass[:,0]:
                    v_inlongtail.append(v)
                    v_inlongtailindex.append(i)
                if p in longtailclass[:,1]:
                    p_inlongtail.append(p)
                    p_inlongtailindex.append(i)
                if hand_id[i]!=5:
                    verb_final_exp.append(verb_final[k].cpu().detach())
                    prep_final_exp.append(prep_final[k].cpu().detach())
                    k=k+1
                else:
                    verb_final_exp.append(torch.zeros(256))     # cpu
                    prep_final_exp.append(torch.zeros(256))

            longtailclass_list = longtailclass.tolist()
            longtailclass0 = longtailclass.tolist()
            longtaillabel_list = longtaillabel.tolist()

            delindex = []
            for i, f in enumerate(flag):
                if f == 1:
                    delindex.append(i)
            for index in reversed(delindex):
                del longtailclass_list[index]
                del longtaillabel_list[index]

            if len(longtaillabel_list)!=0:
                feature_256 = []
                feature_256temp = []
                feature_128 = []
                feature_128temp = []
                vpl_256 = []
                vpl_128 = []
                complabel = []

                complabel256 = []
                complabel128 = []
                for vindex, v in enumerate(v_inlongtail):
                    for pindex, p in enumerate(p_inlongtail):
                        temp = [int(v),int(p)]
                        if temp in longtailclass_list:
                            if p !=175:
                                feature = torch.cat((verb_final_exp[v_inlongtailindex[vindex]],\
                                    prep_final_exp[p_inlongtailindex[pindex]]),dim=-1)
                                feature_256.append(feature)
                                feature_256temp.append(temp)
                                vpl_256.append([v, p, v_inlongtailindex[vindex], p_inlongtailindex[pindex],
                                                longtailclass0.index(temp)])

                            else:
                                feature_128.append(verb_final_exp[v_inlongtailindex[vindex]])
                                feature_128temp.append(temp)
                                vpl_128.append([v, p, v_inlongtailindex[vindex], p_inlongtailindex[pindex],
                                                longtailclass0.index(temp)])
                # set_trace()
                if len(feature_256temp) > 1:
                    for temp in feature_256temp:
                        complabel256.append(longtaillabel_list[int(longtailclass_list.index(temp))])
                if len(feature_128temp) != 0:
                    for temp in feature_128temp:
                        complabel128.append(longtaillabel_list[int(longtailclass_list.index(temp))])

                if len(complabel256) != 0:
                    complabel256 = torch.tensor(complabel256).cuda()
                else:
                    complabel256 = None

                if len(complabel128) != 0:
                    complabel128 = torch.tensor(complabel128).cuda()
                else:
                    complabel128 = None

                # set_trace()
                if len(feature_256)>1:
                    feature_256 = torch.stack([feature_256[i] for i in range(len(feature_256))]).cuda()
                    if len(feature_256)<10:
                        feature_256 = self.mlp5(feature_256)
                    else:
                        sampleind = [int(i) for i in np.linspace(0, len(feature_256)-1, 10)]
                        feature_256 = feature_256[sampleind]
                        feature_256 = self.mlp5(feature_256)
                        complabel256 = complabel256[sampleind]
                        vpl_256 = torch.tensor(vpl_256)[sampleind].tolist()
                else:
                    feature_256 = None
                    vpl_256 = []

                if len(feature_128)>1:
                    feature_128 = torch.stack([feature_128[i] for i in range(len(feature_128))]).cuda()
                    if len(feature_128)<10:
                        feature_128 = self.mlp6(feature_128)
                    else:
                        sampleind = [int(i) for i in np.linspace(0, len(feature_256)-1, 10)]
                        feature_128 = feature_256[sampleind]
                        feature_128 = self.mlp6(feature_128)
                        complabel128 = complabel128[sampleind]
                        vpl_128 = torch.tensor(vpl_128)[sampleind].tolist()
                else:
                    feature_128 = None
                    vpl_128 = []

                if feature_256 is not None and feature_128 is not None:
                    feature_comp = torch.cat((feature_256, feature_128), dim=0)
                    complabel = torch.cat((complabel256,complabel128), dim=0)
                elif feature_256 is not None and feature_128 is None:
                    feature_comp = feature_256
                    complabel = complabel256
                elif feature_256 is None and feature_128 is not None:
                    feature_comp = feature_128
                    complabel = complabel128
                else:
                    feature_comp = None
                    complabel = None
                if feature_comp is not None:
                    feature_comp = self.classifier(feature_comp)

                vpl = vpl_256 + vpl_128

                W_v_new_rev = torch.zeros([self.longtailclassnum, self.nr_actions]).cuda()
                W_p_new_rev = torch.zeros([self.longtailclassnum, self.nr_actions]).cuda()
                
                deno = 0
                for i in range(len(self.Wv)):
                    deno = 0
                    wvcurrent = self.Wv[i]
                    for j in wvcurrent:
                        deno = deno + abs(self.W_v[i][j])
                    W_v_new_rev[i] = abs(self.W_v[i])/deno

                deno = 0
                for i in range(len(self.Wp)):
                    deno = 0
                    wpcurrent = self.Wp[i]
                    for j in wpcurrent:
                        deno = deno + abs(self.W_p[i][j])
                    W_p_new_rev[i] = abs(self.W_p[i])/deno

                lossmatrix_v_all = torch.zeros(20, self.longtailclassnum, self.nr_actions).cuda()
                lossmatrix_p_all = torch.zeros(20, self.longtailclassnum, self.nr_actions).cuda()
                lossmatrix_v_all_W = torch.zeros(20, self.longtailclassnum, self.nr_actions).cuda()
                lossmatrix_p_all_W = torch.zeros(20, self.longtailclassnum, self.nr_actions).cuda()
                lossmatrix_final = torch.zeros(20, self.longtailclassnum, self.nr_actions).cuda()

                # set_trace()
                if len(vpl)!=0:
                    for i in range(len(vpl)): 
                        x = vpl[i][4]  
                        yv = video_label[vpl[i][2]] 
                        fea = feature_comp[i].unsqueeze(0) 
                        la= torch.tensor([complabel[i]]).cuda() 
                        loss_current = cert(fea,la.long())  
                        lossmatrix_v_all[i][x][yv] = loss_current
                    for i in range(lossmatrix_v_all.shape[0]):
                        lossmatrix_v_all_W[i] = lossmatrix_v_all[i] * W_v_new_rev
                    for i in range(len(vpl)):
                        x = vpl[i][4]
                        yv = video_label[vpl[i][2]]
                        yp = video_label[vpl[i][3]]
                        lossmatrix_p_all[i][x][yp] = lossmatrix_v_all[i][x][yv]  
                    for i in range(lossmatrix_p_all.shape[0]):
                        lossmatrix_p_all_W[i] = lossmatrix_p_all[i] * W_p_new_rev
                    for i in range(len(vpl)):
                        lossmatrix_final[i] = lossmatrix_v_all_W[i] + lossmatrix_p_all_W[i] 

                    loss_wvpcomp = torch.sum(lossmatrix_final)/len(complabel)

                else:
                    loss_wvpcomp = None
                    feature_comp = None
                    complabel = None

            else:
                loss_wvpcomp = None
                feature_comp = None
                complabel = None

        else:
            loss_wvpcomp = None
            feature_comp = None
            complabel = None

        return cls, label.long(), feature_comp, complabel, loss_wvpcomp, flag



import torch
import numpy as np

from typing import Dict
from torch import nn
from torch.nn import Parameter
from utils.utils_gcn import get_param
from .gnn_layer import StarEConvLayer

class StarEBase(torch.nn.Module):
    def __init__(self, config):
        super(StarEBase, self).__init__()
        """ Not saving the config dict bc model saving can get a little hairy. """

        self.act = torch.tanh if 'ACT' not in config['STAREARGS'].keys() \
            else config['STAREARGS']['ACT']
        self.bceloss = torch.nn.BCELoss()

        self.emb_dim = config['EMBEDDING_DIM']
        self.num_rel = config['NUM_RELATIONS']
        self.num_ent = config['NUM_ENTITIES']
        self.n_bases = config['STAREARGS']['N_BASES']
        self.n_layer = config['STAREARGS']['LAYERS']
        self.gcn_dim = config['STAREARGS']['GCN_DIM']
        self.hid_drop = config['STAREARGS']['HID_DROP']
        # self.bias = config['STAREARGS']['BIAS']
        self.model_nm = config['MODEL_NAME'].lower()
        self.triple_mode = config['STATEMENT_LEN'] == 3
        self.qual_mode = config['STAREARGS']['QUAL_REPR']

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class StarEEncoder(StarEBase):
    def __init__(self, graph_repr: Dict[str, np.ndarray], config: dict, timestamps: dict = None):
        super().__init__(config)

        self.device = config['DEVICE']

        # Storing the KG
        self.edge_index = torch.tensor(graph_repr['edge_index'], dtype=torch.long, device=self.device)
        self.edge_type = torch.tensor(graph_repr['edge_type'], dtype=torch.long, device=self.device)

        if not self.triple_mode:
            if self.qual_mode == "full":
                self.qual_rel = torch.tensor(graph_repr['qual_rel'], dtype=torch.long, device=self.device)
                self.qual_ent = torch.tensor(graph_repr['qual_ent'], dtype=torch.long, device=self.device)
            elif self.qual_mode == "sparse":
                self.quals = torch.tensor(graph_repr['quals'], dtype=torch.long, device=self.device)

        self.gcn_dim = self.emb_dim if self.n_layer == 1 else self.gcn_dim

        if timestamps is None:
            self.init_embed = get_param((self.num_ent, self.emb_dim))
            self.init_embed.data[0] = 0  # padding



        if self.model_nm.endswith('transe'):
            self.init_rel = get_param((self.num_rel, self.emb_dim))
        elif config['STAREARGS']['OPN'] == 'rotate' or config['STAREARGS']['QUAL_OPN'] == 'rotate':
            phases = 2 * np.pi * torch.rand(self.num_rel, self.emb_dim // 2)
            self.init_rel = nn.Parameter(torch.cat([
                torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1),
                torch.cat([torch.cos(phases), -torch.sin(phases)], dim=-1)
            ], dim=0))
        else:
            self.init_rel = get_param((self.num_rel * 2, self.emb_dim))

        self.init_rel.data[0] = 0 # padding

        self.conv1 = StarEConvLayer(self.emb_dim, self.gcn_dim, self.num_rel, act=self.act,
                                       config=config)
        self.conv2 = StarEConvLayer(self.gcn_dim, self.emb_dim, self.num_rel, act=self.act,
                                       config=config) if self.n_layer == 2 else None

        if self.conv1: self.conv1.to(self.device)
        if self.conv2: self.conv2.to(self.device)

        self.register_parameter('bias', Parameter(torch.zeros(self.num_ent)))

    def forward_base(self, sub, rel, drop1, drop2,
                     quals=None, embed_qualifiers: bool = False, return_mask: bool = False):
        """"
        :param sub:
        :param rel:
        :param drop1:
        :param drop2:
        :param quals: (optional) (bs, maxqpairs*2) Each row is [qp, qe, qp, qe, ...]
        :param embed_qualifiers: if True, we also indexselect qualifier information
        :param return_mask: if True, returns a True/False mask of [bs, total_len] that says which positions were padded
        :return:
        """
        r = self.init_rel if not self.model_nm.endswith('transe') \
            else torch.cat([self.init_rel, -self.init_rel], dim=0)

        if not self.triple_mode:
            if self.qual_mode == "full":
                # x, edge_index, edge_type, rel_embed, qual_ent, qual_rel
                x, r = self.conv1(x=self.init_embed, edge_index=self.edge_index,
                                  edge_type=self.edge_type, rel_embed=r,
                                  qualifier_ent=self.qual_ent,
                                  qualifier_rel=self.qual_rel,
                                  quals=None)

                x = drop1(x)
                x, r = self.conv2(x=x, edge_index=self.edge_index,
                                  edge_type=self.edge_type, rel_embed=r,
                                  qualifier_ent=self.qual_ent,
                                  qualifier_rel=self.qual_rel,
                                  quals=None) if self.n_layer == 2 else (x, r)
            elif self.qual_mode == "sparse":
                # x, edge_index, edge_type, rel_embed, qual_ent, qual_rel
                x, r = self.conv1(x=self.init_embed, edge_index=self.edge_index,
                                  edge_type=self.edge_type, rel_embed=r,
                                  qualifier_ent=None,
                                  qualifier_rel=None,
                                  quals=self.quals)

                x = drop1(x)
                x, r = self.conv2(x=x, edge_index=self.edge_index,
                                  edge_type=self.edge_type, rel_embed=r,
                                  qualifier_ent=None,
                                  qualifier_rel=None,
                                  quals=self.quals) if self.n_layer == 2 else (x, r)

        else:
            x, r = self.conv1(x=self.init_embed, edge_index=self.edge_index,
                              edge_type=self.edge_type, rel_embed=r)

            x = drop1(x)
            x, r = self.conv2(x=x, edge_index=self.edge_index,
                              edge_type=self.edge_type, rel_embed=r) \
                if self.n_layer == 2 else \
                (x, r)

        x = drop2(x) if self.n_layer == 2 else x

        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)

        if embed_qualifiers:
            assert quals is not None, "Expected a tensor as quals."
            # flatten quals
            quals_ents = quals[:, 1::2].view(1,-1).squeeze(0)
            quals_rels = quals[:, 0::2].view(1,-1).squeeze(0)
            qual_obj_emb = torch.index_select(x, 0, quals_ents)
            # qual_obj_emb = torch.index_select(x, 0, quals[:, 1::2])
            qual_rel_emb = torch.index_select(r, 0, quals_rels)
            qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1 ,sub_emb.shape[1])
            qual_rel_emb = qual_rel_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])
            if not return_mask:
                return sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, x
            else:
                # mask which shows which entities were padded - for future purposes, True means to mask (in transformer)
                # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py : 3770
                # so we first initialize with False
                mask = torch.zeros((sub.shape[0], quals.shape[1] + 2)).bool().to(self.device)
                # and put True where qual entities and relations are actually padding index 0
                mask[:, 2:] = quals == 0
                return sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, x, mask

        return sub_emb, rel_emb, x


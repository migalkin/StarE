
# Torch imports
import torch
from torch import nn
import torch.autograd
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

import numpy as np
from typing import List, Optional, Dict, Tuple

# Local imports
from utils.utils_gcn import get_param, scatter_add, MessagePassing, ccorr, rotate
from utils.utils import *
from utils.utils_mytorch import compute_mask
from .gnn_encoder import StarEEncoder, StarEBase


class StarEConvEStatement(StarEEncoder):
    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict):
        super(self.__class__, self).__init__(kg_graph_repr, config)

        self.hid_drop2 = config['STAREARGS']['HID_DROP2']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        self.n_filters = config['STAREARGS']['N_FILTERS']
        self.kernel_sz = config['STAREARGS']['KERNEL_SZ']
        # self.bias = config['STAREARGS']['BIAS']
        self.k_w = config['STAREARGS']['K_W']
        self.k_h = config['STAREARGS']['K_H']

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.n_filters)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.n_filters,
                                       kernel_size=(self.kernel_sz, self.kernel_sz), stride=1,
                                       padding=0, bias=config['STAREARGS']['BIAS'])
        assert 2 * self.k_w > self.kernel_sz and self.k_h > self.kernel_sz, "kernel size is incorrect"
        assert self.emb_dim * (config['MAX_QPAIRS'] - 1) == 2 * self.k_w * self.k_h, "Incorrect combination of conv params and emb dim " \
                                                    " ConvE decoder will not work properly, " \
                                                    " should be emb_dim * (pairs - 1) == 2* k_w * k_h"

        flat_sz_h = int(2 * self.k_w) - self.kernel_sz + 1
        flat_sz_w = self.k_h - self.kernel_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.n_filters
        self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)

    def concat(self, e1_embed, rel_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        """
            arrange quals in the conve format with shape [bs, num_qual_pairs, emb_dim]
            num_qual_pairs is 2 * (any qual tensor shape[1])
            for each datum in bs the order will be 
                rel1, emb
                en1, emb
                rel2, emb
                en2, emb
        """
        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2).view(-1, 2*qual_rel_embed.shape[1], qual_rel_embed.shape[2])
        stack_inp = torch.cat([e1_embed, rel_embed, quals], 1)  # [bs, 2 + num_qual_pairs, emb_dim]
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.k_w, self.k_h))
        return stack_inp

    def forward(self, sub, rel, quals):
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True)
        stk_inp = self.concat(sub_emb, rel_emb, qual_rel_emb, qual_obj_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score


class StarE_ConvKB_Statement(StarEEncoder):
    model_name = 'StarE_ConvKB_Statement'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict):
        super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'StarE_ConvKB_Statement'
        self.hid_drop2 = config['STAREARGS']['HID_DROP2']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        self.n_filters = config['STAREARGS']['N_FILTERS']
        self.kernel_sz = config['STAREARGS']['KERNEL_SZ']
        # self.bias = config['STAREARGS']['BIAS']

        self.pooling = config['STAREARGS']['POOLING']

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.n_filters)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.n_filters,
                                       kernel_size=(config['MAX_QPAIRS']-1, self.kernel_sz), stride=1,
                                       padding=0, bias=config['STAREARGS']['BIAS'])
        self.flat_sz = self.n_filters * (self.emb_dim - self.kernel_sz + 1)
        self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)

    def concat(self, e1_embed, rel_embed, qual_rel_embed, qual_obj_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        """
            arrange quals in the conve format with shape [bs, num_qual_pairs, emb_dim]
            num_qual_pairs is 2 * (any qual tensor shape[1])
            for each datum in bs the order will be 
                rel1, emb
                en1, emb
                rel2, emb
                en2, emb
        """
        quals = torch.cat((qual_rel_embed, qual_obj_embed), 2).view(-1, 2*qual_rel_embed.shape[1], qual_rel_embed.shape[2])
        stack_inp = torch.cat([e1_embed, rel_embed, quals], 1).unsqueeze(1)  # [bs, 1, 2 + num_qual_pairs, emb_dim]
        return stack_inp

    def forward(self, sub, rel, quals):
        sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, all_ent = \
            self.forward_base(sub, rel, self.hidden_drop, self.feature_drop, quals, True)
        stk_inp = self.concat(sub_emb, rel_emb, qual_rel_emb, qual_obj_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score


class StarE_Transformer_Triples(StarEEncoder):
    model_name = 'StarE_Transformer_Triples'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict):

        super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'StarE_Transformer_Triples'
        self.hid_drop2 = config['STAREARGS']['HID_DROP2']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        self.num_transformer_layers = config['STAREARGS']['T_LAYERS']
        self.num_heads = config['STAREARGS']['T_N_HEADS']
        self.num_hidden = config['STAREARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['STAREARGS']['POSITIONAL']
        self.pooling = config['STAREARGS']['POOLING']  # min / avg / concat

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['STAREARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['STAREARGS']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'] - 1, self.d_model)

        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        if self.pooling == "concat":
            self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
            self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        else:
            self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)


    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)

        stack_inp = torch.cat([e1_embed, rel_embed], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        return stack_inp

    def forward(self, sub, rel):
        '''

        :param sub: bs
        :param rel: bs
        :return:

        '''
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)

        # bs*emb_dim , ......, bs*6*emb_dim

        stk_inp = self.concat(sub_emb, rel_emb)

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        x = self.encoder(stk_inp)

        if self.pooling == 'concat':
            x = x.transpose(1, 0).reshape(-1, self.flat_sz)
        elif self.pooling == "avg":
            x = torch.mean(x, dim=0)
        elif self.pooling == "min":
            x, _ = torch.min(x, dim=0)

        x = self.fc(x)

        x = torch.mm(x, all_ent.transpose(1, 0))

        score = torch.sigmoid(x)
        return score


class Transformer_Baseline(StarEBase):
    def __init__(self, config: dict):
        super().__init__(config)

        self.entities = get_param((self.num_ent, self.emb_dim))
        self.relations = get_param((2 * self.num_rel, self.emb_dim))

        self.model_name = 'CompGCN_Transformer_Statement'
        self.hid_drop2 = config['STAREARGS']['HID_DROP2']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        self.num_transformer_layers = config['STAREARGS']['T_LAYERS']
        self.num_heads = config['STAREARGS']['T_N_HEADS']
        self.num_hidden = config['STAREARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['STAREARGS']['POSITIONAL']
        self.pooling = config['STAREARGS']['POOLING']  # min / avg / concat
        self.device = config['DEVICE']

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden,
                                                 config['STAREARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['STAREARGS']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'] - 1, self.d_model)
        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        if self.pooling == "concat":
            self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
            self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        else:
            self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1).transpose(1, 0)  # [2, bs, emb_dim]
        return stack_inp

    def forward(self, sub, rel):

        sub_emb = torch.index_select(self.entities, 0, sub)
        rel_emb = torch.index_select(self.relations, 0, rel)

        # bs*emb_dim , ......, bs*6*emb_dim

        stk_inp = self.concat(sub_emb, rel_emb)
        mask = torch.zeros((sub.shape[0], 2)).bool().to(self.device)
        #mask = mask[:, :2]

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        x = self.encoder(stk_inp, src_key_padding_mask=mask)

        if self.pooling == 'concat':
            x = x.transpose(1, 0).reshape(-1, self.flat_sz)
        elif self.pooling == "avg":
            x = torch.mean(x, dim=0)
        elif self.pooling == "min":
            x, _ = torch.min(x, dim=0)

        x = self.fc(x)

        x = torch.mm(x, self.entities.transpose(1, 0))

        score = torch.sigmoid(x)
        return score


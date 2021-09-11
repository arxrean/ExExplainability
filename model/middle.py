import os
import pdb
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.gcn_base import GraphConvolution, GraphAttention


class WeightFusion(nn.Module):
    def __init__(self, opt):
        super(WeightFusion, self).__init__()
        self.opt = opt
        self.n_hidden = 256
        self.gcn = GCN(opt)

        self.comment_atten = nn.Sequential(
            nn.Linear(self.n_hidden*2, 1),
        )

        self.content_atten = nn.Sequential(
            nn.Linear(self.n_hidden*2, 1),
        )

        self.text_disc = nn.Sequential(
            nn.Linear(self.n_hidden*4, 2),
        )

        self.img_disc = nn.Sequential(
            nn.Linear(self.n_hidden*4, 2),
        )

        self.cow = nn.Parameter(torch.zeros(512, 512))

    def forward(self, text, img, comment, comment_num, adj, score=False):
        comment = torch.split(comment, comment_num)
        text, img, commentq = self.graph_pass(text, img, comment, adj)

        return self.get_weighted_refine(text, img, comment)

    # ex atten
    def get_weighted_refine(self, text, img, comment):
        res = []
        contentw_res = []
        for x, y, z in zip(text, img, comment):
            content = torch.stack([x, y])
            co_w = torch.tanh(
                torch.mm(torch.mm(content, self.cow), z.transpose(0, 1)))
            contentw = F.softmax(self.comment_atten(
                torch.tanh(content + torch.mm(co_w, z))), 0)
            zw = F.softmax(self.content_atten(torch.tanh(
                z + torch.mm(co_w.transpose(0, 1), content))), 0)

            refine_content = torch.sum(content*contentw, 0)
            refine_comment = torch.sum(z*zw, 0)
            res.append(torch.stack([refine_content, refine_comment]))
            contentw_res.append(contentw)

        res = torch.stack(res)
        contentw_res = torch.stack(contentw_res)
        return res[:, 0], res[:, 1], contentw_res, None

    def graph_pass(self, text, img, comment, adj):
        res_text, res_img, res_c = [], [], []
        for x, y, z, a in zip(text, img, comment, adj):
            textc = torch.cat([x.unsqueeze(0), z], 0)
            imgc = torch.cat([y.unsqueeze(0), z], 0)
            _text, _c1 = self.gcn(textc, a)
            _img, _c2 = self.gcn(imgc, a)
            res_text.append(_text)
            res_img.append(_img)
            res_c.append(torch.cat([_c1, _c2], 0))

        res_text = torch.stack(res_text)
        res_img = torch.stack(res_img)

        return res_text, res_img, res_c

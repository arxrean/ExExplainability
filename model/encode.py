import os
import pdb
import math
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# the input data is first processed by the below class
class DoubleEmbed(nn.Module):
    def __init__(self, opt):
        super(DoubleEmbed, self).__init__()
        self.opt = opt
        self.n_hidden = 256
        self.emb = nn.Embedding(opt.max_vocab+2, 100)
        self.emb_drop = nn.Dropout(opt.dropout)
        if opt.gru_layer == 1:
            self.gru = nn.GRU(100, self.n_hidden, batch_first=True,
                              bidirectional=True)
        else:
            self.gru = nn.GRU(100, self.n_hidden, batch_first=True,
                              bidirectional=True, num_layers=opt.gru_layer, dropout=opt.dropout)

        self.atten = nn.Sequential(
            nn.Linear(self.n_hidden*2, self.n_hidden),
            nn.Tanh(),
            nn.Linear(self.n_hidden, 1),
        )

        self.resnet = self.get_resnet18()
        if opt.resnet_freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, text, img, comments, text_lens, comment_lens, atten_res=False):
        img_feat = self.resnet(img).squeeze(-1).squeeze(-1)

        text_embs = self.emb_drop(self.emb(text))
        text_embs = pack_padded_sequence(
            text_embs, text_lens, batch_first=True, enforce_sorted=False)

        comment_embs = self.emb_drop(self.emb(comments))
        comment_embs = pack_padded_sequence(
            comment_embs, comment_lens, batch_first=True, enforce_sorted=False)

        if atten_res:
            text_embs, text_w = self.gru_process(text_embs, atten_res)
            comment_embs, comment_w = self.gru_process(comment_embs, atten_res)

            return text_embs, img_feat, comment_embs, text_w, comment_w
        else:
            text_embs = self.gru_process(text_embs)
            comment_embs = self.gru_process(comment_embs)

            return text_embs, img_feat, comment_embs

    def get_atten_weight(self, gru_out):
        b, t, d = gru_out.shape
        gru_out = gru_out.reshape(-1, d)
        gru_out = self.atten(gru_out)
        gru_out = F.softmax(gru_out.reshape(b, t, 1), 1)

        return gru_out

    def get_resnet18(self):
        net = ptcv_get_model("resnet18", pretrained=True)
        return nn.Sequential(*list(net.children())[:-1])
        # return nn.Sequential(*list(net.children())[0][:-1])

    def gru_process(self, embs, atten_res=False):
        gru_out, self.h = self.gru(embs)
        gru_out, lengths = pad_packed_sequence(gru_out, batch_first=True)
        atten_w = self.get_atten_weight(gru_out)
        gru_out = torch.bmm(gru_out.transpose(1, 2), atten_w).squeeze(-1)

        if atten_res:
            return gru_out, atten_w
        else:
            return gru_out

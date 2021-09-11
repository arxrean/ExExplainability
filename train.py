import os
import pdb
import numpy as np
from tqdm import tqdm

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import utils
from option import get_parser


def train_double(opt):
    utils.init_log_dir(opt)
    writer = SummaryWriter('./save/{}/tb'.format(opt.name))

    encode, middle, decode = utils.get_model(opt)
    if opt.gpu:
        encode, middle, decode = encode.cuda(), middle.cuda(), decode.cuda()

    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam([{'params': encode.parameters()},
                            {'params': middle.parameters()},
                            {'params': decode.parameters()}], opt.base_lr, weight_decay=opt.weight_decay)

    # k fold
    trainset, valset, testset = utils.get_dataset(opt)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size,
                             shuffle=True, num_workers=opt.num_workers, collate_fn=utils.DoublePadSequence(), drop_last=True)
    valloader = DataLoader(valset, batch_size=opt.batch_size,
                           shuffle=False, num_workers=opt.num_workers, collate_fn=utils.DoublePadSequence())

    best_auc = 0.
    for epoch in range(opt.epoches):
        encode, middle, decode = encode.train(), middle.train(), decode.train()
        for step, pack in enumerate(trainloader):
            texts, comments, labels, text_lens, comment_lens, comment_num, img, batch_p, fake_labels, graph = pack
            if opt.gpu:
                texts, comments, labels, img = texts.cuda(
                ), comments.cuda(), labels.cuda(), img.cuda()
                graph = [x.cuda() for x in graph]

            text_embs, img_feat, comment_embs = encode(
                texts, img, comments, text_lens, comment_lens)
            text_embs, img_feat, claim_out, img_out = middle(
                text_embs, img_feat, comment_embs, comment_num, graph)
            if img_feat is not None:
                out = torch.cat([text_embs, img_feat], -1)
            else:
                out = text_embs
            out = decode(out)
            out = out*2 + claim_out[:, 0] + claim_out[:, 1]

            loss = loss_func(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('train/loss', loss.item(),
                              epoch*len(trainloader)+step)

        encode, middle, decode = encode.eval(), middle.eval(), decode.eval()
        with torch.no_grad():
            res, gt = [], []
            for step, pack in enumerate(valloader):
                texts, comments, labels, text_lens, comment_lens, comment_num, img, batch_p, fake_labels, graph = pack
                if opt.gpu:
                    texts, comments, img = texts.cuda(), comments.cuda(), img.cuda()
                    graph = [x.cuda() for x in graph]

                text_embs, img_feat, comment_embs = encode(
                    texts, img, comments, text_lens, comment_lens)
                text_embs, img_feat, claim_out, img_out = middle(
                    text_embs, img_feat, comment_embs, comment_num, graph)
                if img_feat is not None:
                    out = torch.cat([text_embs, img_feat], -1)
                else:
                    out = text_embs
                out = decode(out)
                # out = out*2 + claim_out[:, 0] + claim_out[:, 1]
                out = out.cpu().numpy()[:, 1]
                res.append(out)
                gt.append(labels.numpy())

            res = np.concatenate(res)
            gt = np.concatenate(gt)
            auc = roc_auc_score(gt, res)
            writer.add_scalar('val/auc', auc, epoch)
            if auc >= best_auc:
                best_auc = auc
                torch.save({'encode': encode.module.state_dict() if opt.gpus else encode.state_dict(),
                            'middle': middle.module.state_dict() if opt.gpus else middle.state_dict(),
                            'decode': decode.module.state_dict() if opt.gpus else decode.state_dict(),
                            'optimizer': optimizer}, os.path.join('./save', opt.name, 'check', 'best.pth.tar'))

        print('epoch:{} best_auc:{}'.format(epoch, best_auc))

    writer.flush()
    writer.close()


if __name__ == '__main__':
    opt = get_parser()
    train_double(opt)

import pdb
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class StackedAutoEncoder(nn.Module):
    def __init__(self, opt):
        super(StackedAutoEncoder, self).__init__()
        self.opt = opt
        self.encoder1 = nn.Conv2d(1, 4, 3, 1, 0)
        self.encoder2 = nn.Sequential(
            nn.Linear(4*98*8, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.decoder1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 4*98*8)
        )
        self.decoder2 = nn.ConvTranspose2d(4, 1, 3, 1, 0)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.encoder1(x)
        out = self.encoder2(out.view(out.size(0), -1))
        dout = self.decoder1(out)
        dout = self.decoder2(dout.view(dout.size(0), 4, 98, 8))

        return dout.squeeze(1)

    def get_feat(self, x):
        x = x.unsqueeze(1)
        out = self.encoder1(x)
        out = self.encoder2(out.view(out.size(0), -1))

        return out

    def reconstruct(self, x):
        a2_reconstruct = self.ae3.reconstruct(x)
        a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
        x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
        return x_reconstruct


class FullConnect(nn.Module):
    def __init__(self, opt):
        super(FullConnect, self).__init__()
        self.opt = opt
        self.w = nn.Sequential(
            nn.Linear(968, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        
        return self.w(x)
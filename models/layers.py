import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRA(nn.Module):
    def __init__(self, d_latent, rank=16):
        super().__init__()
        self.mat_A = nn.Parameter(torch.randn(d_latent, rank) / 50)  # match the scale in misc.py
        self.mat_B = nn.Parameter(torch.zeros(rank, d_latent))

    def forward(self, h):
        h = F.linear(h, self.mat_A @ self.mat_B)
        return h

    def get_lr(self, h):
        h = F.linear(h, self.mat_A.T)
        return h
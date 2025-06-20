import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRA(nn.Module):
    def __init__(self, d_embed, rank=16):
        super().__init__()
        self.mat_A = nn.Parameter(torch.randn(d_embed, rank) / 50)  # match the scale in initialization.py
        self.mat_B = nn.Parameter(torch.zeros(rank, d_embed))

    def forward(self, h):
        h = F.linear(h, self.mat_A @ self.mat_B)
        return h
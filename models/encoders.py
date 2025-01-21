import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """ SwiGLU-FFN """
    def __init__(self, d_latent, d_ffn=680):
        super().__init__()
        self.d_latent = d_latent
        self.d_ffn = d_ffn

        self.fc_1 = nn.Linear(d_latent, self.d_ffn, bias=False)
        self.fc_2 = nn.Linear(d_latent, self.d_ffn, bias=False)
        self.fc_3 = nn.Linear(self.d_ffn, d_latent, bias=False)

    def forward(self, h):
        h = self.fc_3(F.silu(self.fc_2(h)) * self.fc_1(h))
        return h


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mha = nn.MultiheadAttention(args.d_latent, args.n_head, batch_first=True)
        self.dropout = nn.Dropout(args.dropout)
        self.norm_mha = nn.LayerNorm(args.d_latent)

        self.register_buffer('mask_causal',
                             torch.triu(torch.full((args.len_trim,
                                                    args.len_trim), True), diagonal=1),
                             persistent=False)

    def forward(self, h, mask):
        h_mha = self.norm_mha(h + self.dropout(self.mha(h, h, h,
                                                    attn_mask=self.mask_causal,
                                                    is_causal=True,
                                                    need_weights=False)[0])) * mask
        return h_mha

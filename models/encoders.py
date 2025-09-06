import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    SwiGLU-FFN
    Require manual Post-norm residual connection.
    """
    def __init__(self,
                 d_embed: int,
                 d_ffn: int = 680,
                 ) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.d_ffn = d_ffn

        self.fc_1 = nn.Linear(d_embed, self.d_ffn, bias=False)
        self.fc_2 = nn.Linear(d_embed, self.d_ffn, bias=False)
        self.fc_3 = nn.Linear(self.d_ffn, d_embed, bias=False)

    def forward(self,
                h: torch.Tensor,
                ) -> torch.Tensor:
        h = self.fc_3(F.silu(self.fc_2(h)) * self.fc_1(h))
        return h


class MultiHeadAttention(nn.Module):
    """
    Post-norm residual connection integrated.
    """
    def __init__(self,
                 d_embed: int,
                 n_head: int,
                 len_trim: int,
                 dropout: float = 0.1,
                 ) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.n_head = n_head
        self.len_trim = len_trim

        self.mha = nn.MultiheadAttention(self.d_embed, self.n_head, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm_mha = nn.LayerNorm(self.d_embed)

        self.register_buffer('mask_causal',
                             torch.triu(torch.full((self.len_trim,
                                                    self.len_trim), True), diagonal=1),
                             persistent=False)

    def forward(self,
                h: torch.Tensor,
                mask: torch.Tensor,
                )-> torch.Tensor:
        h_mha = self.norm_mha(h + self.dropout(self.mha(h, h, h,
                                                    attn_mask=self.mask_causal,
                                                    is_causal=True,
                                                    need_weights=False)[0])) * mask
        return h_mha

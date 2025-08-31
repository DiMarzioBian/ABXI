from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import MultiHeadAttention, FeedForward
from models.layers import LoRA
from models.utils.position import get_absolute_pos_idx

from models.data.evaluation import cal_norm_mask
from models.utils.initialization import init_weights


class ABXI(nn.Module):
    def __init__(self,
                 args: Namespace,
                 ) -> None:
        super().__init__()
        self.bs = args.bs
        self.len_trim = args.len_trim
        self.n_item = args.n_item
        self.n_item_a = args.n_item_a
        self.n_item_b = args.n_item_b
        self.n_neg = args.n_neg
        self.temp = args.temp
        self.dropout = nn.Dropout(p=args.dropout) if args.dropout  > 0. else nn.Identity()

        self.d_embed = args.d_embed
        self.rd = args.rd
        self.ri = args.ri


        # item and positional embedding
        self.ei = nn.Embedding(self.n_item + 1, self.d_embed, padding_idx=0)
        self.ep = nn.Embedding(self.len_trim + 1, self.d_embed, padding_idx=0)

        # encoder, dlora
        self.mha = MultiHeadAttention(args)
        self.ffn = FeedForward(self.d_embed)

        self.dlora_x = LoRA(self.d_embed, self.rd)
        self.dlora_a = LoRA(self.d_embed, self.rd)
        self.dlora_b = LoRA(self.d_embed, self.rd)

        self.norm_sa_x = nn.LayerNorm(self.d_embed)
        self.norm_sa_a = nn.LayerNorm(self.d_embed)
        self.norm_sa_b = nn.LayerNorm(self.d_embed)

        # ilora
        self.ilora_a = LoRA(self.d_embed, self.ri)
        self.ilora_b = LoRA(self.d_embed, self.ri)

        # proj
        self.proj_i = FeedForward(self.d_embed)
        self.proj_a = FeedForward(self.d_embed)
        self.proj_b = FeedForward(self.d_embed)

        self.norm_i2a = nn.LayerNorm(self.d_embed)
        self.norm_i2b = nn.LayerNorm(self.d_embed)
        self.norm_a2a = nn.LayerNorm(self.d_embed)
        self.norm_b2b = nn.LayerNorm(self.d_embed)

        self.apply(init_weights)

    def embed_pos(self,
                  mask: torch.Tensor,
                  ) -> torch.Tensor:
        return self.ep(get_absolute_pos_idx(mask))

    def forward(self,
                seq_x: torch.Tensor,
                seq_a: torch.Tensor,
                seq_b: torch.Tensor,
                mask_x: torch.Tensor,
                mask_a: torch.Tensor,
                mask_b: torch.Tensor,
                mask_gt_a: torch.Tensor,
                mask_gt_b: torch.Tensor,
                ) -> torch.Tensor:

        # embedding
        h_x = (self.ei(seq_x) + self.embed_pos(mask_x)) * mask_x
        h_a = (self.ei(seq_a) + self.embed_pos(mask_a)) * mask_a
        h_b = (self.ei(seq_b) + self.embed_pos(mask_b)) * mask_b

        h_x = self.dropout(h_x)
        h_a = self.dropout(h_a)
        h_b = self.dropout(h_b)

        # mha
        h_x = self.mha(h_x, mask_x)
        h_a = self.mha(h_a, mask_a)
        h_b = self.mha(h_b, mask_b)

        # switch training / evaluating
        if self.training:
            mask_gt_a = mask_gt_a.unsqueeze(-1)
            mask_gt_b = mask_gt_b.unsqueeze(-1)

        else:
            mask_x = mask_a = mask_b = 1
            h_x = h_x[:, -1]
            h_a = h_a[:, -1]
            h_b = h_b[:, -1]

        # ffn + dlora
        h_x = self.norm_sa_x(h_x +
                             self.dropout(self.ffn(h_x)) +
                             self.dropout(self.dlora_x(h_x))
                             ) * mask_x

        h_a = self.norm_sa_a(h_a +
                             self.dropout(self.ffn(h_a)) +
                             self.dropout(self.dlora_a(h_a))
                             ) * mask_a

        h_b = self.norm_sa_b(h_b +
                             self.dropout(self.ffn(h_b)) +
                             self.dropout(self.dlora_b(h_b))
                             ) * mask_b

        # projector + ilora
        h_i = self.proj_i(h_x)

        h_a = (self.norm_i2a((h_x +
                              self.dropout(h_i) +
                              self.dropout(self.ilora_a(h_x))
                              ) * mask_gt_a) +
               self.norm_a2a((h_a +
                              self.dropout(self.proj_a(h_a))
                              ) * mask_gt_a))

        h_b = (self.norm_i2b((h_x +
                              self.dropout(h_i) +
                              self.dropout(self.ilora_b(h_x))
                              ) * mask_gt_b) +
               self.norm_b2b((h_b +
                              self.dropout(self.proj_b(h_b))
                              ) * mask_gt_b))

        h = h_a * mask_gt_a + h_b * mask_gt_b

        return h

    def cal_rec_loss(self,
                     h: torch.Tensor,
                     gt: torch.Tensor,
                     gt_neg: torch.Tensor,
                     mask_gt_a: torch.Tensor,
                     mask_gt_b: torch.Tensor,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """ InfoNCE """
        e_gt = self.ei(gt)
        e_neg = self.ei(gt_neg)

        logits = torch.cat(((h * e_gt).unsqueeze(-2).sum(-1),
                            (h.unsqueeze(-2) * e_neg).sum(-1)), dim=-1).div(self.temp)

        loss = -F.log_softmax(logits, dim=2)[:, :, 0]
        loss_a = (loss * cal_norm_mask(mask_gt_a)).sum(-1).mean()
        loss_b = (loss * cal_norm_mask(mask_gt_b)).sum(-1).mean()
        return loss_a, loss_b

    @staticmethod
    def cal_domain_rank(h: torch.Tensor,
                        e_gt: torch.Tensor,
                        e_mtc: torch.Tensor,
                        mask_gt_a: torch.Tensor,
                        mask_gt_b: torch.Tensor,
                        ) -> tuple[list[float], list[float]]:
        """ calculate domain rank via inner-product similarity """
        logit_gt = (h * e_gt.squeeze(1)).sum(dim=-1, keepdims=True)
        logit_mtc = (h.unsqueeze(1) * e_mtc).sum(-1)

        ranks = (logit_mtc - logit_gt).gt(0).sum(-1).add(1)
        ranks_a = ranks[mask_gt_a == 1].tolist()
        ranks_b = ranks[mask_gt_b == 1].tolist()

        return ranks_a, ranks_b

    def cal_rank(self,
                 h_f: torch.Tensor,
                 gt: torch.Tensor,
                 gt_mtc: torch.Tensor,
                 mask_gt_a: torch.Tensor,
                 mask_gt_b: torch.Tensor,
                 ) -> tuple[list[float], list[float]]:
        """ rank via inner-product similarity """
        mask_gt_a = mask_gt_a.squeeze(-1)
        mask_gt_b = mask_gt_b.squeeze(-1)

        e_gt, e_mtc = self.ei(gt),  self.ei(gt_mtc)

        ranks_f2a, ranks_f2b = self.cal_domain_rank(h_f, e_gt, e_mtc, mask_gt_a, mask_gt_b)

        return ranks_f2a, ranks_f2b
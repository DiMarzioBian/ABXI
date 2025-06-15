from xml.dom import WrongDocumentErr

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import MultiHeadAttention, FeedForward
from models.layers import LoRA

from utils.metrics import cal_norm_mask
from utils.misc import init_weights


class ABXI(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bs = args.bs
        self.len_trim = args.len_trim
        self.n_item = args.n_item
        self.n_item_a = args.n_item_a
        self.n_item_b = args.n_item_b
        self.n_neg = args.n_neg
        self.temp = args.temp
        self.dropout = args.dropout

        self.d_embed = args.d_embed
        self.rd = args.rd
        self.ri = args.ri


        # item and positional embedding
        self.ei = nn.Embedding(self.n_item + 1, self.d_embed, padding_idx=0)
        self.ep = nn.Embedding(self.len_trim + 1, args.d_embed, padding_idx=0)

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

    def forward(self, seq_x, seq_a, seq_b, pos_x, pos_a, pos_b, mask_gt_a, mask_gt_b):
        # masking
        mask_x = torch.where(pos_x != 0, 1, 0).unsqueeze(-1)
        mask_a = torch.where(pos_a != 0, 1, 0).unsqueeze(-1)
        mask_b = torch.where(pos_b != 0, 1, 0).unsqueeze(-1)

        # embedding
        h_x = (self.ei(seq_x) + self.ep(pos_x)) * mask_x
        h_a = (self.ei(seq_a) + self.ep(pos_a)) * mask_a
        h_b = (self.ei(seq_b) + self.ep(pos_b)) * mask_b

        h_x = F.dropout(h_x, p=self.dropout, training=self.training)
        h_a = F.dropout(h_a, p=self.dropout, training=self.training)
        h_b = F.dropout(h_b, p=self.dropout, training=self.training)

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
                             F.dropout(self.ffn(h_x), p=self.dropout, training=self.training) +
                             F.dropout(self.dlora_x(h_x), p=self.dropout, training=self.training)
                             ) * mask_x

        h_a = self.norm_sa_a(h_a +
                             F.dropout(self.ffn(h_a), p=self.dropout, training=self.training) +
                             F.dropout(self.dlora_a(h_a), p=self.dropout, training=self.training)
                             ) * mask_a

        h_b = self.norm_sa_b(h_b +
                             F.dropout(self.ffn(h_b), p=self.dropout, training=self.training) +
                             F.dropout(self.dlora_b(h_b), p=self.dropout, training=self.training)
                             ) * mask_b

        # proj, ilora
        h_i = self.proj_i(h_x)

        h_a = (self.norm_i2a((h_x +
                              F.dropout(h_i, p=self.dropout, training=self.training) +
                              F.dropout(self.ilora_a(h_x), p=self.dropout, training=self.training)
                              ) * mask_gt_a) +
               self.norm_a2a((h_a +
                              F.dropout(self.proj_a(h_a), p=self.dropout, training=self.training)
                              ) * mask_gt_a))

        h_b = (self.norm_i2b((h_x +
                              F.dropout(h_i, p=self.dropout, training=self.training) +
                              F.dropout(self.ilora_b(h_x), p=self.dropout, training=self.training)
                              ) * mask_gt_b) +
               self.norm_b2b((h_b +
                              F.dropout(self.proj_b(h_b), p=self.dropout, training=self.training)
                              ) * mask_gt_b))

        h = h_a * mask_gt_a + h_b * mask_gt_b

        return h

    def cal_rec_loss(self, h, gt, gt_neg, mask_gt_a, mask_gt_b):
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
    def cal_domain_rank(h, e_gt, e_mtc, mask_gt_a, mask_gt_b):
        """ calculate domain rank via inner-product similarity """
        logit_gt = (h * e_gt.squeeze(1)).sum(-1, keepdims=True)
        logit_mtc = (h.unsqueeze(1) * e_mtc).sum(-1)

        ranks = (logit_mtc - logit_gt).gt(0).sum(-1).add(1)
        ranks_a = ranks[mask_gt_a == 1].tolist()
        ranks_b = ranks[mask_gt_b == 1].tolist()

        return ranks_a, ranks_b

    def cal_rank(self, h_f, gt, gt_mtc, mask_gt_a, mask_gt_b):
        """ rank via inner-product similarity """
        mask_gt_a = mask_gt_a.squeeze(-1)
        mask_gt_b = mask_gt_b.squeeze(-1)

        e_gt, e_mtc = self.ei(gt),  self.ei(gt_mtc)

        ranks_f2a, ranks_f2b = self.cal_domain_rank(h_f, e_gt, e_mtc, mask_gt_a, mask_gt_b)

        return ranks_f2a, ranks_f2b
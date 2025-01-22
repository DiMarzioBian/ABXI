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

        self.d_latent = args.d_latent
        self.rd = args.rd
        self.ri = args.ri

        self.v = args.v
        # version control, v =
        ## -4: ABXI-dp, dLoRA -> projector'
        ## -3: ABXI-i3, iLoRA -> projector'
        ## -2: ABXI-i2, -proj_i, iLoRA -> projector'
        ## -1: ABXI-d, shared encoder + dLoRA -> 3 * encoder'
        ## 0: ABXI'
        ## 1: V1, -dLoRA'
        ## 2: V2, -proj'
        ## 3: V3, -iLoRA'
        ## 4: V4, -dLoRA, -proj, -iLoRA'
        ## 5 : V5, use timestamp-guided alignment')

        # item and positional embedding
        self.ei = nn.Embedding(self.n_item + 1, self.d_latent, padding_idx=0)
        self.ep = nn.Embedding(self.len_trim + 1, args.d_latent, padding_idx=0)

        # encoder, dlora
        self.mha = MultiHeadAttention(args)
        self.ffn = FeedForward(self.d_latent)
        self.dropout = nn.Dropout(args.dropout)

        if self.v == -1:
            self.mha_a = MultiHeadAttention(args)
            self.mha_b = MultiHeadAttention(args)

            self.ffn_a = FeedForward(self.d_latent)
            self.ffn_b = FeedForward(self.d_latent)

        elif self.v not in (1, 4):
            if self.v != -4:
                self.dlora_x = LoRA(self.d_latent, self.rd)
                self.dlora_a = LoRA(self.d_latent, self.rd)
                self.dlora_b = LoRA(self.d_latent, self.rd)
            else:
                self.dlora_x = FeedForward(self.d_latent)
                self.dlora_a = FeedForward(self.d_latent)
                self.dlora_b = FeedForward(self.d_latent)

        self.norm_sa_x = nn.LayerNorm(self.d_latent)
        self.norm_sa_a = nn.LayerNorm(self.d_latent)
        self.norm_sa_b = nn.LayerNorm(self.d_latent)

        # ilora
        if self.v in (-2, -3):
            self.proj_x2a = FeedForward(self.d_latent)
            self.proj_x2b = FeedForward(self.d_latent)

            if self.v == -3:
                self.proj_x2i = FeedForward(self.d_latent)

        elif self.v not in (3, 4):
            self.ilora_a = LoRA(self.d_latent, self.ri)
            self.ilora_b = LoRA(self.d_latent, self.ri)

        # proj
        if self.v not in (2, 4):
            self.proj_i = FeedForward(self.d_latent)
            self.proj_a = FeedForward(self.d_latent)
            self.proj_b = FeedForward(self.d_latent)

        self.norm_i2a = nn.LayerNorm(self.d_latent)
        self.norm_i2b = nn.LayerNorm(self.d_latent)
        self.norm_a2a = nn.LayerNorm(self.d_latent)
        self.norm_b2b = nn.LayerNorm(self.d_latent)

        self.apply(init_weights)

    def forward(self, seq_x, seq_a, seq_b, pos_x, pos_a, pos_b, mask_x, mask_a, mask_b, mask_gt_a, mask_gt_b):
        # embedding
        e_x = self.dropout((self.ei(seq_x) + self.ep(pos_x)) * mask_x)
        e_a = self.dropout((self.ei(seq_a) + self.ep(pos_a)) * mask_a)
        e_b = self.dropout((self.ei(seq_b) + self.ep(pos_b)) * mask_b)

        # mha
        h_sa_x = self.mha(e_x, mask_x)
        if self.v == -1:
            h_sa_a = self.mha_a(e_a, mask_a)
            h_sa_b = self.mha_b(e_b, mask_b)

        else:
            h_sa_a = self.mha(e_a, mask_a)
            h_sa_b = self.mha(e_b, mask_b)

        # switch training / evaluating
        if self.training:
            mask_gt_a = mask_gt_a.unsqueeze(-1)
            mask_gt_b = mask_gt_b.unsqueeze(-1)
        else:
            mask_x = mask_a = mask_b = 1
            h_sa_x = h_sa_x[:, -1]
            h_sa_a = h_sa_a[:, -1]
            h_sa_b = h_sa_b[:, -1]

        # ffn
        if self.v in (1, 4):
            h_sa_x = self.norm_sa_x(h_sa_x + self.dropout(self.ffn(h_sa_x))) * mask_x
            h_sa_a = self.norm_sa_a(h_sa_a + self.dropout(self.ffn(h_sa_a))) * mask_a
            h_sa_b = self.norm_sa_b(h_sa_b + self.dropout(self.ffn(h_sa_b))) * mask_b

        elif self.v == -1:
            h_sa_x = self.norm_sa_x(h_sa_x + self.dropout(self.ffn(h_sa_x))) * mask_x
            h_sa_a = self.norm_sa_a(h_sa_a + self.dropout(self.ffn_a(h_sa_a))) * mask_a
            h_sa_b = self.norm_sa_b(h_sa_b + self.dropout(self.ffn_b(h_sa_b))) * mask_b

        else:
            h_sa_x = self.norm_sa_x(h_sa_x + self.dropout(self.ffn(h_sa_x)) + self.dropout(self.dlora_x(h_sa_x))) * mask_x
            h_sa_a = self.norm_sa_a(h_sa_a + self.dropout(self.ffn(h_sa_a)) + self.dropout(self.dlora_a(h_sa_a))) * mask_a
            h_sa_b = self.norm_sa_b(h_sa_b + self.dropout(self.ffn(h_sa_b)) + self.dropout(self.dlora_b(h_sa_b))) * mask_b

        # proj, ilora
        if self.v == 4:
            h_a = self.norm_a2a(h_sa_x + h_sa_a)
            h_b = self.norm_b2b(h_sa_x + h_sa_b)

        elif self.v in (-4, -1, 0, 1, 5):
            p_i = self.proj_i(h_sa_x)

            h_i2a = self.norm_i2a((h_sa_x +
                                   self.dropout(p_i) +
                                   self.dropout(self.ilora_a(h_sa_x))) * mask_gt_a)

            h_a2a = self.norm_a2a((h_sa_a +
                                   self.dropout(self.proj_a(h_sa_a))) * mask_gt_a)

            h_i2b = self.norm_i2b((h_sa_x +
                                   self.dropout(p_i) +
                                   self.dropout(self.ilora_b(h_sa_x))) * mask_gt_b)

            h_b2b = self.norm_b2b((h_sa_b +
                                   self.dropout(self.proj_b(h_sa_b))) * mask_gt_b)

            h_a = h_i2a + h_a2a
            h_b = h_i2b + h_b2b

        elif self.v == 2:
            h_i2a = self.norm_i2a((h_sa_x +
                                   self.dropout(self.ilora_a(h_sa_x))) * mask_gt_a)

            h_i2b = self.norm_i2b((h_sa_x +
                                   self.dropout(self.ilora_b(h_sa_x))) * mask_gt_b)

            h_a = h_i2a + h_sa_a
            h_b = h_i2b + h_sa_b

        elif self.v == 3:
            p_i = self.proj_i(h_sa_x)

            h_i2a = self.norm_i2a((h_sa_x +
                                   self.dropout(p_i)) * mask_gt_a)

            h_a2a = self.norm_a2a((h_sa_a +
                                   self.dropout(self.proj_a(h_sa_a))) * mask_gt_a)

            h_i2b = self.norm_i2b((h_sa_x +
                                   self.dropout(p_i)) * mask_gt_b)

            h_b2b = self.norm_b2b((h_sa_b +
                                   self.dropout(self.proj_b(h_sa_b))) * mask_gt_b)

            h_a = h_i2a + h_a2a
            h_b = h_i2b + h_b2b

        elif self.v == -2:
            h_i2a = self.norm_i2a((h_sa_x +
                                   self.dropout(self.proj_x2a(h_sa_x))) * mask_gt_a)

            h_a2a = self.norm_a2a((h_sa_a +
                                   self.dropout(self.proj_a(h_sa_a))) * mask_gt_a)

            h_i2b = self.norm_i2b((h_sa_x +
                                   self.dropout(self.proj_x2b(h_sa_x))) * mask_gt_b)

            h_b2b = self.norm_b2b((h_sa_b +
                                   self.dropout(self.proj_b(h_sa_b))) * mask_gt_b)

            h_a = h_i2a + h_a2a
            h_b = h_i2b + h_b2b

        elif self.v == -3:
            p_i = self.proj_x2i(h_sa_x)

            h_i2a = self.norm_i2a((h_sa_x +
                                   self.dropout(p_i) +
                                   self.dropout(self.proj_x2a(h_sa_x))) * mask_gt_a)

            h_a2a = self.norm_a2a((h_sa_a +
                                   self.dropout(self.proj_a(h_sa_a))) * mask_gt_a)

            h_i2b = self.norm_i2b((h_sa_x +
                                   self.dropout(p_i) +
                                   self.dropout(self.proj_x2b(h_sa_x))) * mask_gt_b)

            h_b2b = self.norm_b2b((h_sa_b +
                                   self.dropout(self.proj_b(h_sa_b))) * mask_gt_b)

            h_a = h_i2a + h_a2a
            h_b = h_i2b + h_b2b

        else:
            raise NotImplemented(f'Wrong v = {self.v}.')

        h = h_a * mask_gt_a + h_b * mask_gt_b
        return h, h_sa_x, h_sa_a, h_sa_b

    def cal_rec_loss(self, h, gt, gt_neg, mask_gt_a, mask_gt_b, emb_grad=True):
        """ InfoNCE """
        e_gt = self.ei(gt)
        e_neg = self.ei(gt_neg)
        if not emb_grad:
            e_gt = e_gt.detach()
            e_neg = e_neg.detach()

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

    def cal_rank(self, h_f, h_c, h_a, h_b, gt, gt_mtc, mask_gt_a, mask_gt_b):
        """ rank via inner-product similarity """
        mask_gt_a = mask_gt_a.squeeze(-1)
        mask_gt_b = mask_gt_b.squeeze(-1)

        e_gt, e_mtc = self.ei(gt),  self.ei(gt_mtc)

        ranks_f2a, ranks_f2b = self.cal_domain_rank(h_f, e_gt, e_mtc, mask_gt_a, mask_gt_b)
        ranks_c2a, ranks_c2b = self.cal_domain_rank(h_c, e_gt, e_mtc, mask_gt_a, mask_gt_b)
        ranks_a2a, ranks_a2b = self.cal_domain_rank(h_a, e_gt, e_mtc, mask_gt_a, mask_gt_b)
        ranks_b2a, ranks_b2b = self.cal_domain_rank(h_b, e_gt, e_mtc, mask_gt_a, mask_gt_b)

        return ranks_f2a, ranks_f2b, ranks_c2a, ranks_c2b, ranks_a2a, ranks_a2b, ranks_b2a, ranks_b2b

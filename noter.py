import argparse
from typing import List

import os
from os.path import join
import time

import torch.nn as nn


class Noter(object):
    """ console printing and saving into files """
    def __init__(
            self,
            args: argparse,
    ) -> None:
        self.args = args

        self.t_start = time.time()
        self.f_log = join(args.path_log, f'{args.data}-{time.strftime("%m-%d-%H-%M", time.localtime())}-'
                                         f'{str(args.device)[0] + str(args.device)[-1]}-{args.seed}-abxi.log')

        if os.path.exists(self.f_log):
            os.remove(self.f_log)  # remove the existing file if duplicate

        # welcome
        self.log_msg(f'\n{"-" * 30} Experiment {self.args.name} {"-" * 30}')
        self.log_settings()

    def write(
            self,
            msg: str,
    ) -> None:
        with open(self.f_log, 'a') as out:
            print(msg, file=out)

    def log_msg(
            self,
            msg: str,
    ) -> None:
        print(msg)
        self.write(msg)

    def log_num_param(
            self,
            model: nn.Modulem,
    ) -> None:
        self.log_msg(f'[info] model contains {sum(p.numel() for p in model.parameters() if p.requires_grad)} '
                     f'learnable parameters.\n')

    def log_settings(self):
        msg = (f'[Info] {self.args.name} (data:{self.args.data}, cuda:{self.args.cuda})\n'
               f'| Ver.  {self.args.ver} |\n'
               f'| len_max {self.args.len_max} | d_embed {self.args.d_embed} |'
               f'| n_attn {self.args.n_attn} | n_head {self.args.n_head} | dropout {self.args.dropout} |\n'
               f'| lr {self.args.lr:.2e} | l2 {self.args.l2:.2e} | lr_g {self.args.lr_g:.1f} | lr_p {self.args.lr_p} |\n\n'
               f'| seed {self.args.seed} |\n'
               f'| rd {self.args.rd} | ri {self.args.ri} |\n\n')
        self.log_msg(msg)

    def log_lr(
            self,
            msg: str,
    ) -> None:
        msg = f'           | lr  |     ' + msg
        self.log_msg(msg)

    def log_train(
            self,
            i_epoch: int,
            loss_a: List[float],
            loss_b: List[float],
            t_gap: float,
    ) -> None:
        msg = f'-epoch {i_epoch:>3} | tr  | los | {f"{loss_a:.4f}"[:6]} | {f"{loss_b:.4f}"[:6]} | {t_gap:>5.1f}s |'
        self.log_msg(msg)

    def log_valid(
            self,
            res_a: List,
            res_b: List,
    ) -> None:
        msg = f'           | val |     | {res_a[0]:.4f} | {res_a[1]:.4f} | {res_a[2]:.4f} | {res_a[3]:.4f} | {res_b[0]:.4f} | {res_b[1]:.4f} | {res_b[2]:.4f} | {res_b[3]:.4f} |'
        self.log_msg(msg)

    def log_test(
            self,
            ranks: List,
    ) -> None:
        msg = f'           | te  |  *  | {ranks[0][0]:.4f} | {ranks[0][1]:.4f} | {ranks[0][2]:.4f} | {ranks[0][3]:.4f} | {ranks[1][0]:.4f} | {ranks[1][1]:.4f} | {ranks[1][2]:.4f} | {ranks[1][3]:.4f} |  *  |'
        self.log_msg(msg)

    def log_final(
            self,
            ranks: List,
    ) -> None:
        self.log_msg(f'\n{"-" * 10} Experiment ended {"-" * 10}')
        self.log_settings()
        msg = (f'[ Info ] {self.args.name} ({(time.time() - self.t_start) / 60:.1f} min)\n'
               f'|                A                  |                B                  |\n'
               f'|  hr5   |  hr10  | ndcg10 |  mrr   |  hr5   |  hr10  | ndcg10 |  mrr   |\n'
               f'| {ranks[0][0]:.4f} | {ranks[0][1]:.4f} | {ranks[0][2]:.4f} | {ranks[0][3]:.4f} | {ranks[1][0]:.4f} | {ranks[1][1]:.4f} | {ranks[1][2]:.4f} | {ranks[1][3]:.4f} |\n')
        self.log_msg(msg)

    def log_final_result(
            self,
            ranks: List,
    ) -> None:
        self.log_msg(f'\n{"-" * 10} Experiment ended {"-" * 10}')
        self.log_settings()
        msg = (f'[ Info ] {self.args.name} ({(time.time() - self.t_start) / 60:.1f} min)\n'
               f'|                A                  |                B                  |\n'
               f'|  hr5   |  hr10  | ndcg10 |  mrr   |  hr5   |  hr10  | ndcg10 |  mrr   |\n'
               f'| {ranks[0][0]:.4f} | {ranks[0][1]:.4f} | {ranks[0][2]:.4f} | {ranks[0][3]:.4f} | {ranks[1][0]:.4f} | {ranks[1][1]:.4f} | {ranks[1][2]:.4f} | {ranks[1][3]:.4f} |\n')
        self.log_msg(msg)

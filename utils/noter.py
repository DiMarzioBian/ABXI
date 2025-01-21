import os
from os.path import join
import time


class Noter(object):
    """ console printing and saving into files """
    def __init__(self, args):
        self.args = args

        self.t_start = time.time()
        self.f_log = join(args.path_log, f'{args.data}-{time.strftime("%m-%d-%H-%M", time.localtime())}-'
                                         f'{str(args.device)[0] + str(args.device)[-1]}-{args.seed}--x{args.x}-{args.rd}-{args.ri}-abxi.log')

        if os.path.exists(self.f_log):
            os.remove(self.f_log)  # remove the existing file if duplicate

        # welcome
        self.log_msg(f'\n{"-" * 30} Experiment {self.args.name} {"-" * 30}')
        self.log_settings()

    def write(self, msg):
        with open(self.f_log, 'a') as out:
            print(msg, file=out)

    def log_msg(self, msg):
        print(msg)
        self.write(msg)

    def log_num_param(self, model):
        self.log_msg(f'[info] model contains {sum(p.numel() for p in model.parameters() if p.requires_grad)} '
                     f'learnable parameters.\n')

    def log_settings(self):
        msg = (f'[Info] {self.args.name} (data:{self.args.data}, cuda:{self.args.cuda})\n'
               f'| Ver.  {self.args.ver} |\n'
               f'| len_max {self.args.len_max} | n_attn {self.args.n_attn} | n_head {self.args.n_head} | dropout {self.args.dropout} |\n'
               f'| lr {self.args.lr:.2e} | l2 {self.args.l2:.2e} | lr_g {self.args.lr_g:.1f} | lr_p {self.args.lr_p} |\n\n'
               f'| seed {self.args.seed} | x {self.args.x} |\n'
               f'| rd {self.args.rd} | ri {self.args.ri} |\n\n')
        self.log_msg(msg)

    def log_train(self, loss_a, loss_b, t_gap):
        msg = f'| tr  | los | {f"{loss_a:.4f}"[:6]} | {f"{loss_b:.4f}"[:6]} | {t_gap:>5.1f}s |'
        self.log_msg(msg)

    def log_valid(self, res_f2a, res_f2b):
        msg = f'| val |  f  | {res_f2a[0]:.4f} | {res_f2a[1]:.4f} | {res_f2a[2]:.4f} | {res_f2a[3]:.4f} | {res_f2b[0]:.4f} | {res_f2b[1]:.4f} | {res_f2b[2]:.4f} | {res_f2b[3]:.4f} |'
        self.log_msg(msg)

    def log_test(self, res_f2a, res_f2b, res_c2a, res_c2b, res_a2a, res_a2b, res_b2a, res_b2b):
        msg = (f'| te  |  f  | {res_f2a[0]:.4f} | {res_f2a[1]:.4f} | {res_f2a[2]:.4f} | {res_f2a[3]:.4f} | {res_f2b[0]:.4f} | {res_f2b[1]:.4f} | {res_f2b[2]:.4f} | {res_f2b[3]:.4f} |\n'
               f'      |  x  | {res_c2a[0]:.4f} | {res_c2a[1]:.4f} | {res_c2a[2]:.4f} | {res_c2a[3]:.4f} | {res_c2b[0]:.4f} | {res_c2b[1]:.4f} | {res_c2b[2]:.4f} | {res_c2b[3]:.4f} |\n'
               f'      |  a  | {res_a2a[0]:.4f} | {res_a2a[1]:.4f} | {res_a2a[2]:.4f} | {res_a2a[3]:.4f} | {res_a2b[0]:.4f} | {res_a2b[1]:.4f} | {res_a2b[2]:.4f} | {res_a2b[3]:.4f} |\n'
               f'      |  b  | {res_b2a[0]:.4f} | {res_b2a[1]:.4f} | {res_b2a[2]:.4f} | {res_b2a[3]:.4f} | {res_b2b[0]:.4f} | {res_b2b[1]:.4f} | {res_b2b[2]:.4f} | {res_b2b[3]:.4f} |')
        self.log_msg(msg)

    def log_final(self, res_f2a, res_f2b, res_c2a, res_c2b, res_a2a, res_a2b, res_b2a, res_b2b):
        self.log_msg(f'\n{"-" * 10} Experiment ended {"-" * 10}')
        self.log_settings()
        msg = (f'[ Info ] {self.args.name} ({(time.time() - self.t_start) / 60:.1f} min)\n'
               f'      |                A                  |                B                  |\n'
               f'      |  hr5   |  hr10  | ndcg10 |  mrr   |  hr5   |  hr10  | ndcg10 |  mrr   |\n'
               f'|  f  | {res_f2a[0]:.4f} | {res_f2a[1]:.4f} | {res_f2a[2]:.4f} | {res_f2a[3]:.4f} | {res_f2b[0]:.4f} | {res_f2b[1]:.4f} | {res_f2b[2]:.4f} | {res_f2b[3]:.4f} |\n'
               f'|  x  | {res_c2a[0]:.4f} | {res_c2a[1]:.4f} | {res_c2a[2]:.4f} | {res_c2a[3]:.4f} | {res_c2b[0]:.4f} | {res_c2b[1]:.4f} | {res_c2b[2]:.4f} | {res_c2b[3]:.4f} |\n'
               f'|  a  | {res_a2a[0]:.4f} | {res_a2a[1]:.4f} | {res_a2a[2]:.4f} | {res_a2a[3]:.4f} | {res_a2b[0]:.4f} | {res_a2b[1]:.4f} | {res_a2b[2]:.4f} | {res_a2b[3]:.4f} |\n'
               f'|  b  | {res_b2a[0]:.4f} | {res_b2a[1]:.4f} | {res_b2a[2]:.4f} | {res_b2a[3]:.4f} | {res_b2b[0]:.4f} | {res_b2b[1]:.4f} | {res_b2b[2]:.4f} | {res_b2b[3]:.4f} |\n')
        self.log_msg(msg)

import time
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

from dataloader import get_dataloader
from models.ABXI import ABXI
from utils.metrics import cal_metrics


class Trainer(object):
    def __init__(self, args, noter):
        print('[info] Loading data')

        self.dl = get_dataloader(args)
        self.n_user, self.n_item, self.n_item_a, self.n_item_b = self.dl.dataset.get_stat()
        print('Done.\n')

        self.noter = noter
        self.device = args.device

        # model
        self.model = ABXI(args).to(args.device)

        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler_warmup = LinearLR(self.optimizer, start_factor=1e-5, end_factor=1., total_iters=args.n_warmup)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=args.lr_g, patience=args.lr_p)

        noter.log_num_param(self.model)

    def run_epoch(self, i_epoch):
        self.dl.dataset.set_train()
        self.model.train()
        self.optimizer.zero_grad()
        loss_a, loss_b = 0., 0.
        t0 = time.time()

        # training
        self.noter.log_msg(f'\n[epoch {i_epoch:>2}]')
        for batch in tqdm(self.dl, desc='training', leave=False):
            loss_a_batch, loss_b_batch = self.train_batch(batch)

            n_seq = batch[0].size(0)
            loss_a += (loss_a_batch * n_seq) / self.n_user
            loss_b += (loss_b_batch * n_seq) / self.n_user

        self.noter.log_train(i_epoch, loss_a, loss_b, time.time() - t0)

        # validating
        self.model.eval()
        self.dl.dataset.set_valid()
        ranks_f2a, ranks_f2b, ranks_c2a, ranks_c2b, ranks_a2a, ranks_a2b, ranks_b2a, ranks_b2b \
            = [], [], [], [], [], [], [], []

        with torch.no_grad():
            for batch in tqdm(self.dl, desc='validating', leave=False):
                ranks_batch = self.evaluate_batch(batch)

                ranks_f2a += ranks_batch[0]
                ranks_f2b += ranks_batch[1]

        return cal_metrics(ranks_f2a), cal_metrics(ranks_f2b)

    def run_test(self):
        self.model.eval()
        self.dl.dataset.set_test()
        res_ranks = [[] for _ in range(8)]

        with torch.no_grad():
            for batch in tqdm(self.dl, desc='testing', leave=False):
                ranks_batch = self.evaluate_batch(batch)

            res_ranks = [res_set + res for res_set, res in zip(res_ranks, ranks_batch)]

        return *(cal_metrics(ranks) for ranks in res_ranks),

    def train_batch(self, batch):
        seq_x, seq_a, seq_b, pos_x, pos_a, pos_b, gt, gt_neg = map(lambda x: x.to(self.device), batch)

        mask_gt_a = torch.where(gt <= self.n_item_a, 1, 0)
        mask_gt_b = torch.where(gt > self.n_item_a, 1, 0)

        h = self.model(seq_x, seq_a, seq_b, pos_x, pos_a, pos_b, mask_gt_a, mask_gt_b)

        loss_a, loss_b = self.model.cal_rec_loss(h, gt, gt_neg, mask_gt_a, mask_gt_b)
        (loss_a + loss_b).backward()

        self.optimizer.step()
        return loss_a.item(), loss_b.item()

    def evaluate_batch(self, batch):
        seq_x, seq_a, seq_b, pos_x, pos_a, pos_b, gt, gt_mtc = map(lambda x: x.to(self.device), batch)

        mask_gt_a = torch.where(gt <= self.n_item_a, 1, 0)
        mask_gt_b = torch.where(gt > self.n_item_a, 1, 0)

        h_f = self.model(seq_x, seq_a, seq_b, pos_x, pos_a, pos_b, mask_gt_a, mask_gt_b)

        return self.model.cal_rank(h_f, gt, gt_mtc, mask_gt_a, mask_gt_b)

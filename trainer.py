from argparse import Namespace
import time
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

from models.data.dataloader import get_dataloader
from models.ABXI import ABXI
from models.data.evaluation import cal_metrics

from noter import Noter


class Trainer(object):
    def __init__(self,
                 args: Namespace,
                 noter: Noter,
                 ) -> None:
        print('[info] Loading data')
        self.n_warmup = args.n_warmup
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.n_user = args.n_user
        self.n_item_a = args.n_item_a
        print('Done.\n')

        self.noter = noter
        self.device = args.device

        # model
        self.model = ABXI(args).to(args.device)

        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler_warmup = LinearLR(self.optimizer, start_factor=1e-5, end_factor=1., total_iters=args.n_warmup)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=args.lr_g, patience=args.lr_p)

        noter.log_num_param(self.model)

    def run_epoch(self,
                  i_epoch: int,
                  ) -> tuple[list | None, list | None]:
        self.model.train()
        loss_a, loss_b = 0., 0.
        t_0 = time.time()

        # training
        for batch in tqdm(self.train_loader, desc='training', leave=False):
            self.optimizer.zero_grad()

            loss_a_batch, loss_b_batch = self.train_batch(batch)

            n_seq = batch[0].size(0)
            loss_a += (loss_a_batch * n_seq) / self.n_user
            loss_b += (loss_b_batch * n_seq) / self.n_user

        self.noter.log_train(i_epoch, loss_a, loss_b, time.time() - t_0)

        # warm-up quit
        if i_epoch <= self.n_warmup:
            return None, None

        # validating
        self.model.eval()
        ranks_f2a, ranks_f2b = [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='validating', leave=False):
                ranks_batch = self.evaluate_batch(batch)

                ranks_f2a += ranks_batch[0]
                ranks_f2b += ranks_batch[1]

        return cal_metrics(ranks_f2a), cal_metrics(ranks_f2b)

    def run_test(self,
                 ) -> tuple[list, list]:
        self.model.eval()
        ranks_f2a, ranks_f2b = [], []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='testing', leave=False):
                ranks_batch = self.evaluate_batch(batch)

                ranks_f2a += ranks_batch[0]
                ranks_f2b += ranks_batch[1]

        return cal_metrics(ranks_f2a), cal_metrics(ranks_f2b)

    def train_batch(self,
                    batch: list[torch.Tensor],
                    ) -> tuple[float, float]:
        seq_x, seq_a, seq_b, gt, gt_neg = map(lambda x: x.to(self.device), batch)

        # masking
        mask_x = (seq_x != 0).unsqueeze(-1).half()
        mask_a = (seq_a != 0).unsqueeze(-1).half()
        mask_b = (seq_b != 0).unsqueeze(-1).half()

        mask_gt_a = (gt <= self.n_item_a).half()
        mask_gt_b = (gt > self.n_item_a).half()

        h = self.model(seq_x, seq_a, seq_b, mask_x, mask_a, mask_b, mask_gt_a, mask_gt_b)

        loss_a, loss_b = self.model.cal_rec_loss(h, gt, gt_neg, mask_gt_a, mask_gt_b)
        (loss_a + loss_b).backward()

        self.optimizer.step()
        return loss_a.item(), loss_b.item()

    def evaluate_batch(self,
                       batch: list[torch.Tensor],
                       ) -> tuple[list[float], list[float]]:
        seq_x, gt, gt_mtc = map(lambda x: x.to(self.device), batch)

        mask_gt_a = (gt <= self.n_item_a).half()
        mask_gt_b = (gt > self.n_item_a).half()

        # generate specific-domain sequences and masks
        seq_a = torch.zeros_like(seq_x, dtype=torch.long)
        seq_b = torch.zeros_like(seq_x, dtype=torch.long)

        mask_x = (seq_x != 0).unsqueeze(-1).half()
        mask_a = (seq_x > 0) & (seq_x <= self.n_item_a)
        mask_b = seq_x > self.n_item_a

        seq_a[mask_a] = seq_x[mask_a]
        seq_b[mask_b] = seq_x[mask_b]

        mask_a = mask_a.unsqueeze(-1).half()
        mask_b = mask_b.unsqueeze(-1).half()

        h = self.model(seq_x, seq_a, seq_b, mask_x, mask_a, mask_b, mask_gt_a, mask_gt_b)

        return self.model.cal_rank(h, gt, gt_mtc, mask_gt_a, mask_gt_b)

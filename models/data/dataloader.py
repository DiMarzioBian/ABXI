from typing import Tuple, List
import argparse

from os.path import join
import numpy as np
import pickle
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader


def trim_seq(seq: np.array,
             len_trim: int,
             ) -> np.array:
    """ pad sequences to required length """
    return np.concatenate((np.zeros(max(0, len_trim - len(seq)), dtype=np.int32), seq))[-len_trim:]


def get_spe_seq(n_item_a: int,
                seq: np.array,
                gt: np.array,
                ) -> Tuple[np.array, np.array]:
    """
    Through task-guided alignment.
    """
    A = seq[(0 < seq) & (seq <= n_item_a)]
    B = seq[seq > n_item_a]

    seq_a = np.logical_and(0 < gt, gt <= n_item_a).astype(int)
    seq_b = (gt > n_item_a).astype(int)

    # remove first interaction
    if seq[0] > n_item_a:  # B
        seq_a[seq_a.nonzero()[0][0]] = 0
    else:
        seq_b[seq_b.nonzero()[0][0]] = 0

    # remove last interaction
    if gt[-1] > n_item_a:  # B
        A = A[:-1]
    else:
        B = B[:-1]

    seq_a[seq_a != 0] = A
    seq_b[seq_b != 0] = B

    return seq_a, seq_b


def process_train(n_item_a: int,
                  len_trim: int,
                  seq_raw: List[int],
                  ) -> Tuple[np.array, np.array, np.array, np.array, List]:
    """ process training sequences """
    seq_x, gt = seq_raw[:-1], seq_raw[1:]

    seq_a, seq_b = get_spe_seq(n_item_a, seq_x, gt)

    return *(trim_seq(x, len_trim) for x in (seq_x, seq_a, seq_b, gt)), seq_raw


def process_evaluate(len_trim: int,
                     seq_raw: List[int],
                     ) -> Tuple[np.array, np.array, np.array, List]:
    """ process evaluation sequences """
    seq, gt = seq_raw[:-1], seq_raw[-1:]

    return trim_seq(seq, len_trim), gt, seq_raw


def get_dataset(args: argparse,
                ) -> Tuple[Dataset, Dataset, Dataset]:
    """ get datasets """
    if args.raw:
        print('Reading raw data...')
        with open(join(args.path_data, f'map_item_{args.len_max}.txt'), 'r') as f:
            map_i = json.load(f)
            list_dm = np.array(list(map_i.values()))[:, 1]
            args.n_item_a = np.sum(list_dm == 0)
            args.n_item_b = np.sum(list_dm == 1)

        data_seq = []
        with open(join(args.path_data, args.f_raw), 'r', encoding='utf-8') as f:
            for line in f:
                seq = []
                line = line.strip().split(' ')
                for ui in line[1:][-args.len_max:]:
                    seq.append(int(ui.split('|')[0]))

                data_seq.append(np.array(seq))

        print('Serializing data...')
        data_tr = []
        data_val = []
        data_te = []
        for seq in tqdm(data_seq, desc='processing', leave=False):
            data_tr.append(process_train(args.n_item_a, args.len_trim, seq[:-2]))
            data_val.append(process_evaluate(args.len_trim, seq[:-1]))
            data_te.append(process_evaluate(args.len_trim, seq))

        print('Saving serialized seqs...')
        with open(args.f_data, 'wb') as f:
            pickle.dump((data_tr, data_val, data_te, args.n_item_a, args.n_item_b), f)

    else:
        print('Loading serialized seqs...')
        with open(args.f_data, 'rb') as f:
            (data_tr, data_val, data_te, args.n_item_a, args.n_item_b) = pickle.load(f)

    args.n_item = args.n_item_a + args.n_item_b
    args.n_user = len(data_tr)
    assert len(data_tr) == len(data_val) == len(data_te)

    return (TrainDataset(args, data_tr),
            EvalDataset(args, data_val),
            EvalDataset(args, data_te))


class TrainDataset(Dataset):
    """ training dataset """
    def __init__(self,
                 args: argparse,
                 data: List,
                 ) -> None:
        self.len_trim = args.len_trim
        self.n_neg = args.n_neg
        self.n_neg_x2 = args.n_neg * 2
        self.n_item_a = args.n_item_a

        self.data = data
        self.length = len(self.data)

        self.idx_all_a = np.arange(1, args.n_item_a + 1)
        self.idx_all_b = np.arange(args.n_item_a, args.n_item + 1)

    def get_neg(self,
                gt: np.array,
                cand_a: np.array,
                cand_b: np.array,
                ) -> np.array:
        gt_neg = np.zeros((self.len_trim, self.n_neg_x2))

        for i, x in enumerate(gt):
            if x != 0:
                gt_neg[i] = np.concatenate((np.random.default_rng().choice(cand_a, size=self.n_neg, replace=False),
                                            np.random.default_rng().choice(cand_b, size=self.n_neg, replace=False)))

        return gt_neg

    def __len__(self) -> int:
        return self.length

    def __getitem__(self,
                    index: int,
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_x, seq_a, seq_b, gt, seq_raw = self.data[index]

        cand_a = self.idx_all_a[~np.isin(self.idx_all_a, seq_raw[seq_raw <= self.n_item_a], assume_unique=True)]
        cand_b = self.idx_all_b[~np.isin(self.idx_all_b, seq_raw[seq_raw > self.n_item_a], assume_unique=True)]

        gt_neg = self.get_neg(gt, cand_a, cand_b)

        return tuple(map(lambda x: torch.LongTensor(x), (seq_x, seq_a, seq_b, gt, gt_neg)))


class EvalDataset(Dataset):
    """ evaluation dataset """
    def __init__(self,
                 args: argparse,
                 data: List,
                 ) -> None:
        self.len_trim = args.len_trim
        self.n_item_a = args.n_item_a
        self.n_mtc = args.n_mtc
        self.n_rand = args.n_mtc + args.len_trim

        self.data = data
        self.length = len(self.data)

        self.idx_all_a = np.arange(1, args.n_item_a + 1)
        self.idx_all_b = np.arange(args.n_item_a, args.n_item + 1)

    def get_mtc(self,
                gt: List[int],
                seq_raw: List[int],
                ) -> np.array:
        if gt <= self.n_item_a:
            gt_mtc = np.random.default_rng().choice(
                self.idx_all_a[~np.isin(self.idx_all_a, seq_raw[seq_raw <= self.n_item_a], assume_unique=True)],
                size=self.n_mtc, replace=False)

        else:
            gt_mtc = np.random.default_rng().choice(
                self.idx_all_b[~np.isin(self.idx_all_b, seq_raw[seq_raw > self.n_item_a], assume_unique=True)],
                size=self.n_mtc, replace=False)

        return gt_mtc

    def __len__(self) -> int:
        return self.length

    def __getitem__(self,
                    index: int,
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq, gt, seq_raw = self.data[index]

        gt_mtc = self.get_mtc(gt, seq_raw)

        return tuple(map(lambda x: torch.LongTensor(x), (seq, gt, gt_mtc)))


def get_dataloader(args: argparse,
                   ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return loaders for training, evaluation and testing.
    """
    train_set, valid_set, test_set = get_dataset(args)
    trainloader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.n_worker, pin_memory=True)
    valloader = DataLoader(valid_set, batch_size=args.bse, shuffle=False, num_workers=args.n_worker, pin_memory=True)
    testloader = DataLoader(test_set, batch_size=args.bse, shuffle=False, num_workers=args.n_worker, pin_memory=True)
    return trainloader, valloader, testloader

from os.path import join
import numpy as np
import pickle
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader


class CDSRDataset(Dataset):
    def __init__(self, args):
        self.mode = 'train'

        self.path_data = args.path_data
        self.f_raw = args.f_raw
        self.device = args.device

        self.idx_pad = 0
        self.n_neg = args.n_neg
        self.n_mtc = args.n_mtc
        self.len_max = args.len_max
        self.len_trim = args.len_trim

        # load data
        if args.raw:
            print('Reading raw data...')
            data_u, data_ui = self.read_data()
            self.data_tr, self.data_val, self.data_te = self.serialize_data(data_u, data_ui)

            print('Saving serialized seqs...')
            with open(args.f_data, 'wb') as f:
                pickle.dump((self.data_tr, self.data_val, self.data_te, self.n_user, self.n_item_a, self.n_item_b), f)

        else:
            print('Loading serialized seqs...')
            with open(args.f_data, 'rb') as f:
                (self.data_tr, self.data_val, self.data_te, self.n_user, self.n_item_a, self.n_item_b) = pickle.load(f)

        args.n_user = self.length = self.n_user
        args.n_item_a = self.n_item_a
        args.n_item_b = self.n_item_b
        args.n_item = self.n_item = self.n_item_a + self.n_item_b

        self.idx_all_x = np.arange(1, self.n_item + 1)
        self.idx_all_a = np.arange(1, self.n_item_a + 1)
        self.idx_all_b = np.arange(self.n_item_a + 1, self.n_item + 1)

    def read_data(self):
        """ read preprocessed data """
        with open(join(self.path_data, 'map_user.txt'), 'r') as f:
            map_u = json.load(f)
            self.n_user = len(map_u)

        with open(join(self.path_data, 'map_item.txt'), 'r') as f:
            map_i = json.load(f)
            list_dm = np.array(list(map_i.values()))[:, 1]
            self.n_item_a = np.sum(list_dm == 0)
            self.n_item_b = np.sum(list_dm == 1)

        data_u = []
        data_ui = []
        with open(join(self.path_data, self.f_raw), 'r', encoding='utf-8') as f:
            for line in f:
                seq = []
                line = line.strip().split(' ')
                for ui in line[1:][-self.len_max:]:
                    ui = ui.split('|')
                    seq.append(int(ui[0]))

                data_u.append([int(line[0])])
                data_ui.append(np.array(seq))

        return data_u, data_ui

    def serialize_data(self, data_u, data_ui):
        """ serialize data """
        serialized_tr = []
        serialized_val = []
        serialized_te = []

        # a2b, b2a, n = 0, 0, 0

        for idx_u, seq in tqdm(zip(data_u, data_ui), desc='processing', leave=False):

            # n += len(seq)
            # for i, ii in zip(seq[:-1], seq[1:]):
            #     if i <= self.n_item_a and ii > self.n_item_a:
            #         a2b += 1
            #     elif i > self.n_item_a and ii <= self.n_item_a:
            #         b2a += 1

            serialized_tr.append(self.process_train(idx_u, seq))
            serialized_val.append(self.process_valid(idx_u, seq))
            serialized_te.append(self.process_test(idx_u, seq))

        return serialized_tr, serialized_val, serialized_te

    def trim_seq(self, seq):
        """ pad sequences to required length """
        return np.concatenate((np.zeros(max(0, self.len_trim - len(seq)), dtype=np.int32), seq))[-self.len_trim:]

    @staticmethod
    def get_pos_idx(mask):
        """ get position indices """
        pos = np.flip(np.cumsum(np.flip(mask))) * mask
        return pos

    def get_seq_spe(self, seq, gt, mask_gt_a, mask_gt_b):
        A = seq[(0 < seq) & (seq <= self.n_item_a)]
        B = seq[seq > self.n_item_a]

        seq_a = mask_gt_a.copy()
        seq_b = mask_gt_b.copy()

        # remove first interaction
        if seq[0] > self.n_item_a:  # B
            seq_a[seq_a.nonzero()[0][0]] = 0
        else:
            seq_b[seq_b.nonzero()[0][0]] = 0

        # remove last interaction
        if gt[-1] > self.n_item_a:  # B
            A = A[:-1]
        else:
            B = B[:-1]

        seq_a[seq_a != 0] = A
        seq_b[seq_b != 0] = B

        return seq_a, seq_b

    @staticmethod
    def get_last_idx(seq):
        """ make ground truths for domain-specific sequences """
        i = None
        for i in range(-1, -len(seq) - 1, -1):
            if seq[i] != 0:
                break
        return np.array([i])

    def process_train(self, idx_u, seq_raw):
        idx_u = np.array(idx_u)
        seq_x = seq_raw[:-3]
        gt = seq_raw[1:-2]

        mask_gt_a = np.logical_and(0 < gt, gt <= self.n_item_a).astype(int)
        mask_gt_b = (gt > self.n_item_a).astype(int)

        seq_a, seq_b = self.get_seq_spe(seq_x, gt, mask_gt_a, mask_gt_b)  # need re-order to fit the gt mask

        seq_x, seq_a, seq_b, gt, mask_gt_a, mask_gt_b = \
            (self.trim_seq(s) for s in (seq_x, seq_a, seq_b, gt, mask_gt_a, mask_gt_b))

        mask_x = (seq_x != 0).astype(int)
        mask_a = (seq_a != 0).astype(int)
        mask_b = (seq_b != 0).astype(int)

        pos_x = self.get_pos_idx(mask_x)
        pos_a = self.get_pos_idx(mask_a)
        pos_b = self.get_pos_idx(mask_b)

        mask_x = np.expand_dims(mask_x, -1)
        mask_a = np.expand_dims(mask_a, -1)
        mask_b = np.expand_dims(mask_b, -1)

        return idx_u, seq_x, seq_a, seq_b, pos_x, pos_a, pos_b, mask_x, mask_a, mask_b, gt, mask_gt_a, mask_gt_b, seq_raw[:-2]

    def process_valid(self, idx_u, seq_raw):
        idx_u = np.array(idx_u)
        seq_x = seq_raw[:-2]
        gt = np.expand_dims(seq_raw[-2], 0)

        mask_gt_a = np.logical_and(0 < gt, gt <= self.n_item_a).astype(int)
        mask_gt_b = (gt > self.n_item_a).astype(int)

        seq_a = seq_x[(0 < seq_x) & (seq_x <= self.n_item_a)]  # in-sequence order can ignored
        seq_b = seq_x[seq_x > self.n_item_a]

        seq_x, seq_a, seq_b = (self.trim_seq(s) for s in (seq_x, seq_a, seq_b))

        mask_x = (seq_x != 0).astype(int)
        mask_a = (seq_a != 0).astype(int)
        mask_b = (seq_b != 0).astype(int)

        pos_x = self.get_pos_idx(mask_x)
        pos_a = self.get_pos_idx(mask_a)
        pos_b = self.get_pos_idx(mask_b)

        mask_x = np.expand_dims(mask_x, -1)
        mask_a = np.expand_dims(mask_a, -1)
        mask_b = np.expand_dims(mask_b, -1)

        return idx_u, seq_x, seq_a, seq_b, pos_x, pos_a, pos_b, mask_x, mask_a, mask_b, gt, mask_gt_a, mask_gt_b, seq_raw[:-1]

    def process_test(self, idx_u, seq_raw):
        idx_u = np.array(idx_u)
        seq_x = seq_raw[:-1]
        gt = np.expand_dims(seq_raw[-1], 0)

        mask_gt_a = np.logical_and(0 < gt, gt <= self.n_item_a).astype(int)
        mask_gt_b = (gt > self.n_item_a).astype(int)

        seq_a = seq_x[(0 < seq_x) & (seq_x <= self.n_item_a)]  # in-sequence order can ignored
        seq_b = seq_x[seq_x > self.n_item_a]

        seq_x, seq_a, seq_b = (self.trim_seq(s) for s in (seq_x, seq_a, seq_b))

        mask_x = (seq_x != 0).astype(int)
        mask_a = (seq_a != 0).astype(int)
        mask_b = (seq_b != 0).astype(int)

        pos_x = self.get_pos_idx(mask_x)
        pos_a = self.get_pos_idx(mask_a)
        pos_b = self.get_pos_idx(mask_b)

        mask_x = np.expand_dims(mask_x, -1)
        mask_a = np.expand_dims(mask_a, -1)
        mask_b = np.expand_dims(mask_b, -1)

        return idx_u, seq_x, seq_a, seq_b, pos_x, pos_a, pos_b, mask_x, mask_a, mask_b, gt, mask_gt_a, mask_gt_b, seq_raw

    def get_spe_neg(self, gt, n, seq_raw):
        """ get random negative samples from observed items in specific domain """
        if gt == 0:
            return np.zeros(n, dtype=np.int32)

        elif gt <= self.n_item_a:
            gt_neg = np.random.choice(self.idx_all_a, n + len(seq_raw), replace=False)
            gt_neg = gt_neg[~np.isin(gt_neg, seq_raw[seq_raw <= self.n_item_a])][:n]
            return gt_neg

        else:
            gt_neg = np.random.choice(self.idx_all_b, n + len(seq_raw), replace=False)
            gt_neg = gt_neg[~np.isin(gt_neg,  seq_raw[seq_raw > self.n_item_a])][:n]
            return gt_neg

    def get_crx_neg(self, gt, n, seq_raw):
        """ get random negative samples from observed items in shared domain """
        if gt == 0:
            return np.zeros(n * 2, dtype=np.int32)
        else:
            gt_neg_a = np.random.choice(self.idx_all_a, n + len(seq_raw), replace=False)
            gt_neg_a = gt_neg_a[~np.isin(gt_neg_a,  seq_raw)][:n]

            gt_neg_b = np.random.choice(self.idx_all_b, n + len(seq_raw), replace=False)
            gt_neg_b = gt_neg_b[~np.isin(gt_neg_b,  seq_raw)][:n]
            return np.append(gt_neg_a, gt_neg_b)

    def __getitem__(self, index):
        if self.mode == 'train':
            _, seq_x, seq_a, seq_b, pos_x, pos_a, pos_b, mask_x, mask_a, mask_b, gt, mask_gt_a, mask_gt_b, seq_raw = self.data_tr[index]
            gt_neg = np.array([self.get_crx_neg(idx, self.n_neg, seq_raw[:-2]) for idx in gt])

            out = (seq_x, seq_a, seq_b, pos_x, pos_a, pos_b, mask_x, mask_a, mask_b, gt, gt_neg, mask_gt_a, mask_gt_b)

        else:
            data = self.data_val if self.mode == 'valid' else self.data_te

            _, seq_x, seq_a, seq_b, pos_x, pos_a, pos_b, mask_x, mask_a, mask_b, gt, mask_gt_a, mask_gt_b, seq_raw = data[index]
            gt_mtc = self.get_spe_neg(gt, self.n_mtc, seq_raw)

            out = (seq_x, seq_a, seq_b, pos_x, pos_a, pos_b, mask_x, mask_a, mask_b, gt, gt_mtc, mask_gt_a, mask_gt_b)

        return tuple(map(lambda x: torch.LongTensor(x), out))

    def __len__(self):
        return self.length

    def set_train(self):
        self.mode = 'train'

    def set_valid(self):
        self.mode = 'valid'

    def set_test(self):
        self.mode = 'test'

    def get_stat(self):
        """ return counts of users and items'"""
        return self.n_user, self.n_item_a + self.n_item_b, self.n_item_a, self.n_item_b


def get_dataloader(args):
    return DataLoader(CDSRDataset(args), batch_size=args.bs, shuffle=True, num_workers=args.n_worker,
                      pin_memory=False)
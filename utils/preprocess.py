import os
import argparse
import pandas as pd
from os.path import join
from tqdm import tqdm
import json

from constants import MAPPING_FILE_NAME


def read_amazon(path_a, path_b):
    df_a = pd.read_csv(path_a, sep=',', header=None, names=['user', 'item', 'rating', 'ts']).drop(columns=['rating'])
    df_b = pd.read_csv(path_b, sep=',', header=None, names=['user', 'item', 'rating', 'ts']).drop(columns=['rating'])

    i_dup = set(df_a.item) & set(df_b.item)
    if i_dup:
        df_a = df_a[~df_a.item.isin(i_dup)]
        df_b = df_b[~df_b.item.isin(i_dup)]
        print(f'\t\tFound and deleted {len(i_dup)} duplicated item ID in both domains.')

    return df_a, df_b


def retain_overlap_user(df_a, df_b):
    """ filter out non-overlapped user from both domains """
    print(f'\n[info] Retaining users have interactions on both domains only...')

    u_a, u_b = set(df_a['user'].tolist()), set(df_b['user'].tolist())
    u_ab = u_a.intersection(u_b)

    df_a = df_a[df_a['user'].isin(u_ab)]
    df_b = df_b[df_b['user'].isin(u_ab)]

    df_a.insert(3, 'domain', [0] * df_a.shape[0], True)
    df_b.insert(3, 'domain', [1] * df_b.shape[0], True)

    df = pd.concat([df_a, df_b]).sort_values(['user', 'ts'])

    print(f'\t\tDataset remains {len(df["user"].unique())} users, '
          f'{len(df_a["item"].unique())} A items, {df_a.shape[0]} A interactions, '
          f'{len(df_b["item"].unique())} B items and {df_b.shape[0]} B interactions.')
    return df, df_a, df_b


def filter_cold_item(df, k_i):
    """
    filter out items with less than k_i interactions
    """
    print(f'\n[info] Filtering out cold-start items less than {k_i} interactions ...')

    cnt_i = df['item'].value_counts()
    i_k = cnt_i[cnt_i >= k_i].index

    df = df[df['item'].isin(i_k)]
    df_a = df[df['domain'] == 0]
    df_b = df[df['domain'] == 1]

    print(f'\t\tDataset remains {len(df["user"].unique())} users, '
          f'{len(df_a["item"].unique())} A items, {df_a.shape[0]} A interactions, '
          f'{len(df_b["item"].unique())} B items and {df_b.shape[0]} B interactions.')
    return df, df_a, df_b


def filter_mono_domain_user(df, len_max, k_u):
    """
    trim sequences exceeding the maximum interaction length len_max
    filter out users with less than k_u interactions per domain
    """
    print(f'\n[info] Trim sequence lengths to {len_max} and filter users less than {k_u} interactions per domain...')
    df = df.groupby('user').tail(len_max)
    df_a = df[df['domain'] == 0]
    df_b = df[df['domain'] == 1]

    cnt_u_a = df_a['user'].value_counts()
    cnt_u_b = df_b['user'].value_counts()

    u_k_a = cnt_u_a[cnt_u_a >= k_u].index
    u_k_b = cnt_u_b[cnt_u_b >= k_u].index
    list_u = sorted(set(u_k_a).intersection(set(u_k_b)))

    df = df[df['user'].isin(list_u)]
    df_a = df[df['domain'] == 0]
    df_b = df[df['domain'] == 1]

    print(f'\t\tDataset remains {len(df["user"].unique())} users, '
          f'{len(df_a["item"].unique())} A items, {df_a.shape[0]} A interactions, '
          f'{len(df_b["item"].unique())} B items and {df_b.shape[0]} B interactions.')
    return df, df_a, df_b, list_u


def reindex(df, df_a, df_b, list_u):
    """ filter out users with less than k interactions per domain """
    print(f'\n[info] Reindexing users and items ...')

    map_u, map_i = {}, {}
    for i, x in enumerate(list_u):
        map_u[x] = i

    list_i_a = sorted(set(df_a['item'].tolist()))
    list_i_b = sorted(set(df_b['item'].tolist()))
    for i, x in enumerate(list_i_a):
        map_i[x] = (i + 1, 0)
    i += 2
    for ii, x in enumerate(list_i_b):
        map_i[x] = (ii + i, 1)

    col_u = df['user'].tolist()
    col_i = df['item'].tolist()
    df['user'] = [map_u[u] for u in col_u]
    df['item'] = [map_i[i][0] for i in col_i]

    print(f'\t   Done.')
    return df, map_u, map_i


def save(df, path, f_name, map_u, map_i):
    """ filter out users/items with less than k interactions """
    print(f'\n[info] Saving files to {path} ...')

    with open(join(path, 'map_user.txt'), 'w') as f:
        json.dump(map_u, f)
    with open(join(path, 'map_item.txt'), 'w') as f:
        json.dump(map_i, f)

    with open(join(path, f_name), 'w') as f:
        for u in tqdm(map_u.values(), desc='\t - saving sequences', leave=True):
            line = f'{u}'
            for _, ui in df[df['user'] == u].iterrows():
                line += f' {ui["item"]}|{ui["ts"]}'
            f.write(line + '\n')

    print(f'\t   Done.')


def main():
    parser = argparse.ArgumentParser(description='CDSR Leave-One-Out Preprocess Script')

    # Training
    parser.add_argument('--data', type=str, default='abe', help='name of the dataset')
    parser.add_argument('--k_i', type=int, default=10, help='least interactions for each users/items in both domains')
    parser.add_argument('--k_u', type=int, default=5, help='least interactions for each users/items in each domain')
    parser.add_argument('--len_max', type=int, default=50, help='least interactions for each users/items in each domain')
    args = parser.parse_args()

    (path_a, path_b) = MAPPING_FILE_NAME[args.data]
    path_a = f'../data/raw/{path_a}'
    path_b = f'../data/raw/{path_b}'
    path_processed = f'../data/{args.data}/'
    f_processed = f'{args.data}_{args.len_max}_preprocessed.txt'
    if not os.path.exists(path_processed):
        os.makedirs(path_processed)

    if args.data in MAPPING_FILE_NAME.keys():
        print(f'\n[info] Start preprocessing "{args.data}" dataset...')
        df_a, df_b = read_amazon(path_a, path_b)
    else:
        raise NotImplementedError(f'Selected dataset "{args.data}" is not supported.')

    df, *_ = retain_overlap_user(df_a, df_b)

    df, *_ = filter_cold_item(df, args.k_i)

    df, df_a, df_b, list_u = filter_mono_domain_user(df, args.len_max, args.k_u)

    df, map_u, map_i = reindex(df, df_a, df_b, list_u)

    save(df, path_processed, f_processed, map_u, map_i)


if __name__ == '__main__':
    main()

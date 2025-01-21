#!/bin/bash

# afk
python main.py --cuda 0 --bs 256 --data afk --n_worker 23 --lr 1e-4 --l2 5e0 --x 0 --rd 16 -- ri 64 --seed 3407

# abe
python main.py --cuda 0 --bs 256 --data abe --n_worker 23 --lr 1e-4 --l2 5e0 --x 0 --rd 64 -- ri 64 --seed 3407

# amb
python main.py --cuda 0 --bs 256 --data afk --n_worker 23 --lr 1e-4 --l2 5e0 --x 0 --rd 4 -- ri 4 --seed 3407

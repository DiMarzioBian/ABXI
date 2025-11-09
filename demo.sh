#!/bin/bash

# For reference only.
# As we re-write the model architecture to improve speed, some hyperparameters may need changes as well.
# afk
python main.py --n_worker 28 --bs 256 --data afk --lr 1e-4 --l2 5e0 --rd 16 --ri 64 --cuda 0 --seed 3407

# abe
python main.py --n_worker 28 --bs 256 --data abe --lr 1e-4 --l2 5e0 --rd 64 --ri 64 --cuda 0 --seed 3407

# amb
python main.py --n_worker 28 --bs 256 --data amb --lr 1e-4 --l2 5e0 --rd 4 --ri 4 --cuda 0 --seed 3407

# ABXI-PyTorch

This is the ***official*** Pytorch implementation of paper "ABXI: Invariant Interest Adaptation for Task-Guided Cross-Domain Sequential Recommendation" accepted by WebConf'25 (WWW'25).


## 1. Data
In argument '--data', 'afk' refers to Amazon Food-Kitchen dataset, 'amb' refers to Amazon Movie-Book dataset, and 'abe' refers to Amazon Beauty-Electronics dataset.

Processed data are stored in /data/. If you wanna process your own data, please put the data under /data/raw/, and check the preprocess scripts /utils/preprocess.py.

## 2. Usage
Please check demo.sh on running on different datasets.

### 2.1. Ablation

Please specific `--x` command to switch from different ablation variants. The default is `--x 0` that refers to ABXI. Other variants' indices are listed below.
```
-3: encoder + dLoRA + proj + 2*proj (ABXI-i3)
-2: encoder + dLoRA + 2*proj (ABXI-i2)
-1: 3*encoder + proj + iLoRA (ABXI-d)
 0: encoder + dLoRA + proj + iLoRA (ABXI)
 1: encoder + proj + iLoRA (v1)
 2: encoder + dLoRA + iLoRA (v2)
 3: encoder + dLoRA + proj (v3)
 4: encoder (v4)
 5: encoder + dLoRA + proj + iLoRA + old alignment
```


## 3. File Tree
```
ABXI-Anony/
├── data/
│   ├── abe/
│   │   ├── abe_50_preprocessed.txt
│   │   ├── abe_50_seq.pkl
│   │   ├── abe_50_seq_old.pkl
│   │   ├── map_item.txt
│   │   └── map_user.txt
│   ├── afk/
│   │   ├── afk_50_preprocessed.txt
│   │   ├── afk_50_seq.pkl
│   │   ├── afk_50_seq_old.pkl
│   │   ├── map_item.txt
│   │   └── map_user.txt
│   └── amb/
│       ├── amb_50_preprocessed.txt
│       ├── map_item.txt
│       └── map_user.txt
├── dataloader.py
├── dataloader_old.py
├── demo.sh
├── main.py
├── models/
│   ├── ABXI.py
│   ├── encoders.py
│   └── layers.py
├── README.md
├── requirements.txt
├── trainer.py
└── utils/
    ├── constants.py
    ├── metrics.py
    ├── misc.py
    ├── noter.py
    └── preprocess.py
```

Because the files 'amb_50_seq.pkl' and 'amb_50_seq_old.pkl' exceeds Github file size limit, you have to manually generate this preprocessed serialized data first, by adding argument '--raw' in the first experiment on Movie-Book dataset.
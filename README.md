# ABXI-PyTorch

This is the ***official*** Pytorch implementation of paper "ABXI: Invariant Interest Adaptation for Task-Guided Cross-Domain Sequential Recommendation" accepted by WebConf'25 (WWW'25).

Links: [ACM](https://dl.acm.org/doi/10.1145/3696410.3714819), [Arxiv](https://arxiv.org/abs/2501.15118), [DOI](https://doi.org/10.1145/3696410.3714819)


## 1. Data
In argument '--data', 'afk' refers to Amazon Food-Kitchen dataset, 'amb' refers to Amazon Movie-Book dataset, and 'abe' refers to Amazon Beauty-Electronics dataset.

Processed data are stored in /data/. If you wanna process your own data, please put the data under /data/raw/, and check the preprocess scripts /utils/preprocess.py.

## 2. Usage
Please check demo.sh on running on different datasets.

### 2.1. Ablation

Please specific `--v` command to switch from different ablation variants. The default is `--v 0` that refers to ABXI. Other variants' indices are listed below.

    -4: ABXI-dp3, dLoRA -> projector
    -3: ABXI-i3, iLoRA -> projector
    -2: ABXI-i2, -proj_i, iLoRA -> projector
    -1: ABXI-e3, shared encoder + dLoRA -> 3 * encoder
    0: ABXI
    1: V1, -dLoRA
    2: V2, -proj
    3: V3, -iLoRA
    4: V4, -dLoRA, -proj, -iLoRA
    5: V5, use timestamp-guided alignment


## 3. Citation

If you found the codes are useful, please cite our paper.

    @inproceedings{bian2025abxi,
      title={ABXI: Invariant Interest Adaptation for Task-Guided Cross-Domain Sequential Recommendation},
      author={Bian, Qingtian and de Carvalho, Marcus and Li, Tieying and Xu, Jiaxing and Fang, Hui and Ke, Yiping},
      booktitle={Proceedings of the ACM on Web Conference 2025},
      pages={3183--3192},
      year={2025}
    }


## 4. File Tree

    ABXI/
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


Because the files 'amb_50_seq.pkl' and 'amb_50_seq_old.pkl' exceed Github file size limit, you have to manually generate this preprocessed serialized data first, by adding argument '--raw' in the first experiment on Movie-Book dataset.
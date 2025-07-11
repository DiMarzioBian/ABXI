# ABXI-PyTorch

This is the ***official*** Pytorch implementation of paper "ABXI: Invariant Interest Adaptation for Task-Guided Cross-Domain Sequential Recommendation" accepted by WebConf'25 (WWW'25).

Links: [ACM](https://dl.acm.org/doi/10.1145/3696410.3714819), [Arxiv](https://arxiv.org/abs/2501.15118), [DOI](https://doi.org/10.1145/3696410.3714819)


## 1. Data
In argument '--data', 'afk' refers to Amazon Food-Kitchen dataset, 'amb' refers to Amazon Movie-Book dataset, and 'abe' refers to Amazon Beauty-Electronics dataset.

Processed data are stored in /data/. If you wanna process your own data, please put the data under /data/raw/, and check the preprocess scripts /data/prepare_amazon.py.


## 2. Usage
Please check demo.sh on running on different datasets.


## 3. Citation

If you found the codes are useful, please leave a star on our repo and cite our paper.

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
    │   │   ├── map_item_50.txt
    │   │   └── map_user_50.txt
    │   ├── afk/
    │   │   ├── afk_50_preprocessed.txt
    │   │   ├── afk_50_seq.pkl
    │   │   ├── map_item_50.txt
    │   │   └── map_user_50.txt
    │   ├── amb/
    │   │   ├── amb_50_preprocessed.txt
    │   │   ├── amb_50_seq.pkl
    │   │   ├── map_item_50.txt
    │   │   └── map_user_50.txt
    │   ├── mapper_raw_file.py
    │   └── prepare_amazon.py
    ├── models/
    │   ├── data/
    │   │   ├── datalaoder.py
    │   │   └── evaluation.py
    │   ├── utils/
    │   │   ├── initialization.py
    │   │   └── position.py
    │   ├── ABXI.py/
    │   ├── encoders.py
    │   └── layers.py
    ├── demo.sh
    ├── main.py
    ├── noter.py
    ├── README.md
    ├── requirements.txt
    └── trainer.py

## 5. Update note (v2.1)
Major efficiency boost and restructured project for improved maintainability (compared with v1.0).
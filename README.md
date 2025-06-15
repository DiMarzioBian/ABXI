# ABXI-PyTorch

This is the ***official*** Pytorch implementation of paper "ABXI: Invariant Interest Adaptation for Task-Guided Cross-Domain Sequential Recommendation" accepted by WebConf'25 (WWW'25).

Links: [ACM](https://dl.acm.org/doi/10.1145/3696410.3714819), [Arxiv](https://arxiv.org/abs/2501.15118), [DOI](https://doi.org/10.1145/3696410.3714819)


## 1. Data
In argument '--data', 'afk' refers to Amazon Food-Kitchen dataset, 'amb' refers to Amazon Movie-Book dataset, and 'abe' refers to Amazon Beauty-Electronics dataset.

Processed data are stored in /data/. If you wanna process your own data, please put the data under /data/raw/, and check the preprocess scripts /data/prepare_amazon.py.


## 2. Usage
Please check demo.sh on running on different datasets.


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
        └── prepare_amazon.py
        └── mapper_raw_file.py
    │   ├── abe/
    │   │   ├── abe_50_preprocessed.txt
    │   │   ├── abe_50_seq.pkl
    │   │   ├── map_item.txt
    │   │   └── map_user.txt
    │   ├── afk/
    │   │   ├── afk_50_preprocessed.txt
    │   │   ├── afk_50_seq.pkl
    │   │   ├── map_item.txt
    │   │   └── map_user.txt
    │   └── amb/
    │       ├── amb_50_preprocessed.txt
    │       ├── map_item.txt
    │       └── map_user.txt
    ├── dataloader.py
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
        ├── metrics.py
        ├── misc.py
        ├── noter.py


Because the files 'amb_50_seq.pkl' exceeds Github file size limit, you have to manually generate this preprocessed serialized data first, by adding argument '--raw' in the first experiment on Movie-Book dataset.
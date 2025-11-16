# READ BEFORE
Current version has undergone significant code refactoring for efficiency, training stability, and code readability, resulting in a 2.0x to 3.0x increase in training speed compared to the v1.0 version.

The new results under 5 seeds for each datasets are listed below:

|         Dataset         |   seed   |      A      |             |             |             |      B      |             |             |             |
|:-----------------------:|:--------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|                         |          |    HR@5     |    HR@10    |   NDCG@10   |     MRR     |    HR@5     |    HR@10    |   NDCG@10   |     MRR     |
|    Food-Kitchen (FK)    |   3407   |   0.2538    |   0.3241    |   0.1999    |   0.1694    |   0.1763    |   0.2468    |   0.1441    |   0.1226    |
|                         |    0     |   0.2501    |   0.3220    |   0.1959    |   0.1647    |   0.1771    |   0.2478    |   0.1456    |   0.1242    |
|                         |    1     |   0.2559    |   0.3245    |   0.1953    |   0.1627    |   0.1750    |   0.2421    |   0.1424    |   0.1224    |
|                         |    2     |   0.2489    |   0.3253    |   0.1964    |   0.1643    |   0.1714    |   0.2385    |   0.1399    |   0.1202    |
|                         |    3     |   0.2509    |   0.3258    |   0.1980    |   0.1666    |   0.1725    |   0.2440    |   0.1413    |   0.1201    |
|                         | **mean** | **0.2519**  | **0.3243**  | **0.1971**  | **0.1655**  | **0.1745**  | **0.2438**  | **0.1427**  | **0.1219**  | 
|                         | **std**  | **±0.0029** | **±0.0015** | **±0.0019** | **±0.0026** | **±0.0024** | **±0.0037** | **±0.0023** | **±0.0017** |
| Beauty-Electronics (BE) |   3407   |   0.2656    |   0.3749    |   0.2170    |   0.1773    |   0.1589    |   0.2336    |   0.1340    |   0.1154    |
|                         |    0     |   0.2869    |   0.3797    |   0.2233    |   0.1838    |   0.1543    |   0.2339    |   0.1328    |   0.1141    |
|                         |    1     |   0.2752    |   0.3792    |   0.2205    |   0.1808    |   0.1562    |   0.2320    |   0.1336    |   0.1156    |
|                         |    2     |   0.2747    |   0.3749    |   0.2160    |   0.1767    |   0.1581    |   0.2239    |   0.1326    |   0.1176    |
|                         |    3     |   0.2747    |   0.3803    |   0.2234    |   0.1839    |   0.1597    |   0.2405    |   0.1368    |   0.1173    |
|                         | **mean** | **0.2754**  | **0.3778**  | **0.2200**  | **0.1805**  | **0.1574**  | **0.2328**  | **0.1340**  | **0.1160**  |
|                         | **std**  | **±0.0076** | **±0.0027** | **±0.0035** | **±0.0034** | **±0.0022** | **±0.0059** | **±0.0017** | **±0.0014** |
|     Movie-Book (MB)     |   3407   |   0.2793    |   0.3608    |   0.2337    |   0.2072    |   0.1918    |   0.2493    |   0.1622    |   0.1463    |
|                         |    0     |   0.2780    |   0.3581    |   0.2347    |   0.2096    |   0.1898    |   0.2472    |   0.1603    |   0.1444    |
|                         |    1     |   0.2779    |   0.3593    |   0.2331    |   0.2070    |   0.1918    |   0.2500    |   0.1609    |   0.1442    |
|                         |    2     |   0.2843    |   0.3607    |   0.2354    |   0.2094    |   0.1941    |   0.2502    |   0.1631    |   0.1470    |
|                         |    3     |   0.2845    |   0.3626    |   0.2374    |   0.2114    |   0.1932    |   0.2487    |   0.1624    |   0.1467    |
|                         | **mean** | **0.2808**  | **0.3603**  | **0.2349**  | **0.2089**  | **0.1921**  | **0.2491**  | **0.1618**  | **0.1457**  | 
|                         | **std**  | **±0.0033** | **±0.0017** | **±0.0017** | **±0.0018** | **±0.0016** | **±0.0012** | **±0.0011** | **±0.0013** |

To be noticed, these changes lead to fluctuations in the predictive performance of different metrics, which may be due to subtle numerical variations in data processing or initialization. 
Specifically, compared to the reported results, the metric fluctuation range for the BE dataset is -1.43% to +2.11%, for the FK it is -5.12% to -1.49%, and for the MB it is -3.11% to -1.37%.

**For exact replication of the results reported in the paper, please use this old version: [link](https://github.com/DiMarzioBian/ABXI/tree/effcd56526bac6d7fadbece9b5b675cd03997a64).**



# ABXI-PyTorch

This is the ***official*** Pytorch implementation of paper "ABXI: Invariant Interest Adaptation for Task-Guided Cross-Domain Sequential Recommendation" accepted by WebConf'25 (WWW'25).
**s**
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

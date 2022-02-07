# SurRF: Unsupervised Multi-view Stereopsis by Learning Surface Radiance Field[[Paper]](https://ieeexplore.ieee.org/document/9555381)[[Demo]()]

The official implementation for the paper:

**SurRF: Unsupervised Multi-view Stereopsis by Learning Surface Radiance Field**

Jinzhi Zhang\*, Mengqi Ji*, Guangyu Wang, Xue Zhiwei, Shengjin Wang, Lu Fang.

Accepted by [TPAMI, 2021](https://ieeexplore.ieee.org/document/9555381).

## Installation

The code is compatible with python 3.6 and pytorch 1.5.0. Conda environment and additional dependencies including pytorch3d can be installed by running:

```
conda env create -f environment.yml
conda activate SurRF
```





## Usage



#### Data

We optimize and evaluate SurRF on [DTU MVS dataset](http://roboimagedata.compute.dtu.dk/?page_id=36) and [Tanks and Temples dataset](https://www.tanksandtemples.org/).

#### Optimizing Surface Radiance Field



#### Point Cloud Reconstruction



#### Novel View Synthesis



## Citation

If you find this project useful for your research, please cite:

```
@ARTICLE{9555381,
  author={Zhang, Jinzhi and Ji, Mengqi and Wang, Guangyu and Zhiwei, Xue and Wang, Shengjin and Fang, Lu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={SurRF: Unsupervised Multi-view Stereopsis by Learning Surface Radiance Field}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3116695}}
```




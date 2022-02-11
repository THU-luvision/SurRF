# SurRF: Unsupervised Multi-view Stereopsis by Learning Surface Radiance Field[[Paper]](https://ieeexplore.ieee.org/document/9555381)[[Demo]()]

<img src="https://github.com/GuangyuWang99/SurRF/raw/main/media/method_1.png" width="250" height="250" alt="method"/><img align='right' src="https://github.com/GuangyuWang99/SurRF/raw/main/media/pipeline.png" width="600" height="250" alt="pipeline"/>

The official implementation for the paper:

**SurRF: Unsupervised Multi-view Stereopsis by Learning Surface Radiance Field**

Jinzhi Zhang\*, Mengqi Ji*, Guangyu Wang, Xue Zhiwei, Shengjin Wang, Lu Fang.

Accepted by [TPAMI, 2021](https://ieeexplore.ieee.org/document/9555381).

## Installation

In order to run the code you will need a GPU with CUDA support and Python interpreter. The code is compatible with python 3.6 and pytorch 1.5.0. Conda environment and additional dependencies including pytorch3d can be installed by running:

```
conda env create -f environment.yml
conda activate SurRF
```

## Usage

This official implementation reproduces the optimization, view synthesis and point cloud reconstruction of SurRF. All the parameters to be specified and tuned are listed.

- `agents/base.py`: optimization of surface radiance field, novel view rendering, point cloud reconstruction, etc.
- `configs/parameter.py`: all the directories and parameters to be specified and tuned.
- `dataset/dataset.py`: data preparation code.
- `graphs/render/rasterizer.py`: rasterization of the input 3D triangulation. 
- `graphs/render/neural_shading.py`: neural rendering by disentangling geometry, texture and lighting. 
- `graphs/render/render_base.py`: base code for rasterization and rendering, point cloud reconstruction scripts are also included.
- `graphs/models/network.py`: neural network architecture.

#### Data

We optimize and evaluate SurRF on publicly available MVS datasets, including [DTU MVS dataset](http://roboimagedata.compute.dtu.dk/?page_id=36) and [Tanks and Temples dataset](https://www.tanksandtemples.org/).

When using DTU dataset, it is recommended to download the raw data from the [DTU MVS dataset](http://roboimagedata.compute.dtu.dk/?page_id=36), including the rectified images, camera poses, bounding boxes, etc. According to the definition of Surface Radiance Field representation, a sparse and coarse point cloud is also needed as input, which can be easily obtained by running any existing Structure from Motion (SfM) algorithms. Here, we provide a coarse input point cloud obtained from SfM for DTU scan9 as an example, which can be found in `data/preprocess/ply_rmvs/9/4.0_surface_xyz_flow.ply`. 

This implementation uses DTU scan9 as an example. To run the code, the dataset and recording directory should be firstly specified in `root_params()`  and `load_params()` from `config/parameter.py`, such as `_input_data_rootFld`, `root_file`, `load_checkpoint_dir`, `datasetFolder` etc. It is recommended to arrange all the input data inside the `datasetFolder`.

When using other datasets, the images, camera poses and input point cloud should be organized in the similar way as the provided DTU scan9 example.

#### Optimizing Surface Radiance Field

To optimize the Surface Radiance Field, 

- specify `mode` in `root_params()` of `config/parameter.py` as `'train'`,
- set `img_h` and `img_w` as the original image resolution from the dataset, e.g. `img_h=1200, img_w=1600` for [DTU MVS dataset](http://roboimagedata.compute.dtu.dk/?page_id=36),
- set `render_image_size`, `random_render_image_size`, `compress_ratio_h`, `compress_ratio_w`, `compress_ratio_total` as the rendered output size, the random render resolution range for training, number of image crops along the h-axis, number of image crops along the w-axis, down sample ratio of the original input images (e.g. `render_image_size=400`, `random_render_image_size=[360,400]`, `compress_ratio_h=2`, `compress_ratio_w=2`, `compress_ratio_total=2` means that the input image (e.g. with original resolution 1200x1600) is firstly down-sampled twice (e.g. 600x800), then cropped as 4 segments (e.g. each with size 300x400) for training), 
- specify `trainViewList` in `load_params()` of `config/parameter.py` as a list of training view index, e.g. any integer between 1 to 49 for DTU scan9,
- then run: 

```
cd agents/
python base.py
```

#### Point Cloud Reconstruction

To reconstruct point cloud based on the optimized SurRF, 

- specify `mode` in `root_params()` of `config/parameter.py` as `'reconstruct'`,
- specify `load_checkpoint_dir` in `root_params()` of `config/parameter.py` as the training checkpoint directory,
- adjust `sample_resolution` in `reconstruct_params()` of `config/parameter.py` as the desired point cloud resolution,
- then run: 

```
cd agents/
python base.py
```

#### Novel View Synthesis

To interpolate novel views based on the optimized SurRF, 

- specify the `mode` in `root_params()` of `config/parameter.py` as `'reconstruct'`.
- specify the `load_checkpoint_dir` in `root_params()` of `config/parameter.py` as the training checkpoint directory.
- specify `testViewList` in `load_params()` of `config/parameter.py` as a list of training view index, e.g. any integer between 1 to 49 for DTU scan9,
- then run: 

```
cd agents/
python base.py
```

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




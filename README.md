# Go One Dimension Higher: Can Neural Networks for Point Cloud Analysis Help in Image Recognition? [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/chenzhaiyu/pointset-image-recognition/blob/master/LICENSE)

![](https://i.imgur.com/zDdbQei.png)

## Introduction

Inspired by the sanity check made by Qi et al., which treats an image from MNIST dataset as a 2D point cloud, in this project we discuss whether neural networks designed for point cloud analysis can help image recognition. 

> *While we focus on 3D point cloud learning, a sanity check experiment is to apply our network on a 2D point clouds - pixel sets.*

Three neural networks originally for point cloud analysis are employed, namely PointNet, PointNet++ and PointCNN. We have no ambition to defeat the state-of-the-art CNNs. Instead, the goal of this project is to discover the potential of these networks designed for 3D point set on 2D pixel set. The impact of shape information and the shift variance characteristic are discussed. For more details, please refer to the [blog](https://hackmd.io/@zhaiyuchen/cs4245).

## Requirements

* PyTorch 1.2

* PyTorch Geometric 1.5
  
  Please refer to [environment.yml](https://github.com/chenzhaiyu/pointset-image-recognition/blob/master/environment.yml) for a complete list of required packages.

## Usage

Create the environment from the `environment.yml` file:

```bash
conda env create -f environment.yml
```

### Classification

```bash
# example training
python train_cls.py --batch_size 64 --data_dir 'path/to/data' --log_dir 'path/to/log' --dataset_name 'fashion' --epoch 250 --model 'pointcnn_cls' --num_class 10 --num_point 256
```

### Semantic Segmentation

```bash
# example training
python train_semseg.py --batch_size 64 --data_dir 'path/to/data' --log_dir 'path/to/log' --epoch 250 --model 'pointnet2_sem_seg' --num_class 20 --num_point 4096
```

### Tensorboard

```bash
tensorboard --logdir 'path/to/log'
```


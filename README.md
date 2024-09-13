# Tensor4ML

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
[![GitHub stars](https://img.shields.io/github/stars/xinychen/tensor-learning.svg?logo=github&label=Stars&logoColor=white)](https://github.com/xinychen/tensor-learning)

<h6 align="center">Made by Xinyu Chen • :globe_with_meridians: <a href="https://xinychen.github.io">https://xinychen.github.io</a></h6>

**Tensor Decomposition for Machine Learning (Tensor4ML).** This article summarizes the development of tensor decomposition models and algorithms in the literature, offering comprehensive reviews and tutorials on topics ranging from matrix and tensor computations to tensor decomposition techniques across a wide range of scientific areas and applications. Since the decomposition of tensors is often formulated as an optimization problem, this article also provides a preliminary introduction to some classical methods for solving convex and nonconvex optimization problems. This work aims to offer valuable insights to both the machine learning and data science communities by drawing strong connections with the key concepts related to tensor decomposition. To ensure reproducibility and sustainability, we provide resources such as datasets and Python implementations, primarily utilizing Python’s `numpy` library. The content includes:

<br>

In a hurry? Please check out our **contents** as follows.

- Introduction
  - Tensor decomposition in the past 10-100 years
  - Tensor decomposition in the past decade
- What Are Tensors?
  - Tensors in algebra & machine learning
  - Tensors in data science
- Foundation of Tensor Computations
  - Norms
  - Matrix trace
  - Kronecker product
  - Khatri-Rao product
  - Modal product
  - Outer product
  - Derivatives
- [Foundation of Optimization](https://spatiotemporal-data.github.io/tensor4ml/opt_foundation/)
  - Gradient descent methods
  - Alternating minimization
  - Alternating direction method of multipliers
  - Greedy methods for <img style="display: inline;" src="https://latex.codecogs.com/svg.latex?&space;\ell_0"/>-norm minimization
  - Bayesian optimization
  - Power iteration
  - Procrustes problems

<br>


<h2 align="center">:book: Reproducing Literature in Python</h2>
<p align="right"><a href="#Tensor4ML"><sup>▴ Back to top</sup></a></p>

We reproduce some tensor learning experiments in the previous literature.

| Year | Title | PDF | Authors' Code | Our Code | Status |
|:---|:------:|:---:|:---:|:-----:|----:|
|  2015 | Accelerated Online Low-Rank Tensor Learning for Multivariate Spatio-Temporal Streams | [ICML 2015](http://proceedings.mlr.press/v37/yua15.pdf) | [Matlab code](http://roseyu.com/Materials/accelerate_online_low_rank_tensor.zip) | [Python code](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/baselines/Online-LRTL.ipynb) | Under development |
|  2016 | Scalable and Sound Low-Rank Tensor Learning | [AISTATS 2016](http://proceedings.mlr.press/v51/cheng16.pdf) | - | [xx](xx) | Under development |

<br>


<h2 align="center">Quick Run</h2>
<p align="right"><a href="#Tensor4ML"><sup>▴ Back to top</sup></a></p>

- If you want to run the code, please
  - download (or clone) this repository,
  - open the `.ipynb` file using [Jupyter notebook](https://jupyter.org/install.html),
  - and run the code.

<br>

<h2 align="center">Citing</h2>
<p align="right"><a href="#Tensor4ML"><sup>▴ Back to top</sup></a></p>

This repository is from the following paper, please cite our paper if it helps your research.

<br>

<h2 align="center">License</h2>
<p align="right"><a href="#Tensor4ML"><sup>▴ Back to top</sup></a></p>

This work is released under the MIT license. If you find these codes useful, please star (★) this repository.

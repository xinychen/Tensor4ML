# Tensor Learning (张量学习)

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
[![GitHub stars](https://img.shields.io/github/stars/xinychen/tensor-learning.svg?logo=github&label=Stars&logoColor=white)](https://github.com/xinychen/tensor-learning)


Tutorials and Python codes for tensor factorization, tensor completion and tensor regression techniques with the following real-world applications:

- Image inpainting
- Spatiotemporal data imputation
- Recommender systems
- Multivariate time series imputation and forecasting

In a hurry? Please check out our contents as follows.

Our Work
---

- **Ma**chine Learning for Multivariate **T**ime **S**eries Forecasting (**mats**)

<h5 align="center"><i>Low-Rank Autoregressive Tensor Completion for Multivariate Time Series Forecasting.<br>
  [<a href="https://arxiv.org/abs/2005">arXiv</a>]</i></h5>

<p align="center">
<img align="middle" src="https://github.com/xinychen/transdim/blob/master/images/predictor-explained.png" width="666" />
</p>

Building multivariate time series forecasting tool on the well-understood Low-Rank Tensor Completion (LRTC), we develop a **Low-Rank Autoregressive Tensor Completion** which takes into account:

- autoregressive process on the matrix structure to capture local temporal states,
- and low-rank assumption on the tensor structure to capture global low-rank patterns simultaneously.

Code for reproducing experiments is provided in the **`mats`** folder. Please check out `LATC-imputer.ipynb` and `LATC-predictor.ipynb` for details.


Tutorial
---

- Part 1: Foundations
  - 1 Proximal Methods
    - [1.1 Iterative Shrinkage Thresholding Algorithm (ISTA)](xxxx)
    - [1.2 Singular Value Thresholding](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/content/SVT.ipynb)

  - 2 Bayesian Inference Methods

  - 3 Time Series Analysis
    - [1.1 Vector Autoregressive (VAR) Model](xxxx)

- Part 2: Matrix Factorization and Completion Techniques
  - 1 Low-Rank Matrix Completion
    - [1.1 Building on Nuclear Norm Regularization](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/content/LRMC.ipynb)
    - [1.2 Building on Nonconvex Regularization](xxxx)

  - 2 Low-Rank Matrix Factorization
    - [2.1 A Gradient Descent Solution](xxxx)
    - [2.2 An Alternating Least Square Solution](xxxx)
    - [2.3 A Probabilistic Solution](xxxx)
    - [2.4 A Bayesian Solution](xxxx)

  - 3 Temporal Regularized Matrix Factorization
    - [3.1 An Alternating Least Square Solution](xxxx)
    - [3.2 A Probabilistic Solution](xxxx)

  - 4 Bayesian Temporal Matrix Factorization
    - [4.1 Incorporating Autoregressive (AR) Model](xxxx)
    - [4.2 Incorporating Vector Autoregressive (VAR) Model](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/content/BTMF.ipynb)

- Part 3: Tensor Factorization Techniques
  - [1  Tensor Factorization with Alternating Least Square (ALS)](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/part-03/chapter-01.ipynb)

  - [2  Nonnegative Tensor Factorization](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/part-03/chapter-02.ipynb)

  - [3  Bayesian Gaussian Tensor Factorization](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/part-03/chapter-03.ipynb)

- Part 4: Low-Rank Tensor Completion Techniques [coming soon!]
  - 1 Tensor Robust Principal Component Analysis
    - [1.1 Modeling for Tensor Recovery Problem](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/content/TRPCA.ipynb)
    - [1.2 Modeling for Outlier Detection Problem](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/content/TRPCA-Outlier.ipynb)

- Part 5: Multidimensional Tensor Regression [coming soon!]

Quick Run
---

- If you just want to read the tutorial, please follow the link of above contents directly.

- If you want to run the code, please

  - download (or clone) this repository,
  - open the `.ipynb` file using [Jupyter notebook](https://jupyter.org/install.html),
  - and run the code.

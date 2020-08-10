# Tensor Learning (张量学习)

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
[![GitHub stars](https://img.shields.io/github/stars/xinychen/tensor-learning.svg?logo=github&label=Stars&logoColor=white)](https://github.com/xinychen/tensor-learning)

<h6 align="center">Made by Xinyu Chen • :globe_with_meridians: <a href="https://twitter.com/chenxy346">https://twitter.com/chenxy346</a></h6>

Python codes for tensor factorization, tensor completion, and tensor regression techniques with the following real-world applications:

- [**geotensor**](https://github.com/xinychen/geotensor) | Image inpainting
- [**transdim**](https://github.com/xinychen/transdim) | Spatiotemporal traffic data imputation and prediction
- Recommender systems
- [**mats**](https://github.com/xinychen/tensor-learning/tree/master/mats) | Multivariate time series imputation and forecasting

In a hurry? Please check out our contents as follows.


<h2 align="center">Our Research</h2>
<p align="right"><a href="#tensor-learning-张量学习"><sup>▴ Back to top</sup></a></p>

We conduct extensive experiments on some real-world data sets:

  - [Large-scale PeMS traffic speed data set](https://doi.org/10.5281/zenodo.3939792) registers traffic speed time series from 11160 sensors over 4/8/12 weeks (for PeMS-4W/PeMS-8W/PeMS-12W) with 288 time points per day (i.e., 5-min frequency) in California, USA. You can download this data set and place it at the folder of `datasets`.
  
    - Data path example: `../datasets/California-data-set/pems-4w.csv`.
    - Open data in Python with `Pandas`:

```python
import pandas as pd

data = pd.read_csv('../datasets/California-data-set/pems-4w.csv', header = None)
```

## mats

**mats** is a project in the tensor learning repository, and it aims to develop **ma**chine learning models for multivariate **t**ime **s**eries forecasting. In this project, we propose the following low-rank tensor learning models:

- [x] [Low-Rank Autoregressive Tensor Completion (LATC)](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/mats/LATC-3min-intro.ipynb) with Nuclear Norm minimization (i.e., LATC-NN) and Truncated Nuclear Norm minimization (i.e., LATC-TNN) by [Chen and Sun, (2020)](https://arxiv.org/abs/2006.10436):

  - on middle-scale data sets (e.g., PeMS, Guangzhou, Electricity) [[xx](xx)] [[Python code](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/mats/LATC-predictor.ipynb)]
  - on large-scale data sets (e.g., PeMS-4W, PeMS-8W) [[xx](xx)]

- [x] Low-Tubal-Rank Autoregressive Tensor Completion (LATC-Tubal) by [Chen et al., (2020)](https://arxiv.org/abs/2008.03194) on large-scale data sets:

  - without autoregressive norm [[xx](xx)]
  - with autoregressive norm [[xx](xx)]

The **baseline models** include:

- [x] Bayesian Probabilistic Matrix Factorization (BPMF, [Salakhutdinov and Mnih, 2008](https://www.cs.toronto.edu/~amnih/papers/bpmf.pdf)) [[Python code](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/mats/BPMF-imputer.ipynb)]
  
- [x] Bayesian Gaussian CP decomposition (BGCP, [Chen et al., 2019](https://doi.org/10.1016/j.trc.2018.11.003)) [[Python code](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/mats/BGCP-imputer.ipynb)]

- [x] High-accuracy Low-Rank Tensor Completion (HaLRTC, [Liu et al., 2013](https://doi.org/10.1109/TPAMI.2012.39)) [[Python code](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/mats/HaLRTC-imputer.ipynb)]

- [x] Low-Rank Tensor Completion with Truncated Nuclear Norm minimization (LRTC-TNN, [Chen et al., 2020](https://doi.org/10.1016/j.trc.2020.102673)) [[Python code](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/mats/LRTC-TNN-imputer.ipynb)]

- [x] Tensor Nuclear Norm minimization with Discrete Cosine Transform (TNN-DCT, [Lu et al., 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_Low-Rank_Tensor_Completion_With_a_New_Tensor_Nuclear_Norm_Induced_CVPR_2019_paper.pdf)) [[Python code](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/mats/TNN-DCT-imputer.ipynb)]

<h2 align="center">:book: Reproducing Literature in Python</h2>
<p align="right"><a href="#tensor-learning-张量学习"><sup>▴ Back to top</sup></a></p>

We reproduce some tensor learning experiments in the previous literature.

| Year | Title | PDF | Authors' Code | Our Code | Status |
|:---|:------:|:---:|:---:|:-----:|----:|
|  2015 | Accelerated Online Low-Rank Tensor Learning for Multivariate Spatio-Temporal Streams | [ICML 2015](http://proceedings.mlr.press/v37/yua15.pdf) | [Matlab code](http://roseyu.com/Materials/accelerate_online_low_rank_tensor.zip) | [Python code](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/baselines/Online-LRTL.ipynb) | Under development |


<h2 align="center">:book: Tutorial</h2>
<p align="right"><a href="#tensor-learning-张量学习"><sup>▴ Back to top</sup></a></p>

We summarize some preliminaries for better understanding tensor learning. They are given in the form of tutorial as follows.

- **Foundations of Python Numpy Programming**

  - Generating random numbers in Matlab and Numpy [[Jupyter notebook](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/tutorial/random_matlab_numpy.ipynb)] [[blog post](xx)]

- **Foundations of Tensor Computations**

  - Kronecker product

- **Singular Value Decomposition (SVD)**

  - Randomized singular value decomposition [[Jupyter notebook](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/tutorial/randomized_svd.ipynb)] [[blog post](https://t.co/fkgMQTsz6G?amp=1)]
  - Tensor singular value decomposition

If you find these codes useful, please star (★) this repository.

<h2 align="center">Helpful Material</h2>
<p align="right"><a href="#tensor-learning-张量学习"><sup>▴ Back to top</sup></a></p>

We believe that these material will be a valuable and useful source for the readers in the further study or advanced research.

- Vladimir Britanak, Patrick C. Yip, K.R. Rao (2006). Discrete Cosine and Sine Transforms: General Properties, Fast Algorithms and Integer Approximations. Academic Press. [[About the book](https://www.sciencedirect.com/book/9780123736246/discrete-cosine-and-sine-transforms)]

- Ruye Wang (2010). Introduction to Orthogonal Transforms with Applications in Data Processing and Analysis. Cambridge University Press. [[PDF](http://fourier.eng.hmc.edu/book/lectures/mybook.pdf)]

- J. Nathan Kutz, Steven L. Brunton, Bingni Brunton, Joshua L. Proctor (2016). Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems. SIAM. [[About the book](http://www.dmdbook.com/)]

- Yimin Wei, Weiyang Ding (2016). Theory and Computation of Tensors: Multi-Dimensional Arrays. Academic Press.

- Steven L. Brunton, J. Nathan Kutz (2019). Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control. Cambridge University Press. [[PDF](http://databookuw.com/databook.pdf)] [[data & code](http://databookuw.com/)]

<h2 align="center">Quick Run</h2>
<p align="right"><a href="#tensor-learning-张量学习"><sup>▴ Back to top</sup></a></p>

- If you want to run the code, please
  - download (or clone) this repository,
  - open the `.ipynb` file using [Jupyter notebook](https://jupyter.org/install.html),
  - and run the code.

<h2 align="center">Citing</h2>
<p align="right"><a href="#tensor-learning-张量学习"><sup>▴ Back to top</sup></a></p>

This repository is from the following paper, please cite our paper if it helps your research.

- Xinyu Chen, Lijun Sun (2020). **Low-rank autoregressive tensor completion for multivariate time series forecasting**. arXiv: 2006.10436. [[preprint](https://arxiv.org/abs/2006.10436)] [[data & Python code](https://github.com/xinychen/tensor-learning)]


<h2 align="center">Acknowledgements</h2>
<p align="right"><a href="#tensor-learning-张量学习"><sup>▴ Back to top</sup></a></p>

This research is supported by the [Institute for Data Valorization (IVADO)](https://ivado.ca/en/ivado-scholarships/excellence-scholarships-phd/).

<h2 align="center">License</h2>
<p align="right"><a href="#tensor-learning-张量学习"><sup>▴ Back to top</sup></a></p>

This work is released under the MIT license.

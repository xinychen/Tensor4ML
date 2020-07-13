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

## mats

**mats** is a project in the tensor learning repository, and it aims to develop **ma**chine learning for multivariate **t**ime **s**eries forecasting.

<h5 align="center"><i>Low-Rank Autoregressive Tensor Completion for Multivariate Time Series Forecasting.<br>
  [<a href="https://arxiv.org/abs/2006.10436">arXiv</a>]</i></h5>

<p align="center">
<img align="middle" src="https://github.com/xinychen/transdim/blob/master/images/predictor-explained.png" width="700" />
</p>

<h6 align="center">
<b>Figure 1</b>: Illustration of our proposed Low-Rank Tensor Completion (LATC) imputer/predictor with a prediction window τ (green nodes: observed values; white nodes: missing values; red nodes/panel: prediction; blue panel: training data to construct the tensor).
</h6>

In this work, we develop a **Low-Rank Autoregressive Tensor Completion** for multivariate time series forecasting in the presence of missing values. To overcome the challenge of missing time series values, our LATC model takes into account:

- autoregressive process on the matrix structure to capture local temporal states,
- and low-rank assumption on the tensor structure to capture global low-rank patterns simultaneously.

Python codes for reproducing experiments are provided in the [**../mats**](https://github.com/xinychen/tensor-learning/tree/master/mats) folder. Since these Python codes were written on the Jupyter Notebook, you could also view them on the nbviewer. Please open

- [LATC-imputer.ipynb](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/mats/LATC-imputer.ipynb)
- [LATC-predictor.ipynb](https://nbviewer.jupyter.org/github/xinychen/tensor-learning/blob/master/mats/LATC-predictor.ipynb)

If you find these codes useful, please star (★) this repository.

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

<h2 align="center">Helpful Learning Material</h2>
<p align="right"><a href="#tensor-learning-张量学习"><sup>▴ Back to top</sup></a></p>

- Ruye Wang (2010). Introduction to Orthogonal Transforms with Applications in Data Processing and Analysis. Cambridge University Press. [[PDF](http://fourier.eng.hmc.edu/book/lectures/mybook.pdf)]

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

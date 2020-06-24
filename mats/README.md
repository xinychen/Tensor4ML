

# mats

**Ma**chine learning for multivariate **t**ime **s**eries with missing values.

> This folder includes our latest **imputer** and **predictor** for multivariate time series analysis.

-------------------------------------------



<h5 align="center"><i>Low-Rank Autoregressive Tensor Completion for Multivariate Time Series Forecasting.<br>
  [<a href="https://arxiv.org/abs/2006.10436">arXiv</a>]</i></h5>

<p align="center">
<img align="middle" src="https://github.com/xinychen/transdim/blob/master/images/predictor-explained.png" width="666" />
</p>

Our aim is to build multivariate time series forecasting tool on the well-understood Low-Rank Tensor Completion (LRTC). In this work, we develop a Low-Rank Autoregressive Tensor Completion (LATC) which takes into account:

- autoregressive process on the matrix structure to capture local temporal states,
- and low-rank assumption on the tensor structure to capture global low-rank patterns simultaneously,

for handling missing data problems.

We evaluate our LATC on both imputation and prediction tasks on **three real-world data sets**:

- (**P**) PeMS traffic speed data set (228 time series with 288 x 44 time points),
- (**G**) Guangzhou traffic speed data set (214 time series with 144 x 61 time points),
- and (**E**) Electricity data set (321 time series with 24 x 35 time points).

Code for reproducing experiments is provided in this folder. Please check out `LATC-imputer.ipynb` and `LATC-predictor.ipynb` for details.

### Quick example

Here, we would like to provide few steps for reproducing our experiments:

- Clone this GitHub repository on your personal computer first.
- Then open the notebook what you want to run on your Jupyter Notebook.

For your convenience, these notebooks do not rely on too much Python packages. You need to make sure there is `Numpy` on your Python.


### Results

- **Imputation (`imputer`)**

We provide this report by evaluating models on three real-world data sets with certain amount of missing values.

<p align="center">
<img align="middle" src="https://github.com/xinychen/transdim/blob/master/images/lact_imputation_result.png" width="500" />
</p>

> For more detail, please check out our paper, appendix, and Python code.


- **Prediction (`predictor`)**

We provide this report by evaluating models on three real-world data sets. The rolling forecasting scenario is accompanied with missing values, and these missing values are created both in completely random manner and non-random manner.

<p align="center">
<img align="middle" src="https://github.com/xinychen/transdim/blob/master/images/lact_prediction_result.png" width="500" />
</p>


> For more detail, please also check our paper, appendix, and Python code.


### Citation

If you use these codes in your work, please cite our paper:

```bibtex
@article{chen2020lowrank,
    author={Xinyu Chen, Lijun Sun},
    title={{Low-Rank Autoregressive Tensor Completion for Multivariate Time Series Forecasting}},
    year={2020},
    journal={arXiv:2006.10436}
}
```

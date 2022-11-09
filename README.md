<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.8 +-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div>
<br>

Predictive UNcertainty Calibration and Conformalization (PUNCC) is an open-source library that enables ad-hoc integration of AI models into a theoretically sound uncertainty estimation framework based on conformal prediction. Prediction sets are constructed with guaranteed coverage probability according to a nominal level of error $\alpha$.

## 📚 Table of contents

- [🐾 Instalation](#-installation)
- [🚀 Quick Start](#-quickstart)
- [📚 Citation](#-citation)
- [💻 Contributing](#-contributing)
- [📝 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

# 🐾 Installation

## Clone the repo

```bash
git clone ssh://git@forge.deel.ai:22012/statistic-guarantees/puncc.git
```

## Installation

It is recommended to install puncc in a virtual environment to not mess with your system's dependencies.

### Using pip

#### For developpers

```bash
pip install -e .[dev]
```

#### For users
```bash
pip install -e .[interactive]
```

### Using makefile

```bash
make prepare-dev
```

### From Makefile

```bash
make prepare-dev
```

# 🚀 Quickstart

We propose two ways of defining and using conformal prediction wrappers:
- A fast way based on preconfigured conformal prediction wrappers
- A flexible way based of full customization of the prediction model, the residual computation and the fit/calibration data split plan

A comparison of both approaches is provided [here](docs/quickstart.ipynb) for a simple regression problem. A showcase of basic uncertainty quantification features for the first approach is given hereafter.

## Data
```python
import numpy as np
from sklearn import datasets

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
X_train = diabetes_X[:-100]
X_test = diabetes_X[-100:]

# Split the targets into training/testing sets
y_train = diabetes_y[:-100]
y_test = diabetes_y[-100:]

# Split fit and calibration data
X_fit, X_calib = X_train[:-100], X_train[-100:]
y_fit, y_calib = y_train[:-100], y_train[-100:]
```
## Linear regression model

```python
from sklearn import linear_model

# Create linear regression model
regr = linear_model.LinearRegression()
```


## Split conformal prediction
``` python
from deel.puncc.regression import SplitCP

# Coverage target is 1-alpha = 90%
alpha=.1
# Instanciate the split cp wrapper on the linear model
split_cp = SplitCP(regr)
# Train model on the fitting dataset and compute residuals on the calibration
# dataset
split_cp.fit(X_fit, y_fit, X_calib, y_calib)
# Estimate the prediction interval
y_pred, y_pred_lower, y_pred_upper = split_cp.predict(X_test, alpha=alpha)
```

## Results

```python
from deel.puncc.api.utils import plot_prediction_interval

# Figure of the prediction bands

plot_prediction_interval(
    X = X_test[:,0],
    y_true=y_test,
    y_pred=y_pred,
    y_pred_lower=y_pred_lower,
    y_pred_upper=y_pred_upper,
    sort_X=True,
    size=(10, 6),
    loc="upper left")
```

![90% Prediction Interval with the Split Conformal Prediction Method](docs/source/results_quickstart_split_cp_pi.png)

## 📚 Citation

This library was built to support the work presented in our COPA 2022 paper on conformal prediction for timeseries. If you use our library for your work, please cite our paper:

```
@inproceedings{mendil2022robust,
  title={Robust Gas Demand Forecasting With Conformal Prediction},
  author={Mendil, Mouhcine and Mossina, Luca and Nabhan, Marc and Pasini, Kevin},
  booktitle={Conformal and Probabilistic Prediction with Applications},
  pages={169--187},
  year={2022},
  organization={PMLR}
}
```

## 💻 Contributing

Contributions are welcome! Feel free to report an issue or open a pull
request. Take a look at our guidelines [here](CONTRIBUTING.md).

## 🔑 License

The package is released under [MIT](LICENSES/headers/MIT-Clause.txt) license.

Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry, CRIAQ
and ANITI - https://www.deel.ai/

## 🙏 Acknowledgments

<img align="right" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png" width="25%">
This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.

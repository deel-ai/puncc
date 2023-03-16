<!-- Banner section -->
<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/banner_light2.png">
  <img src="docs/assets/banner_light.png" alt="PUNCC" width="80%" align="center" style="margin: 0px 0px 0px 10%;">
</picture>
</div>

<!-- Badge section -->
<div align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/Python-3.8 +-efefef">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/License-MIT-efefef">
  </a>
  <a href="https://github.com/deel-ai/puncc/actions/workflows/linter.yml">
    <img alt="PyLint" src="https://github.com/deel-ai/puncc/actions/workflows/linter.yml/badge.svg">
  </a>
  <a href="https://github.com/deel-ai/puncc/actions/workflows/tests.yml">
    <img alt="Tox" src="https://github.com/deel-ai/puncc/actions/workflows/tests.yml/badge.svg">
  </a>
</div>
<br>

Predictive UNcertainty Calibration and Conformalization (PUNCC) is an open-source Python library that integrates a collection of state-of-the-art Conformal Prediction algorithms and related techniques for regression and classification problems. PUNCC can be used with any predictive model to provide rigorous uncertainty estimations. Under data exchangeability (or *i.i.d*), the generated prediction sets are guaranteed to cover the true outputs within a user-defined error $\alpha$.

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

#### For users
```bash
pip install -e .[interactive]
```

You can alternatively use the makefile to automatically create a virtual environment
`puncc-user-env` and install all requirements:

```bash
make install-user
```

#### For developpers

```bash
pip install -e .[dev]
```

You can alternatively use the makefile to automatically create a virtual environment
`puncc-dev-env` and install the dev requirements:

```bash
make prepare-dev
```


# 🚀 Quickstart
<div align="center">

<font size=3>📙 You can find the detailed implementation of the example below in the [**Quickstart Notebook**](docs/quickstart.ipynb)</font>.

</div>
Let’s consider a simple regression problem on diabetes data provided by Scikit-learn. We want to evaluate the uncertainty associated with the prediction using inductive (or split) conformal prediction.

## Split Conformal Prediction

For this example, the prediction intervals are obtained throught the split
conformal prediction method provided by the class
`deel.puncc.regression.SplitCP`, applied on a linear model.

```python
from sklearn import linear_model
from deel.puncc.api.prediction import BasePredictor

# Load fiting (X_fit, y_fit) and calibration (X_calib, y_calib) data
# ...

# Use your favorite regression linear model 
# linear_model = ...


# Create a predictor to wrap the linear regression model defined earlier
# The argument `is_trained` is set to False to tell that the the linear model needs to be trained before the calibration.
lin_reg_predictor =  BasePredictor(linear_model, is_trained=False)

# Instanciate the split cp wrapper around the linear predictor.
split_cp = SplitCP(lin_reg_predictor)

# Train model (`is_trained` is False) on the fitting dataset and
# compute the residuals on the calibration dataset.
split_cp.fit(X_fit, y_fit, X_calib, y_calib)

# The `predict` returns the output of the linear model `y_pred` and
# the calibrated interval [`y_pred_lower`, `y_pred_upper`].
y_pred, y_pred_lower, y_pred_upper = split_cp.predict(X_test, alpha=alpha)
```

The library provides several metrics (`deel.puncc.metrics`) and plotting capabilities (`deel.puncc.plotting`) to evaluate and visualize the results of a conformal procedure. For a target error rate of $\alpha = 0.1$, the marginal coverage reached in this example on the test set is $95\%$ (see [Quickstart Notebook](docs/quickstart.ipynb)):

![90% Prediction Interval with the Split Conformal Prediction Method](docs/source/results_quickstart_split_cp_pi.png)

### Further readings: more flexibility with the API
The library PUNCC provides two ways of defining and using conformal prediction wrappers:
- A direct approach to run state-of-the-art conformal prediction procedures. This is what we used in the previous conformal regression example.
- **Low-level API**: a more flexible approach based of full customization of the prediction model, the choice of nonconformity scores and the split between fit and calibration datasets.

A quick comparison of both approaches is provided in the [Quickstart Notebook](docs/quickstart.ipynb) for a simple regression problem.

## 📚 Citation

This library was initially built to support the work presented in our COPA 2022 paper on conformal prediction for time series. If you use our library for your work, please cite our paper:

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

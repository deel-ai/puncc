<!-- Banner -->
<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/banner_light.png">
  <img src="docs/assets/banner_light.png" alt="Puncc" width="90%" align="right">
</picture>
</div>
<br>

<!-- Badges -->
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

***Puncc*** (short for **P**redictive **un**certainty **c**alibration and **c**onformalization) is an open-source Python library. It seamlessly integrates a collection of state-of-the-art conformal prediction algorithms and associated techniques for diverse machine learning tasks, including regression, classification and anomaly detection.
***Puncc*** can be used with any predictive model to provide rigorous uncertainty estimations.
Under data exchangeability (or *i.i.d*), the generated prediction sets are guaranteed to cover the true outputs within a user-defined error $\alpha$.

Documentation is available [**online**](https://deel-ai.github.io/puncc/index.html).

## 📚 Table of contents

- [🐾 Installation](#-installation)
- [📖 Documentation](#-documentation)
- [👨‍🎓 Tutorials](#-tutorials)
- [🚀 QuickStart](#-quickstart)
- [📚 Citation](#-citation)
- [💻 Contributing](#-contributing)
- [🙏 Acknowledgments](#-acknowledgments)
- [👨‍💻 Creators](#-creators)
- [📝 License](#-license)

## 🐾 Installation

*puncc* requires a version of python higher than 3.8 and several libraries including Scikit-learn and Numpy. It is recommended to install *puncc* in a virtual environment to not mess with your system's dependencies.

You can directly install the library using pip:

```bash
pip install git+https://github.com/deel-ai/puncc
```

<!--
You can alternatively clone the repo and use the makefile to automatically create a virtual environment
and install the requirements:

* For users:

```bash
make install-user
```

* For developpers:

```bash
make prepare-dev
```
-->

## 📖 Documentation

For comprehensive documentation, we encourage you to visit the [**official documentation page**](https://deel-ai.github.io/puncc/index.html).

## 👨‍🎓 Tutorials

We highly recommand following the introduction tutorials to get familiar with the library and its API:

* [**Introduction tutorial**](docs/puncc_intro.ipynb)</font> <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TC_BM7JaEYtBIq6yuYB5U4cJjeg71Tch) </sub>

* [**API tutorial**](docs/api_intro.ipynb) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1d06qQweM1X1eSrCnixA_MLEZil1vXewj) </sub>

You can also familiarize yourself with the architecture of *puncc* to build more efficiently your own conformal prediction methods:

* [**Architecture overview**](docs/puncc_architecture.ipynb)

## 🚀 Quickstart

Conformal prediction enables to transform point predictions into interval predictions with high probability of coverage. The figure below shows the result of applying the split conformal algorithm on a linear regressor.

<figure style="text-align:center">
<img src="docs/assets/cp_process.png"/>
</figure>

Many conformal prediction algorithms can easily be applied using *puncc*.  The code snippet below shows the example of split conformal prediction wrapping a linear model,  done in few lines of code:

```python
from sklearn import linear_model
from deel.puncc.api.prediction import BasePredictor

# Load training data and test data
# ...

# Instanciate a linear regression model
# linear_model = ...


# Create a predictor to wrap the linear regression model defined earlier.
# This enables interoperability with different ML libraries.
# The argument `is_trained` is set to False to tell that the the linear model
# needs to be trained before the calibration.
lin_reg_predictor =  BasePredictor(linear_model, is_trained=False)

# Instanciate the split cp wrapper around the linear predictor.
split_cp = SplitCP(lin_reg_predictor)

# Fit model (as is_trained` is False) on the fit dataset and
# compute the residuals on the calibration dataset.
# The fit (resp. calibration) subset is randomly sampled from the training
# data and constitutes 80% (resp. 20%) of it (fit_ratio = 80%).
split_cp.fit(X_train, y_train, fit_ratio=.8)

# The predict returns the output of the linear model y_pred and
# the calibrated interval [y_pred_lower, y_pred_upper].
y_pred, y_pred_lower, y_pred_upper = split_cp.predict(X_test, alpha=alpha)
```

The library provides several metrics (`deel.puncc.metrics`) and plotting capabilities (`deel.puncc.plotting`) to evaluate and visualize the results of a conformal procedure. For a target error rate of $\alpha = 0.1$, the marginal coverage reached in this example on the test set is higher than $90$% (see [Introduction tutorial](docs/puncc_intro.ipynb)):

<figure style="text-align:center">
<img src="docs/assets/results_quickstart_split_cp_pi.png" alt="90% Prediction Interval with the Split Conformal Prediction Method"/>
<div align=center>90% Prediction Interval with Split Conformal Prediction.</div>
</figure>
<br>

### More flexibility with the API

*Puncc* provides two ways of defining and using conformal prediction wrappers:
- A direct approach to run state-of-the-art conformal prediction procedures. This is what we used in the previous conformal regression example.
- **Low-level API**: a more flexible approach based of full customization of the prediction model, the choice of nonconformity scores and the split between fit and calibration datasets.

A quick comparison of both approaches is provided in the [API tutorial](docs/api_intro.ipynb) for a regression problem.

## 📚 Citation

If you use our library for your work, please cite our paper:

```
@inproceedings{mendil2023puncc,
  title={PUNCC: a Python Library for Predictive Uncertainty Calibration and Conformalization},
  author={Mendil, Mouhcine and Mossina, Luca and Vigouroux, David},
  booktitle={Conformal and Probabilistic Prediction with Applications},
  pages={582--601},
  year={2023},
  organization={PMLR}
}
```

*Puncc* has been used to support the work presented in our COPA 2022 paper on conformal prediction for time series.

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

## 🙏 Acknowledgments

<img align="right" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png" width="25%">
This project received funding from the French ”Investing for the Future – PIA3” program within the Artificial and Natural Intelligence Toulouse Institute (ANITI). The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.

## 👨‍💻 Creators

[Mouhcine MENDIL](https://github.com/M-Mouhcine) initially developed this library as a research tool, with assistance from [Lucas MOSSINA](https://github.com/lmossina). We have recently welcomed [Joseba DALMAU](https://github.com/jdalch) to the team to help enhance **puncc** and work on the development of new features.

## 🔑 License

The package is released under [MIT](LICENSES/headers/MIT-Clause.txt) license.

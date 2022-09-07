Predictive UNcertainty Calibration and Conformalization (PUNCC) is an open-source library that enables ad-hoc integration of AI models into a theoretically sound uncertainty estimation framework based on conformal prediction. Prediction sets are constructed with guaranteed coverage probability according to a nominal level of error $\alpha$.

# Installation

## Clone the repo

```bash
git clone ssh://git@forge.deel.ai:22012/statistic-guarantees/puncc.git
```

## Install the virtual environment

First install `virtualenv`:
```bash
pip install virtualenv
```
Then create a virtual environment named puncc, and activate it:
```bash
virtualenv env-puncc
# Under Mac OS / Linux
source env-puncc/bin/activate
```

## Install puncc
Install puncc and requirements

### For users:
```bash
pip install -e .[interactive]
```

### For developers:
```bash
pip install -e .[dev]
```

## Jupyter notebook

To use the current virtual environment in jupyter notebook, make sure to add it:

```bash
python -m ipykernel install --user --name=env-puncc
```

[comment]: <Later, to uninstall the kernel: jupyter kernelspec uninstall env-puncc>

# Quickstart

We propose two ways of defining and using conformal prediction wrappers:
- A fast way based on preconfigured conformal prediction wrappers
- A flexible way based of full customization of the prediction model, the residual computation and the fit/calibration data split plan

A comparison of both approaches is provided [here](doc/quickstart.ipynb) for a simple regression problem. A showcase of basic uncertainty quantification features for the first approach is given hereafter.

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
from puncc.common.confomalizers import SplitCP

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
from puncc.utils import plot_prediction_interval

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

![90% Prediction Interval with the Split Conformal Prediction Method](doc/results_quickstart_split_cp_pi.png)


# Architecture Overview

As mentionned [before](#quickstart), the present implementation enables a turnkey solution and a fully customized approach to conformal prediction. The former is as simple as calling the conformal prediction wrappers in `puncc.common.conformalizers` (as in [here](#split-conformal-prediction)). The currently implemented wrappers are the following:
* `puncc.common.conformalizers.SplitCP`: Split Conformal Prediction
* `puncc.common.conformalizers.LocallyAdaptiveCP`: Locally Adaptive Conformal Prediction
* `puncc.common.conformalizers.CQR`: Conformalized Quantile Regression
* `puncc.common.conformalizers.CvPlus`: CV + (cross-validation)
* `puncc.common.conformalizers.EnbPI`: Ensemble Batch Prediction Intervals method

Each of these wrappers conformalize point-based or interval-based models that are passed as argument in the object constructor. Such models **need** to implement the `fit` and `predict` methods and operate on numpy arrays. We will see later how to use models even when these requirements are not met (e.g., using pytorch and tensorflow on time series).

The **fully customized** approach offers more flexibility into defining conformal prediction wrappers. Let's say we want to fit/calibrate a neural-network interval-estimator with a cross-validation plan; or that we want to experiment different user-defined nonconformity scores. In such cases and others, the user can fully construct their wrappers using the proposed **Predictor-Calibrator-Splitter** paradigm. It boils down to assembling into `puncc.conformalization.ConformalPredictor`:
1) Regression model(s)
2) An estimator of nonconformity scores for construction/calibration of prediction intervals
3) A strategy of data assignement into fitting and calibration sets.

## ConformalPredictor

`puncc.conformalization.ConformalPredictor` is the canvas of conformal prediction wrappers. An object instance is constructed by, as we will explain later, a predictor, a calibrator and a splitter:

```python
# Imports
from sklearn import linear_model
from conformalization import ConformalPredictor
from predictor import MeanPredictor
from calibration import MeanCalibrator
from splitter import KFoldSplit

# Regression linear model
model = linear_model.LinearRegression()

# Definition of mean-based predictor, a mean-based calibrator and a K-fold splitter
# (This will be explained later)
mean_predictor = MeanPredictor(model) # Predictor
mean_calibrator = MeanCalibrator() # Calibrator
kfold_splitter = KFoldSplitter(K=20, random_state=42) # Splitter

# Conformal prediction canvas
conformal_predictor = ConformalPredictor(predictor=mean_predictor,
                                         calibrator=mean_calibrator,
                                         splitter=kfold_splitter)
```
`puncc.conformalization.ConformalPredictor` implements two methods:
* A `fit` method that fits the predictor model and computes nonconformity scores accodingly to the calibrator and to the data split strategy provided by the splitter

```python
# X_train and y_train are the full training dataset
# The splitter passed as argument to ConformalPredictor assigns data
# to the fit and calibration sets based on the provided splitting strategy
conformal_predictor.fit(X_train, y_train)
```
* a `predict` that estimates for new samples a 90% (1-alpha) coverage prediction interval [y_pred_low, y_pred_high] and point predictions if applicable (e.g., conditional mean and dispersion)
```python
# Coverage target of 1-alpha = 90%
y_pred, y_pred_low, y_pred_high, sigma_pred = conformal_predictor.predict(X_new, alpha=.1)
```

## Predictor

The `puncc.prediction.BasePredictor` class is a wrapper for regression models that aims to standardize their interface and force compliance with the previously presented requirements:
* The models have to implement the `fit` and `predict` methods
* The models have to operate on datasets formated as numpy arrays (in `_format`)

Any specific preprocessing should be included in a **subclass** of `puncc.prediction.BasePredictor`.

A special attention is to be directed towards the `predict` method.

```python
# Args:
#   X: new example features
# Returns:
#   A tuple composed of (y_pred, y_pred_lower, y_pred_upper, sigma_pred)
predict(X: numpy.array) -> (numpy.array, numpy.array, numpy.array, numpy.array)
```
The tuple returned by the method contains four elements:
*   `y_pred`: point predictions
*   `y_pred_lower`: lower bound of the prediction intervals
*   `y_pred_upper`: upper bound of the prediction intervals
*   `sigma_pred`: variability estimations

If the model does not estimate some of these values, they are substituted by `None` at the right position. For example, a quantile regressor that returns upper and lower interval bounds is wrapped as follows:

```python
class QuantilePredictor(BasePredictor):
    def __init__(self, q_lo_model, q_hi_model, is_trained=False):
        """
        Args:
            q_lo_model: lower quantile model
            q_hi_model: upper quantile model
        """
        super().__init__(is_trained)
        self.q_lo_model = q_lo_model
        self.q_hi_model = q_hi_model

    def fit(self, X, y, **kwargs):
        """Fit model to the training data.
        Args:
            X: train features
            y: train labels
        """
        self.q_lo_model.fit(X, y)
        self.q_hi_model.fit(X, y)
        self.is_trained = True

    def predict(self, X, **kwargs):
        """Compute predictions on new examples.
        Args:
            X: new examples' features
        Returns:
            y_pred, y_lower, y_upper, sigma_pred
        """
        y_pred_lower = self.q_lo_model.predict(X)
        y_pred_upper = self.q_hi_model.predict(X)
        return None, y_pred_lower, y_pred_upper, None
```

To cover a large range of conformal prediction methods, three subclasses of `puncc.prediction.BasePredictor` have been implemented:

* `puncc.prediction.MeanPredictor`: wrapper of point-based models
* `puncc.prediction.MeanVarPredictor`: wrapper of point-based models that also estimates variability statistics (e.g., standard deviation)
* `puncc.prediction.QuantilePredictor`: wrapper of specific interval-based models that estimate upper and lower quantiles of the data generating distribution

PS: User-defined predictors have to subclass `puncc.prediction.BasePredictor` and redefine its methods.

## Calibrator

The `puncc.calibration.Calibrator` is meant to compute **nonconformity scores** (e.g., residuals) on the calibration dataset (`puncc.calibration.Calibrator.estimate` method) and uses them to **construct** and/or **calibrate** prediction intervals (`puncc.calibration.Calibrator.calibrate` method).

To cover a large range of conformal prediction methods, three subclasses of `puncc.calibration.Calibrator` have been implemented:

* `puncc.calibration.MeanCalibrator`: constructs prediction intervals based on point predictions by adding and substracting quantiles of mean absolute deviations
* `puncc.calibration.MeanVarCalibrator`: constructs prediction intervals based on point predictions and dispersion estimations by adding and substracting quantiles of **scaled** mean absolute deviations
* `puncc.calibration.QuantileCalibrator`: calibrates quantile-based prediction intervals by shrinking or dialating them accordingly to the quantiles of nonconformity scores

PS: User-defined calibrators have to subclass `puncc.calibration.Calibrator` and redefine its methods.

## Splitter

In conformal prediction, the assignement of data to fitting and calibration sets is motivated by two criteria: data availability and computational resources. If quality data is abundant, we can split the training samples into disjoint subsets $D_{fit}$ and $D_{calib}$. When data is scarce, a cross-validation strategy is preferred but is more ressources consuming as different models are trained and nonconformity scores are computed for different disjoint folds.

The two plans are implemented in `puncc.splitting` module.
* `puncc.splitting.RandomSplitter`: assignement of samples in $D_{fit}$ and $D_{calib}$
* `puncc.splitting.KFoldSplitter`: assignement of samples into K disjoint folds. Note that if K equals the size of training set, the split is identified with the leave-one-out strategy

These methods produce **iterables** that are used by the `ConformalPredictor` instance.

Additionnaly, if the user already implemted a split plan, the obtained data asignement is wrapped in `puncc.splitting.IdSplitter` to produce iterables.

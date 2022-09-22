# Architecture Overview

The present implementation enables a turnkey solution and a fully customized approach to conformal prediction. The former is as simple as calling the conformal prediction wrappers in `deel.puncc.regression`. The currently implemented wrappers are the following:
* `deel.puncc.regression.SplitCP`: Split Conformal Prediction
* `deel.puncc.regression.LocallyAdaptiveCP`: Locally Adaptive Conformal Prediction
* `deel.puncc.regression.CQR`: Conformalized Quantile Regression
* `deel.puncc.regression.CvPlus`: CV + (cross-validation)
* `deel.puncc.regression.EnbPI`: Ensemble Batch Prediction Intervals method
* `deel.puncc.regression.WeightedSCP`: Weighted Split Conformal Prediction

Each of these wrappers conformalize point-based or interval-based models that are passed as argument in the object constructor. Such models **need** to implement the `fit` and `predict` methods and operate on numpy arrays. We will see later how to use models even when these requirements are not met (e.g., using pytorch and tensorflow on time series).

The **fully customized** approach offers more flexibility into defining conformal prediction wrappers. Let's say we want to fit/calibrate a neural-network interval-estimator with a cross-validation plan; or that we want to experiment different user-defined nonconformity scores. In such cases and others, the user can fully construct their wrappers using the proposed **Predictor-Calibrator-Splitter** paradigm. It boils down to assembling into `puncc.api.conformalization.ConformalPredictor`:
1) Regression model(s)
2) An estimator of nonconformity scores for construction/calibration of prediction intervals
3) A strategy of data assignement into fitting and calibration sets.

## ConformalPredictor

`deel.puncc.api.conformalization.ConformalPredictor` is the canvas of conformal prediction wrappers. An object instance is constructed by, as we will explain later, a predictor, a calibrator and a splitter:

```python
# Imports
from sklearn import linear_model
from deel.puncc.api.conformalization import ConformalPredictor
from deel.puncc.api.predictor import MeanPredictor
from deel.puncc.api.calibration import MeanCalibrator
from deel.puncc.api.splitter import KFoldSplit

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
`deel.puncc.api.conformalization.ConformalPredictor` implements two methods:
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

The `deel.puncc.api.prediction.BasePredictor` class is a wrapper for regression models that aims to standardize their interface and force compliance with the previously presented requirements:
* The models have to implement the `fit` and `predict` methods
* The models have to operate on datasets formated as numpy arrays (in `_format`)

Any specific preprocessing should be included in a **subclass** of `deel.puncc.api.prediction.BasePredictor`.

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

To cover a large range of conformal prediction methods, three subclasses of `deel.puncc.api.prediction.BasePredictor` have been implemented:

* `deel.puncc.api.prediction.MeanPredictor`: wrapper of point-based models
* `deel.puncc.api.prediction.MeanVarPredictor`: wrapper of point-based models that also estimates variability statistics (e.g., standard deviation)
* `deel.puncc.api.prediction.QuantilePredictor`: wrapper of specific interval-based models that estimate upper and lower quantiles of the data generating distribution

PS: User-defined predictors have to subclass `deel.puncc.api.prediction.BasePredictor` and redefine its methods.

## Calibrator

The `deel.puncc.api.calibration.Calibrator` is meant to compute **nonconformity scores** (e.g., residuals) on the calibration dataset (`deel.puncc.api.calibration.Calibrator.estimate` method) and uses them to **construct** and/or **calibrate** prediction intervals (`deel.puncc.api.calibration.Calibrator.calibrate` method).

To cover a large range of conformal prediction methods, three subclasses of `deel.puncc.api.calibration.Calibrator` have been implemented:

* `deel.puncc.api.calibration.MeanCalibrator`: constructs prediction intervals based on point predictions by adding and substracting quantiles of mean absolute deviations
* `deel.puncc.api.calibration.MeanVarCalibrator`: constructs prediction intervals based on point predictions and dispersion estimations by adding and substracting quantiles of **scaled** mean absolute deviations
* `deel.puncc.api.calibration.QuantileCalibrator`: calibrates quantile-based prediction intervals by shrinking or dialating them accordingly to the quantiles of nonconformity scores

PS: User-defined calibrators have to subclass `deel.puncc.api.calibration.Calibrator` and redefine its methods.

## Splitter

In conformal prediction, the assignement of data to fitting and calibration sets is motivated by two criteria: data availability and computational resources. If quality data is abundant, we can split the training samples into disjoint subsets $D_{fit}$ and $D_{calib}$. When data is scarce, a cross-validation strategy is preferred but is more ressources consuming as different models are trained and nonconformity scores are computed for different disjoint folds.

The two plans are implemented in `deel.puncc.api.splitting` module.
* `deel.puncc.api.splitting.RandomSplitter`: assignement of samples in $D_{fit}$ and $D_{calib}$
* `deel.puncc.api.splitting.KFoldSplitter`: assignement of samples into K disjoint folds. Note that if K equals the size of training set, the split is identified with the leave-one-out strategy

These methods produce **iterables** that are used by the `ConformalPredictor` instance.

Additionnaly, if the user already implemted a split plan, the obtained data asignement is wrapped in `deel.puncc.api.splitting.IdSplitter` to produce iterables.

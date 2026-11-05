# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements usual conformal regression wrappers.
"""

from deel.puncc.nonconformity_scores import (
    absolute_difference,
    cqr_score,
)  # ,scaled_ad
from deel.puncc.prediction_sets import (
    constant_interval,
    cqr_interval,
)  # , scaled_interval
from deel.puncc.core.split import (
    PresetSplitConformalPredictor,
    LocallyAdaptiveMixin,
    LeverageWeightedMixin,
)


class SplitConformalRegression(PresetSplitConformalPredictor):
    """
    Implementation of split conformal regression.

    This method constructs prediction intervals by calibrating the absolute prediction errors on a held-out
    calibration dataset.

    Args:
        model: Predictive model returning point predictions with shape (n_samples,).
        fit_function: Optional custom function used to train the model. Defaults to None.

    Returns:
        The predict() method returns a ConformalPrediction containing:
            - The point predictions, with shape (n_samples,).
            - The prediction intervals, with shape (n_samples, 2), ordered as (lower_bound, upper_bound).

    Example::

        import numpy as np

        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor

        from deel.puncc.regression.split import SplitConformalRegression
        from deel.puncc.metrics import (
            regression_mean_coverage,
            regression_mean_width,
        )

        # Generate a random regression problem
        X, y = make_regression(
            n_samples=1000,
            n_features=4,
            n_informative=2,
            noise=10,
            random_state=0,
        )

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Split training data into fit and calibration sets
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0
        )

        # Create a random forest regressor
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=0,
        )

        # Initialize the conformal predictor
        cp = SplitConformalRegression(model=rf_model)

        # Fit the model and calibrate separately
        cp.fit(X_fit, y_fit)
        cp.calibrate(X_calib, y_calib)

        # Construct prediction intervals for alpha = 20%
        y_pred, intervals = cp.predict(X_test, alpha=0.2)

        y_pred_lower = intervals[:, 0]
        y_pred_upper = intervals[:, 1]

        # Compute marginal coverage and average interval width
        coverage = regression_mean_coverage(
            y_test, y_pred_lower, y_pred_upper
        )
        width = regression_mean_width(
            y_pred_lower, y_pred_upper
        )

        print(f"Marginal coverage: {np.round(coverage, 2)}")
        print(f"Average interval width: {np.round(width, 2)}")
    """

    nc_score_function = absolute_difference()
    pred_set_function = constant_interval()


class CQR(PresetSplitConformalPredictor):
    """
    Implementation of Conformalized Quantile Regression (CQR).

    This method constructs conformal prediction intervals from lower and upper quantile predictions.
    Calibration adjusts the predicted bounds using nonconformity scores computed on a held-out calibration set.

    Args:
        model: Predictive model returning lower and upper quantile predictions with shape (n_samples, 2),
            ordered as (lower_quantile, upper_quantile).
        fit_function: Optional custom function used to train the model.
            Defaults to None.

    Notes:
        The predict() method returns a ConformalPrediction containing the original quantile predictions and
        the calibrated prediction intervals, both with shape (n_samples, 2).

    Example::

        import numpy as np

        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import GradientBoostingRegressor

        from deel.puncc.regression.split import CQR
        from deel.puncc.core.predictors import stack_predictors
        from deel.puncc.metrics import (
            regression_mean_coverage,
            regression_sharpness,
        )

        # Generate a random regression problem
        X, y = make_regression(
            n_samples=1000,
            n_features=4,
            n_informative=2,
            noise=10,
            random_state=0,
        )

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Split training data into fit and calibration sets
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0
        )

        # Create lower and upper quantile regressors
        lower_model = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.1,
            random_state=0,
        )
        upper_model = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.9,
            random_state=0,
        )

        # Stack the two quantile predictors
        model = stack_predictors(lower_model, upper_model)

        # Initialize the conformal predictor
        cp = CQR(model=model)

        # Fit the models and calibrate separately
        cp.fit(X_fit, y_fit)
        cp.calibrate(X_calib, y_calib)

        # Construct prediction intervals for alpha = 20%
        y_pred, intervals = cp.predict(X_test, alpha=0.2)

        y_pred_lower = intervals[:, 0]
        y_pred_upper = intervals[:, 1]

        # Compute marginal coverage and average interval width
        coverage = regression_mean_coverage(
            y_test, y_pred_lower, y_pred_upper
        )
        width = regression_sharpness(
            y_pred_lower, y_pred_upper
        )

        print(f"Marginal coverage: {np.round(coverage, 2)}")
        print(f"Average interval width: {np.round(width, 2)}")
    """

    nc_score_function = cqr_score()
    pred_set_function = cqr_interval()


class LocallyAdaptiveCP(LocallyAdaptiveMixin, SplitConformalRegression):
    """
    Implementation of locally adaptive conformal regression.

    This method constructs prediction intervals whose widths adapt to the estimated local prediction error.
    A second predictive model estimates the dispersion of the residuals of the main model.

    During calibration, absolute residuals are normalized by the estimated dispersion.
    At prediction time, the conformal quantile is rescaled using the dispersion estimated for each test sample.

    Args:
        model: Predictive model returning point predictions with shape (n_samples,).
            dispersion_estimator: Predictive model used to estimate the dispersion of the main model's residuals.
            Its predictions must be strictly positive and have shape (n_samples,) or (n_samples, 1).
        dispersion_estimation_function: Function computing the training targets for the dispersion estimator from the
            main model's predictions and the true targets.
            Defaults to absolute_difference().
        fit_function: Optional custom function used to train the main predictive model.
            Defaults to None.

    Notes:
        The dispersion estimator is fitted automatically when fit() is called, using residuals computed on the training dataset.

        The predict() method returns a ConformalPrediction containing the point predictions and the locally adaptive prediction
        intervals, with shape (n_samples, 2).

        The dispersion estimates must be strictly positive during calibration and prediction.

    Example::

        import numpy as np

        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor

        from deel.puncc.regression.split import LocallyAdaptiveCP
        from deel.puncc.metrics import (
            regression_mean_coverage,
            regression_sharpness,
        )

        # Generate a regression problem with heteroscedastic noise
        X, y = make_regression(
            n_samples=1000,
            n_features=4,
            n_informative=2,
            noise=0,
            random_state=0,
        )

        rng = np.random.default_rng(0)
        y = y + rng.normal(
            loc=0,
            scale=5 + 10 * np.abs(X[:, 0]),
            size=len(y),
        )

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Split training data into fit and calibration sets
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0
        )

        # Create the main regression model
        mean_model = RandomForestRegressor(
            n_estimators=100,
            random_state=0,
        )

        # Create a dispersion estimator with positive predictions
        class DispersionEstimator:
            def __init__(self):
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=1,
                )

            def fit(self, X, y):
                self.model.fit(X, y)
                return self

            def __call__(self, X):
                return np.maximum(self.model.predict(X), 1e-6)

        # Initialize the conformal predictor
        cp = LocallyAdaptiveCP(
            model=mean_model,
            dispersion_estimator=DispersionEstimator(),
        )

        # Fit both models and calibrate separately
        cp.fit(X_fit, y_fit)
        cp.calibrate(X_calib, y_calib)

        # Construct locally adaptive intervals for alpha = 20%
        y_pred, intervals = cp.predict(X_test, alpha=0.2)

        y_pred_lower = intervals[:, 0]
        y_pred_upper = intervals[:, 1]

        # Compute marginal coverage and average interval width
        coverage = regression_mean_coverage(
            y_test, y_pred_lower, y_pred_upper
        )
        width = regression_sharpness(
            y_pred_lower, y_pred_upper
        )

        print(f"Marginal coverage: {np.round(coverage, 2)}")
        print(f"Average interval width: {np.round(width, 2)}")
    """


class LeverageWeightedCP(LeverageWeightedMixin, SplitConformalRegression):
    """
    Implementation of leverage-weighted conformal regression.

    This method constructs prediction intervals whose widths depend on the statistical leverage of the input samples.

    Leverage scores are computed from standardized training features.
    They are used to scale nonconformity scores during calibration and to adjust interval widths at prediction time.

    Args:
        model: Predictive model returning point predictions with shape (n_samples,).
        leverage_weight_function: Function mapping leverage scores to strictly positive weights.
            Defaults to inverse_root_leverage_weight, defined as 1 / sqrt(1 + leverage).
        fit_function: Optional custom function used to train the model.
            Defaults to None.

    Notes:
        The leverage statistics are computed automatically during fit().

        Training inputs must contain more samples than features, and no feature may be constant.

        The predict() method returns a ConformalPrediction containing point predictions and prediction intervals of
        shape (n_samples, 2).

    Example::

        import numpy as np

        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression

        from deel.puncc.regression.split import LeverageWeightedCP
        from deel.puncc.metrics import (
            regression_mean_coverage,
            regression_sharpness,
        )

        # Generate a random regression problem
        X, y = make_regression(
            n_samples=1000,
            n_features=4,
            n_informative=2,
            noise=10,
            random_state=0,
        )

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Split training data into fit and calibration sets
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0
        )

        # Create a linear regression model
        model = LinearRegression()

        # Initialize the conformal predictor
        cp = LeverageWeightedCP(model=model)

        # Fit the model and compute training leverage statistics
        cp.fit(X_fit, y_fit)

        # Calibrate the conformal predictor
        cp.calibrate(X_calib, y_calib)

        # Construct leverage-adaptive intervals for alpha = 20%
        y_pred, intervals = cp.predict(X_test, alpha=0.2)

        y_pred_lower = intervals[:, 0]
        y_pred_upper = intervals[:, 1]

        # Compute marginal coverage and average interval width
        coverage = regression_mean_coverage(
            y_test, y_pred_lower, y_pred_upper
        )
        width = regression_sharpness(
            y_pred_lower, y_pred_upper
        )

        print(f"Marginal coverage: {np.round(coverage, 2)}")
        print(f"Average interval width: {np.round(width, 2)}")
    """

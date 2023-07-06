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

import numpy as np

from deel.puncc.regression import (
    SplitCP,
    LocallyAdaptiveCP,
    CQR,
    CVPlus,
    EnbPI,
    AdaptiveEnbPI,
)

from deel.puncc.api.prediction import (
    BasePredictor,
    MeanVarPredictor,
    DualPredictor,
)

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from deel.puncc.metrics import regression_mean_coverage
from deel.puncc.metrics import regression_sharpness


def test_splitcp():
    # Generate a random regression problem
    X, y = make_regression(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        random_state=0,
        shuffle=False,
    )

    # Split data into train and test
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Split train data into fit and calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Create a random forest model and wrap it in a predictor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_predictor = BasePredictor(rf_model, is_trained=False)

    # CP method initialization
    split_cp = SplitCP(rf_predictor)

    # The call to `fit` trains the model and computes the nonconformity
    # scores on the calibration set
    split_cp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)

    # The predict method infers prediction intervals with respect to
    # the significance level alpha = 20%
    y_pred, y_pred_lower, y_pred_upper = split_cp.predict(X_test, alpha=0.2)

    # Compute marginal coverage and average width of the prediction intervals
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(
        y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper
    )
    assert y_pred is not None
    assert y_pred_lower is not None
    assert y_pred_upper is not None


def test_lacp():
    # Generate a random regression problem
    X, y = make_regression(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        random_state=0,
        shuffle=False,
    )

    # Split data into train and test
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Split train data into fit and calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Create two models mu (mean) and sigma (dispersion)
    mu_model = RandomForestRegressor(n_estimators=100, random_state=0)
    sigma_model = RandomForestRegressor(n_estimators=100, random_state=0)
    # Wrap models in a mean/variance predictor
    mean_var_predictor = MeanVarPredictor(models=[mu_model, sigma_model])

    # CP method initialization
    lacp = LocallyAdaptiveCP(mean_var_predictor)

    # The call to `fit` trains the model and computes the nonconformity
    # scores on the calibration set
    lacp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)

    # The predict method infers prediction intervals with respect to
    # the significance level alpha = 20%
    y_pred, y_pred_lower, y_pred_upper = lacp.predict(X_test, alpha=0.2)

    # Compute marginal coverage and average width of the prediction intervals
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(
        y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper
    )
    assert y_pred is not None
    assert y_pred_lower is not None
    assert y_pred_upper is not None


def test_cqr():
    # Generate a random regression problem
    X, y = make_regression(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        random_state=0,
        shuffle=False,
    )

    # Split data into train and test
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Split train data into fit and calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Lower quantile regressor
    regressor_q_low = GradientBoostingRegressor(
        loss="quantile", alpha=0.2 / 2, n_estimators=250
    )
    # Upper quantile regressor
    regressor_q_hi = GradientBoostingRegressor(
        loss="quantile", alpha=1 - 0.2 / 2, n_estimators=250
    )
    # Wrap models in predictor
    predictor = DualPredictor(models=[regressor_q_low, regressor_q_hi])

    # CP method initialization
    crq = CQR(predictor)

    # The call to `fit` trains the model and computes the nonconformity
    # scores on the calibration set
    crq.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)

    # The predict method infers prediction intervals with respect to
    # the significance level alpha = 20%
    y_pred, y_pred_lower, y_pred_upper = crq.predict(X_test, alpha=0.2)

    # Compute marginal coverage and average width of the prediction intervals
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(
        y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper
    )
    assert y_pred is not None
    assert y_pred_lower is not None
    assert y_pred_upper is not None


def test_cvplus():
    # Generate a random regression problem
    X, y = make_regression(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        random_state=0,
        shuffle=False,
    )

    # Split data into train and test
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create a random forest model and wrap it in a predictor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_predictor = BasePredictor(rf_model, is_trained=False)

    # CP method initialization
    cv_cp = CVPlus(rf_predictor, K=20, random_state=0)

    # The call to `fit` trains the model and computes the nonconformity
    # scores on the K-fold calibration sets
    cv_cp.fit(X, y)

    # The predict method infers prediction intervals with respect to
    # the significance level alpha = 20%
    y_pred, y_pred_lower, y_pred_upper = cv_cp.predict(X_test, alpha=0.2)

    # Compute marginal coverage and average width of the prediction intervals
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(
        y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper
    )

    assert y_pred_lower is not None
    assert y_pred_upper is not None


def test_enbpi():
    # Generate a random regression problem
    X, y = make_regression(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        random_state=0,
        shuffle=False,
    )

    # Split data into train and test
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create rf regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    # Wrap model in a predictor
    rf_predictor = BasePredictor(rf_model)
    # CP method initialization
    enbpi = EnbPI(
        rf_predictor,
        B=30,
        agg_func_loo=np.mean,
        random_state=0,
    )

    # The call to `fit` trains the model and computes the nonconformity
    # scores on the oob calibration sets
    enbpi.fit(X, y)

    # The predict method infers prediction intervals with respect to
    # the significance level alpha = 20%
    y_pred, y_pred_lower, y_pred_upper = enbpi.predict(
        X_test, alpha=0.2, y_true=y_test, s=None
    )

    # Compute marginal coverage and average width of the prediction intervals
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(
        y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper
    )

    assert y_pred is not None
    assert y_pred_lower is not None
    assert y_pred_upper is not None


def test_aenbpi():
    # Generate a random regression problem
    X, y = make_regression(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        random_state=0,
        shuffle=False,
    )

    # Split data into train and test
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create two models mu (mean) and sigma (dispersion)
    mean_model = RandomForestRegressor(n_estimators=100, random_state=0)
    sigma_model = RandomForestRegressor(n_estimators=100, random_state=0)

    # Wrap models in a mean/variance predictor
    mean_var_predictor = MeanVarPredictor([mean_model, sigma_model])

    # CP method initialization
    aenbpi = AdaptiveEnbPI(
        mean_var_predictor,
        B=30,
        agg_func_loo=np.mean,
        random_state=0,
    )

    # The call to `fit` trains the model and computes the nonconformity
    # scores on the oob calibration sets
    aenbpi.fit(X, y)

    # The predict method infers prediction intervals with respect to
    # the significance level alpha = 20%
    y_pred, y_pred_lower, y_pred_upper = aenbpi.predict(
        X_test, alpha=0.2, y_true=y_test, s=None
    )

    # Compute marginal coverage and average width of the prediction intervals
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(
        y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper
    )

    assert y_pred is not None
    assert y_pred_lower is not None
    assert y_pred_upper is not None

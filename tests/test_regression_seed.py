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
import pytest
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from deel.puncc.api.prediction import BasePredictor
from deel.puncc.api.prediction import DualPredictor
from deel.puncc.api.prediction import MeanVarPredictor
from deel.puncc.metrics import regression_mean_coverage
from deel.puncc.metrics import regression_sharpness
from deel.puncc.regression import CQR
from deel.puncc.regression import CvPlus
from deel.puncc.regression import EnbPI
from deel.puncc.regression import LocallyAdaptiveCP
from deel.puncc.regression import SplitCP

# from deel.puncc.regression import AdaptiveEnbPI


RESULTS = {
    "scp": {"cov": 0.95, "width": 218.98},
    "nescp": {"cov": 0.96, "width": 230.1},
    "lacp": {"cov": 0.96, "width": 347.87},
    "cqr": {"cov": 0.93, "width": 204.52},
    "cv+": {"cov": 0.9, "width": 231.04},
    "enbpi": {"cov": 0.9, "width": 221.5},
    "aenbpi": {"cov": 0.87, "width": 272.14},
}


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_split_cp(diabetes_data, alpha, random_state):
    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data
    # split train data into fit and calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, random_state=random_state
    )

    # Create linear regression predictor
    predictor = BasePredictor(linear_model.LinearRegression())
    # CP method initialization
    split_cp = SplitCP(predictor)

    # The fit method trains the model and computes the residuals on the
    # calibration set
    split_cp.fit(X_fit, y_fit, X_calib, y_calib)  # type: ignore
    # The predict method infers prediction intervals with respect to
    # the risk alpha
    y_pred, y_pred_lower, y_pred_upper = split_cp.predict(X_test, alpha=alpha)
    assert y_pred is not None

    # Compute marginal coverage
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)
    res = {"cov": np.round(coverage, 2), "width": np.round(width, 2)}
    assert RESULTS["scp"] == res


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_ne_split_cp(diabetes_data, alpha, random_state):
    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data
    # split train data into fit and calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, random_state=random_state
    )

    def w_estimator_gen(gamma):
        def w_estimator(X):
            return [gamma ** (len(X) + 1 - i) for i in range(len(X))]

        return w_estimator

    # Create linear regression predictor
    predictor = BasePredictor(linear_model.LinearRegression())
    # CP method initialization
    w_split_cp = SplitCP(predictor, weight_func=w_estimator_gen(0.95))
    # The fit method trains the model and computes the residuals on the
    # calibration set
    w_split_cp.fit(X_fit, y_fit, X_calib, y_calib)  # type: ignore
    # The predict method infers prediction intervals with respect to
    # the risk alpha
    y_pred, y_pred_lower, y_pred_upper = w_split_cp.predict(X_test, alpha=alpha)
    assert y_pred is not None

    # Compute marginal coverage
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)
    res = {"cov": np.round(coverage, 2), "width": np.round(width, 2)}
    assert RESULTS["nescp"] == res


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_locally_adaptive_cp(diabetes_data, alpha, random_state):
    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data
    # split train data into fit and calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, random_state=random_state
    )

    # Create linear regression object
    mu_model = linear_model.LinearRegression()
    # Create RF regression object
    var_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    # Create predictor
    predictor = MeanVarPredictor(models=[mu_model, var_model])
    # CP method initialization
    la_cp = LocallyAdaptiveCP(predictor)

    # Fit and conformalize
    la_cp.fit(X_fit, y_fit, X_calib, y_calib)  # type: ignore
    y_pred, y_pred_lower, y_pred_upper = la_cp.predict(X_test, alpha=alpha)
    assert y_pred is not None

    # Compute marginal coverage
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)
    res = {"cov": np.round(coverage, 2), "width": np.round(width, 2)}
    assert RESULTS["lacp"] == res


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_cqr(diabetes_data, alpha, random_state):
    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data
    # split train data into fit and calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, random_state=random_state
    )

    gbr_params = {
        "n_estimators": 250,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_samples_leaf": 9,
        "min_samples_split": 9,
        "random_state": random_state,
    }

    # Lower quantile regression
    regressor_q_low = GradientBoostingRegressor(
        loss="quantile", alpha=alpha / 2, **gbr_params
    )
    # Upper quantile regression
    regressor_q_hi = GradientBoostingRegressor(
        loss="quantile", alpha=1 - alpha / 2, **gbr_params
    )
    # Wrap models in predictor
    predictor = DualPredictor(models=[regressor_q_low, regressor_q_hi])
    # CP method initialization
    crq = CQR(predictor)

    # Fit and conformalize
    crq.fit(X_fit, y_fit, X_calib, y_calib)  # type: ignore
    _, y_pred_lower, y_pred_upper = crq.predict(X_test, alpha=alpha)

    # Compute marginal coverage
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)
    res = {"cov": np.round(coverage, 2), "width": np.round(width, 2)}

    assert RESULTS["cqr"] == res


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_cv_plus(diabetes_data, alpha, random_state):
    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data

    # Create RF regression object and wrap it by a predictor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    predictor = BasePredictor(rf_model)
    # CP method initialization
    cv_cp = CvPlus(predictor, K=20, random_state=random_state)

    # Fit and conformalize
    cv_cp.fit(X_train, y_train)
    _, y_pred_lower, y_pred_upper = cv_cp.predict(X_test, alpha=alpha)

    # Compute marginal coverage
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)
    res = {"cov": np.round(coverage, 2), "width": np.round(width, 2)}
    assert RESULTS["cv+"] == res


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_enbpi(diabetes_data, alpha, random_state):
    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data
    # Create RF regression object and wrap it by a predictor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    predictor = BasePredictor(rf_model)
    # Fit and conformalize
    enbpi = EnbPI(predictor, B=30, agg_func_loo=np.mean, random_state=random_state)
    enbpi.fit(X_train, y_train)
    y_pred, y_pred_lower, y_pred_upper = enbpi.predict(
        X_test, alpha=alpha, y_true=y_test, s=None
    )
    assert y_pred is not None
    # Compute marginal coverage
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)
    res = {"cov": np.round(coverage, 2), "width": np.round(width, 2)}
    assert RESULTS["enbpi"] == res


# @pytest.mark.parametrize(
#     "alpha, random_state",
#     [(0.1, 42)],
# )
# def test_adaptive_enbpi(diabetes_data, alpha, random_state):
#     # Get data
#     (X_train, X_test, y_train, y_test) = diabetes_data
#     # Create RF regression object
#     rf_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
#     var_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
#     # Fit and conformalize
#     aenbpi = AdaptiveEnbPI(
#         rf_model,
#         var_model,
#         B=30,
#         agg_func_loo=np.mean,
#         random_state=random_state,
#     )
#     aenbpi.fit(X_train, y_train)
#     y_pred, y_pred_lower, y_pred_upper = aenbpi.predict(
#         X_test, alpha=alpha, y_true=y_test, s=None
#     )
#     assert y_pred is not None
#     # Compute marginal coverage
#     coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
#     width = regression_sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)
#     res = {"cov": np.round(coverage, 2), "width": np.round(width, 2)}
#     assert RESULTS["aenbpi"] == res

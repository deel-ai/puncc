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
from typing import Callable

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

from deel.puncc.api import nonconformity_scores
from deel.puncc.api import prediction_sets
from deel.puncc.api.calibration import BaseCalibrator
from deel.puncc.api.calibration import ClasswiseCalibrator
from deel.puncc.api.calibration import CvPlusCalibrator
from deel.puncc.api.calibration import LeveragedCalibrator
from deel.puncc.api.calibration import ScoreCalibrator


@pytest.mark.parametrize(
    "alpha, random_state",
    [[0.1, 42], [0.3, 42], [0.5, 42], [0.7, 42], [0.9, 42]],
)
def test_regression_calibrator(rand_reg_data, alpha, random_state):
    # Generate data
    (y_pred_calib, y_calib, y_pred_test, y_test) = rand_reg_data

    # Nonconformity score function that takes as argument
    # the predicted values y_pred = model(X) and the true labels y_true. In
    # this example, we reimplement the mean absolute deviation that is
    # already defined in `deel.puncc.api.nonconformity_scores.mad`
    def nonconformity_function(y_pred, y_true):
        return np.abs(y_pred - y_true)

    # Prediction sets are computed based on points predictions and
    # the quantiles of nonconformity scores. The function below returns a
    # fixed size interval around the point predictions.
    def prediction_set_function(y_pred, scores_quantile):
        y_lo = y_pred - scores_quantile
        y_hi = y_pred + scores_quantile
        return y_lo, y_hi

    # The calibrator is instantiated by passing the two functions defined
    # above to the constructor.
    calibrator = BaseCalibrator(
        nonconf_score_func=nonconformity_function,
        pred_set_func=prediction_set_function,
    )

    # The nonconformity scores are computed by calling the `fit` method
    # on the calibration dataset.
    calibrator.fit(y_pred=y_pred_calib, y_true=y_calib)

    # The lower and upper bounds of the prediction interval are then returned
    # by the call to calibrate on the new data w.r.t a risk level alpha.
    y_pred_lower, y_pred_upper = calibrator.calibrate(
        y_pred=y_pred_test, alpha=alpha
    )

    assert y_pred_lower is not None
    assert y_pred_upper is not None
    assert not (True in np.isnan(y_pred_lower))
    assert not (True in np.isnan(y_pred_upper))


@pytest.mark.parametrize(
    "alpha, random_state",
    [[0.1, 42], [0.3, 42], [0.5, 42], [0.7, 42], [0.9, 42]],
)
def test_classification_calibrator(rand_class_data, alpha, random_state):
    # Generate data
    (y_pred_calib, y_calib, y_pred_test, y_test) = rand_class_data

    # The calibrator is instantiated
    calibrator = BaseCalibrator(
        nonconf_score_func=nonconformity_scores.raps_score_builder(
            lambd=0.1, k_reg=2
        ),
        pred_set_func=prediction_sets.raps_set_builder(lambd=0.1, k_reg=2),
    )

    # The nonconformity scores are computed by calling the `fit` method
    # on the calibration dataset.
    calibrator.fit(y_pred=y_pred_calib, y_true=y_calib)

    # The lower and upper bounds of the prediction interval are then returned
    # by the call to calibrate on the new data w.r.t a risk level alpha.
    set_pred = calibrator.calibrate(y_pred=y_pred_test, alpha=alpha)

    assert set_pred is not None


@pytest.mark.parametrize(
    "alpha, random_state",
    [[0.1, 42], [0.3, 42], [0.5, 42], [0.7, 42], [0.9, 42]],
)
def test_anomaly_detection_calibrator(
    rand_anomaly_detection_data, alpha, random_state
):
    # Generate data
    (z_train, z_test) = rand_anomaly_detection_data

    # Split data into proper fitting and calibration sets
    z_fit, z_calib = train_test_split(
        z_train, train_size=0.8, random_state=random_state
    )

    # Instantiate the LOF anomaly detection algorithm
    algorithm = LocalOutlierFactor(n_neighbors=35, novelty=True)

    # Fit the LOF on the proper fitting dataset
    algorithm.fit(X=z_fit)

    # The nonconformity scores are defined as the LOF scores (anomaly score).
    # By default, score_samples return the opposite of LOF scores.
    ncf = lambda X: -algorithm.score_samples(X)

    # The ScoreCalibrator is instantiated by passing the LOF score function
    # to the constructor
    cad = ScoreCalibrator(nonconf_score_func=ncf)

    # The LOF scores are computed by calling the `fit` method
    # on the calibration dataset
    cad.fit(z_calib)

    # We set the maximum false detection rate to 1%
    alpha = 0.01

    # The method `is_conformal` is called on the new data points
    # to test which are conformal (not anomalous) and which are not
    results = cad.is_conformal(z_test, alpha=alpha)
    not_anomalies = z_test[results]
    anomalies = z_test[np.invert(results)]

    assert anomalies is not None
    assert anomalies.shape == (117, 2)
    assert not_anomalies is not None
    assert not_anomalies.shape == (33, 2)


@pytest.mark.parametrize(
    "alpha, random_state",
    [
        [0.1, 42],
        [0.3, 42],
        [0.5, 42],
        [0.7, 42],
        [0.9, 42],
        [np.array([0.1, 0.3]), 42],
        [np.array([0.5, 0.7]), 42],
    ],
)
def test_multivariate_regression_calibrator(
    rand_multivariate_reg_data, alpha, random_state
):
    # Generate data
    (y_pred_calib, y_calib, y_pred_test, y_test) = rand_multivariate_reg_data

    # Nonconformity score function that takes as argument
    # the predicted values y_pred = model(X) and the true labels y_true. In
    # this example, we reimplement the mean absolute deviation that is
    # already defined in `deel.puncc.api.nonconformity_scores.mad`
    def nonconformity_function(y_pred, y_true):
        return np.abs(y_pred - y_true)

    # Prediction sets are computed based on point predictions and
    # the quantiles of nonconformity scores. The function below returns a
    # fixed size rectangle around the point predictions.
    def prediction_set_function(y_pred, scores_quantile):
        y_lo = y_pred - scores_quantile
        y_hi = y_pred + scores_quantile
        return y_lo, y_hi

    # The calibrator is instantiated by passing the two functions defined
    # above to the constructor.
    calibrator = BaseCalibrator(
        nonconf_score_func=nonconformity_function,
        pred_set_func=prediction_set_function,
    )

    # The nonconformity scores are computed by calling the `fit` method
    # on the calibration dataset.
    calibrator.fit(y_pred=y_pred_calib, y_true=y_calib)

    # The lower and upper bounds of the prediction interval are then returned
    # by the call to calibrate on the new data w.r.t a risk level alpha.
    y_pred_lower, y_pred_upper = calibrator.calibrate(
        y_pred=y_pred_test, alpha=alpha
    )

    assert y_pred_lower is not None
    assert y_pred_upper is not None
    assert not (True in np.isnan(y_pred_lower))
    assert not (True in np.isnan(y_pred_upper))


def test_base_calibrator_compute_quantile_and_barber_weights():
    calibrator = BaseCalibrator(
        nonconf_score_func=lambda y_pred, y_true: np.abs(y_pred - y_true),
        pred_set_func=lambda y_pred, scores_quantile: (y_pred - scores_quantile, y_pred + scores_quantile),
    )

    with pytest.raises(RuntimeError):
        calibrator.compute_quantile(alpha=0.2)

    y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_true = np.array([[0.5, 1.5], [2.5, 3.5]])
    calibrator.fit(y_pred=y_pred, y_true=y_true)
    q = calibrator.compute_quantile(alpha=np.array([0.4, 0.4]))

    assert q.shape == (2,)
    np.testing.assert_allclose(
        BaseCalibrator.barber_weights(np.array([1.0, 3.0])),
        np.array([0.2, 0.6, 0.2]),
    )


def test_classwise_calibrator_handles_missing_class_scores():
    residuals = np.array(
        [
            [0.1, np.nan, 0.4],
            [0.2, np.nan, 0.3],
            [0.3, np.nan, np.nan],
        ]
    )
    calibrator = ClasswiseCalibrator(
        nonconf_score_func=lambda y_pred, y_true: residuals,
        pred_set_func=lambda y_pred, scores_quantile: scores_quantile,
    )
    calibrator.fit(y_pred=np.zeros_like(residuals), y_true=np.zeros_like(residuals))

    quantiles = calibrator.compute_quantile(alpha=0.5)

    assert np.isinf(quantiles[1])
    assert quantiles.shape == (3,)


def test_classwise_calibrator_compute_quantile_raises_before_fit():
    calibrator = ClasswiseCalibrator(
        nonconf_score_func=lambda y_pred, y_true: y_pred,
        pred_set_func=lambda y_pred, scores_quantile: scores_quantile,
    )

    with pytest.raises(RuntimeError):
        calibrator.compute_quantile(alpha=0.5)


def test_leveraged_calibrator_fit_and_calibrate():
    calibrator = LeveragedCalibrator(
        nonconf_score_func=lambda X, y_pred, y_true, weight_func: np.abs(y_pred - y_true) * weight_func(X),
        pred_set_func=lambda y_pred, scores_quantile, weights: (y_pred - scores_quantile * weights, y_pred + scores_quantile * weights),
        weight_func=lambda x: x + 1.0,
        leverage_func=lambda x: x * 2.0,
    )

    X = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 4.0, 6.0])
    y_true = np.array([1.5, 3.5, 5.5])

    calibrator.fit(X=X, y_pred=y_pred, y_true=y_true)
    y_lo, y_hi = calibrator.calibrate(alpha=0.5, X=np.array([1.0, 3.0]), y_pred=np.array([10.0, 20.0]))

    assert y_lo.shape == (2,)
    assert y_hi.shape == (2,)
    assert np.all(y_lo < y_hi)


def test_score_calibrator_warning_and_weighted_conformity():
    calibrator = ScoreCalibrator(
        nonconf_score_func=lambda z: np.asarray(z, dtype=float),
        weight_func=lambda z: np.array([0.1, 0.2, 0.3, 0.4]),
    )

    with pytest.raises(RuntimeError):
        calibrator.is_conformal(np.array([1.0]), alpha=0.5)

    calibrator.fit(np.array([1.0, 2.0, 3.0, 4.0]))
    with pytest.warns(UserWarning):
        calibrator.set_nonconformity_scores(np.array([1.0, 2.0, 3.0, 4.0]))

    result = calibrator.is_conformal(np.array([1.0, 2.0, 5.0, 2.5]), alpha=0.5)
    np.testing.assert_array_equal(result, np.array([True, True, False, True]))


def test_cvplus_calibrator_error_paths():
    with pytest.raises(RuntimeError):
        CvPlusCalibrator(None)

    with pytest.raises(RuntimeError):
        CvPlusCalibrator({0: None})


class DummyCvPredictor:
    def __init__(self, output):
        self.output = output

    def predict(self, X):
        del X
        return self.output


class DummyCvCalibrator:
    def __init__(self, scores, norm_weights=None, pred_set_func=None):
        self._scores = scores
        self._norm_weights = norm_weights
        self.pred_set_func = pred_set_func or (
            lambda y_pred, nconf_scores: (y_pred - nconf_scores, y_pred + nconf_scores)
        )

    def get_nonconformity_scores(self):
        return self._scores

    def get_norm_weights(self):
        return self._norm_weights


class BadArrayLike:
    @property
    def shape(self):
        return (1,)

    def __len__(self):
        return 1

    def __array__(self, dtype=None):
        raise ValueError("cannot cast")


def test_score_calibrator_get_nonconformity_scores():
    calibrator = ScoreCalibrator(nonconf_score_func=lambda z: np.asarray(z, dtype=float))
    calibrator.fit(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_allclose(
        calibrator.get_nonconformity_scores(), np.array([1.0, 2.0, 3.0])
    )


def test_cvplus_calibrator_additional_error_paths():
    missing_scores = DummyCvCalibrator(scores=None)
    cv = CvPlusCalibrator({0: missing_scores})
    with pytest.raises(RuntimeError):
        cv.fit()

    good = DummyCvCalibrator(scores=np.array([1.0, 2.0]))
    cv_none_pred = CvPlusCalibrator({0: good})
    with pytest.raises(RuntimeError):
        cv_none_pred.calibrate(
            X=np.array([[0.0]]),
            kfold_predictors_dict={0: DummyCvPredictor(None)},
            alpha=0.5,
        )


def test_cvplus_calibrator_weighted_and_multivariate_paths():
    cal0 = DummyCvCalibrator(
        scores=np.array([1.0, 2.0]),
        norm_weights=np.array([0.2, 0.3]),
    )
    cal1 = DummyCvCalibrator(
        scores=np.array([0.5, 1.5]),
        norm_weights=np.array([0.1, 0.4]),
    )
    cv = CvPlusCalibrator({0: cal0, 1: cal1})

    with pytest.raises(ValueError):
        cv.calibrate(
            X=np.array([[0.0], [1.0]]),
            kfold_predictors_dict={
                0: DummyCvPredictor(np.array([[10.0, 20.0], [30.0, 40.0]])),
                1: DummyCvPredictor(np.array([[11.0, 21.0], [31.0, 41.0]])),
            },
            alpha=0.4,
        )


def test_cvplus_calibrator_noncastable_scores_raise_runtime_error():
    bad = DummyCvCalibrator(scores=BadArrayLike())
    cv = CvPlusCalibrator({0: bad})

    with pytest.raises(RuntimeError):
        cv.calibrate(
            X=np.array([[0.0]]),
            kfold_predictors_dict={0: DummyCvPredictor(np.array([10.0]))},
            alpha=0.5,
        )


def test_cvplus_calibrator_empty_predictor_dict_hits_sanity_check(monkeypatch):
    import deel.puncc.api.calibration as calibration_module

    cv = CvPlusCalibrator({})
    monkeypatch.setattr(calibration_module, "alpha_calib_check", lambda alpha, n: None)

    with pytest.raises(RuntimeError):
        cv.calibrate(X=np.array([[0.0]]), kfold_predictors_dict={}, alpha=0.5)


def test_cvplus_calibrator_weighted_concat_path_reaches_both_inf_appends():
    pred_func = lambda y_pred, nconf_scores: (
        np.array([y_pred.reshape(-1)[0] - np.asarray(nconf_scores).reshape(-1)[0]]),
        np.array([y_pred.reshape(-1)[0] + np.asarray(nconf_scores).reshape(-1)[0]]),
    )
    cal0 = DummyCvCalibrator(
        scores=np.array([[1.0], [2.0]]),
        norm_weights=np.array([0.25, 0.25]),
        pred_set_func=pred_func,
    )
    cv = CvPlusCalibrator({0: cal0})

    with pytest.raises(IndexError):
        cv.calibrate(
            X=np.array([[0.0]]),
            kfold_predictors_dict={0: DummyCvPredictor(np.array([10.0]))},
            alpha=0.5,
        )

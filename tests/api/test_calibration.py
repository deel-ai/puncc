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

    # We set the target false detection rate to 1%
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

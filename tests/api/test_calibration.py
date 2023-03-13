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

from deel.puncc.api import nonconformity_scores
from deel.puncc.api import prediction_sets
from deel.puncc.api.calibration import BaseCalibrator


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

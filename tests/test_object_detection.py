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

from deel.puncc.api import nonconformity_scores
from deel.puncc.api import prediction_sets
from deel.puncc.object_detection import SplitBoxWise


class DummyPredictor:
    def predict(self, x):
        return x


@pytest.mark.parametrize(
    ("method", "expected_score_func", "expected_pred_set_func"),
    [
        ("additive", nonconformity_scores.difference, prediction_sets.constant_bbox),
        (
            "multiplicative",
            nonconformity_scores.scaled_bbox_difference,
            prediction_sets.scaled_bbox,
        ),
    ],
)
def test_split_boxwise_selects_expected_functions(
    method, expected_score_func, expected_pred_set_func
):
    predictor = DummyPredictor()
    boxwise = SplitBoxWise(predictor, method=method, train=False, random_state=7)

    assert boxwise.predictor is predictor
    assert boxwise.calibrator.nonconf_score_func is expected_score_func
    assert boxwise.calibrator.pred_set_func is expected_pred_set_func
    assert boxwise.train is False
    assert boxwise.random_state == 7


def test_split_boxwise_rejects_invalid_method():
    with pytest.raises(ValueError, match="method must be 'additive' or 'multiplicative'"):
        SplitBoxWise(DummyPredictor(), method="invalid")


def test_split_boxwise_predict_delegates_to_conformal_predictor():
    class DummyConformalPredictor:
        def __init__(self):
            self.calls = []

        def predict(self, x_test, alpha, correction_func):
            self.calls.append((x_test, alpha, correction_func))
            return ("pred", "lower", "upper")

    boxwise = SplitBoxWise(DummyPredictor(), method="additive")
    dummy = DummyConformalPredictor()
    boxwise.conformal_predictor = dummy

    x_test = np.array([[1.0, 2.0, 3.0, 4.0]])
    correction_func = lambda alpha: np.full(4, alpha / 4)

    result = boxwise.predict(x_test, alpha=0.2, correction_func=correction_func)

    assert result == ("pred", "lower", "upper")
    assert dummy.calls == [(x_test, 0.2, correction_func)]

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

from deel.puncc.regression.split import SplitConformalRegression
from tests._utils import tensor, to_numpy


class MeanPredictor:
    """Simple fittable model for testing conformal predictor lifecycle."""

    def __init__(self):
        self.mean_ = None

    def fit(self, X, y):
        self.mean_ = float(to_numpy(y).mean())
        return self

    def predict(self, X):
        if self.mean_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return tensor(np.full(len(X), self.mean_, dtype=np.float32))


@pytest.fixture
def regression_data():
    X = np.arange(24, dtype=float).reshape(-1, 1)
    y = X[:, 0].copy()

    X, y = tensor(X, "float32"), tensor(y, "float32")
    return (
        X[:8],
        y[:8],
        X[8:20],
        y[8:20],
        X[20:],
        y[20:],
    )


def test_fit_calibrate_predict(regression_data):
    X_fit, y_fit, X_calib, y_calib, X_test, _ = regression_data

    predictor = SplitConformalRegression(
        model=MeanPredictor(),
    )

    assert predictor.fit(X_fit, y_fit) is predictor

    with pytest.raises(RuntimeError, match="not been calibrated"):
        _ = predictor.nc_scores

    assert predictor.calibrate(X_calib, y_calib) is predictor

    expected_prediction = np.full(
        len(X_calib),
        to_numpy(y_fit).mean(),
    )

    np.testing.assert_allclose(
        to_numpy(predictor.nc_scores),
        np.abs(expected_prediction - to_numpy(y_calib)),
    )

    result = predictor.predict(X_test, alpha=0.2)

    expected_test_prediction = np.full(
        len(X_test),
        to_numpy(y_fit).mean(),
    )

    np.testing.assert_allclose(
        to_numpy(result.prediction),
        expected_test_prediction,
    )

    interval = to_numpy(result.prediction_set)

    assert interval.shape == (len(X_test), 2)
    assert np.all(interval[:, 0] <= to_numpy(result.prediction))
    assert np.all(to_numpy(result.prediction) <= interval[:, 1])


def test_calibrate_pretrained_model(regression_data):
    X_fit, y_fit, X_calib, y_calib, X_test, _ = regression_data

    model = MeanPredictor().fit(X_fit, y_fit)

    predictor = SplitConformalRegression(model=model)
    predictor.calibrate(X_calib, y_calib)

    assert predictor.len_calibr == len(X_calib)

    result = predictor.predict(X_test, alpha=0.2)

    np.testing.assert_allclose(
        to_numpy(result.prediction),
        to_numpy(model.predict(X_test)),
    )
    assert result.prediction_set.shape == (len(X_test), 2)


def test_save_load_calibration_state(regression_data, tmp_path):
    X_fit, y_fit, X_calib, y_calib, X_test, _ = regression_data

    predictor = SplitConformalRegression(
        model=MeanPredictor(),
    )
    predictor.fit(X_fit, y_fit)
    predictor.calibrate(X_calib, y_calib)

    path = tmp_path / "conformal_state.pkl"
    predictor.save(path)

    assert path.is_file()

    restored_model = MeanPredictor().fit(X_fit, y_fit)

    loaded_predictor = SplitConformalRegression.load(
        path,
        model=restored_model,
    )

    assert loaded_predictor is not predictor

    np.testing.assert_allclose(
        to_numpy(loaded_predictor.nc_scores),
        to_numpy(predictor.nc_scores),
    )

    original_result = predictor.predict(X_test, alpha=0.2)
    loaded_result = loaded_predictor.predict(X_test, alpha=0.2)

    np.testing.assert_allclose(
        to_numpy(loaded_result.prediction),
        to_numpy(original_result.prediction),
    )
    np.testing.assert_allclose(
        to_numpy(loaded_result.prediction_set),
        to_numpy(original_result.prediction_set),
    )

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

from deel.puncc import ops
from deel.puncc.anomaly_detection import SplitCAD
from tests._utils import tensor, to_numpy


class DummyScorePredictor:
    def __init__(self):
        self.fit_calls = []

    def fit(self, X):
        self.fit_calls.append(to_numpy(X).copy())
        return self

    def predict(self, X):
        return tensor(X, "float32")[:, 0]


def test_splitcad_fit_calibrate_predict():
    X_fit = np.array([[1.0], [2.0], [3.0]])
    X_calib = np.arange(9, dtype=np.float32)[:, None]
    X_test = np.array(
        [[-1.0], [6.0], [7.5], [9.0]],
        dtype=np.float32,
    )

    X_fit = tensor(X_fit, "float32")
    X_calib = tensor(X_calib, "float32")
    X_test = tensor(X_test, "float32")
    model = DummyScorePredictor()
    cad = SplitCAD(model=model)

    with pytest.raises(RuntimeError, match="not been calibrated"):
        _ = cad.nc_scores

    assert cad.fit(X_fit) is cad
    assert len(model.fit_calls) == 1
    np.testing.assert_array_equal(model.fit_calls[0], to_numpy(X_fit))

    # Anomaly detection does not require calibration labels.
    assert cad.calibrate(X_calib) is cad
    assert len(model.fit_calls) == 1
    assert cad.len_calibr == len(X_calib)

    np.testing.assert_allclose(
        to_numpy(cad.nc_scores),
        np.arange(9),
    )

    result = cad.predict(X_test, alpha=0.3)

    np.testing.assert_allclose(
        to_numpy(result.prediction),
        [-1.0, 6.0, 7.5, 9.0],
    )

    # Calibration scores: [0, ..., 8].
    # With the finite-sample +inf correction, alpha=0.3
    # gives threshold 6. Anomalies have score > threshold.
    np.testing.assert_array_equal(
        to_numpy(result.prediction_set),
        [False, False, True, True],
    )

    # Lower alpha raises the threshold and detects fewer anomalies.
    conservative = cad.predict(X_test, alpha=0.1)

    np.testing.assert_array_equal(
        to_numpy(conservative.prediction_set),
        [False, False, False, True],
    )


def test_splitcad_with_pretrained_predictor():
    # A callable predictor can be calibrated without being fitted by PUNCC.
    predictor = lambda X: X[:, 0]

    X_calib = tensor(np.arange(9, dtype=np.float32)[:, None], "float32")
    cad = SplitCAD(model=predictor)

    with pytest.raises(NotImplementedError, match="fit method"):
        cad.fit(X_calib)

    cad.calibrate(X_calib)

    result = cad.predict(
        tensor([[7.0]], "float32"),
        alpha=0.3,
    )

    np.testing.assert_array_equal(
        to_numpy(result.prediction_set),
        [True],
    )


def test_splitcad_rejects_empty_calibration():
    cad = SplitCAD(model=DummyScorePredictor())

    with pytest.raises(ValueError, match="must not be empty"):
        cad.calibrate(tensor(np.empty((0, 1), dtype=np.float32)))


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_splitcad_rejects_invalid_alpha(alpha):
    cad = SplitCAD(model=DummyScorePredictor())
    cad.calibrate(tensor(np.arange(9, dtype=np.float32)[:, None]))

    with pytest.raises(ValueError, match="strictly between"):
        cad.predict(
            tensor([[4.0]], "float32"),
            alpha=alpha,
        )


def test_splitcad_with_local_outlier_factor():
    neighbors = pytest.importorskip("sklearn.neighbors")

    class LOFScorePredictor:
        def __init__(self):
            self.model = neighbors.LocalOutlierFactor(
                n_neighbors=20,
                novelty=True,
            )

        def fit(self, X):
            self.model.fit(to_numpy(X))
            return self

        def predict(self, X):
            return tensor(-self.model.score_samples(to_numpy(X)), "float32")

    rng = np.random.default_rng(42)

    X_fit = rng.normal(scale=0.3, size=(150, 2)).astype(np.float32)

    X_calib = rng.normal(scale=0.3, size=(60, 2)).astype(np.float32)

    X_normal = rng.normal(scale=0.3, size=(24, 2)).astype(np.float32)

    X_anomaly = np.array(
        [
            [5.0, 5.0],
            [-5.0, -5.0],
            [5.0, -5.0],
            [-5.0, 5.0],
        ],
        dtype=np.float32,
    )

    X_test = np.concatenate([X_normal, X_anomaly], axis=0)

    X_fit = tensor(X_fit, "float32")
    X_calib = tensor(X_calib, "float32")
    X_test = tensor(X_test, "float32")
    cad = SplitCAD(model=LOFScorePredictor())
    cad.fit(X_fit)
    cad.calibrate(X_calib)

    result = cad.predict(X_test, alpha=0.1)
    anomalies = to_numpy(result.prediction_set)

    assert anomalies.shape == (len(X_test),)
    assert anomalies.dtype == np.bool_

    # The four distant observations must be detected.
    assert np.all(anomalies[-4:])

    # Normal observations should not all be classified as anomalies.
    assert not np.all(anomalies[:24])

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

from deel.puncc.anomaly_detection import SplitCAD
from deel.puncc.api.prediction import BasePredictor

RESULTS = {
    "split_cad": {"normal_shape": (33, 2), "anomaly_shape": (117, 2)},
}


@pytest.mark.parametrize(
    "alpha, random_state",
    [[0.1, 42], [0.3, 42], [0.5, 42], [0.7, 42], [0.9, 42]],
)
def test_anomaly_detection(rand_anomaly_detection_data, alpha, random_state):
    # Generate data
    (z_train, z_test) = rand_anomaly_detection_data

    # Split data into proper fitting and calibration sets
    z_fit, z_calib = train_test_split(
        z_train, train_size=0.8, random_state=random_state
    )

    class ADPredictor(BasePredictor):
        def predict(self, X):
            return -self.model.score_samples(X)

    # Instantiate the LOF anomaly detection algorithm
    # and wrap it in a predictor
    # The nonconformity scores are defined as the LOF scores (anomaly score).
    # By default, score_samples return the opposite of LOF scores.
    lof_predictor = ADPredictor(
        LocalOutlierFactor(n_neighbors=35, novelty=True)
    )

    # Instantiate CAD on top of LOF predictor
    lof_cad = SplitCAD(lof_predictor, train=True, random_state=random_state)

    # Fit the LOF on the proper fitting dataset and
    # calibrate it using calibration dataset
    lof_cad.fit(z_fit=z_fit, z_calib=z_calib)

    # We set the target false detection rate to 1%
    alpha = 0.01

    # The method `predict` is called on the new data points
    # to test which are anomalous and which are not
    results = lof_cad.predict(z_test, alpha=alpha)
    anomalies = z_test[results]
    not_anomalies = z_test[np.invert(results)]

    assert anomalies is not None
    np.testing.assert_allclose(
        anomalies.shape, RESULTS["split_cad"]["anomaly_shape"]
    )
    assert not_anomalies is not None
    np.testing.assert_allclose(
        not_anomalies.shape, RESULTS["split_cad"]["normal_shape"]
    )


class DummyAnomalyPredictor:
    def __init__(self, trained=False):
        self.is_trained = trained
        self.fit_calls = []

    def fit(self, X, **kwargs):
        self.fit_calls.append((np.asarray(X), kwargs))
        self.is_trained = True

    def predict(self, X):
        return np.sum(np.asarray(X), axis=1)


def test_splitcad_fit_with_random_split_trains_predictor():
    z = np.arange(20, dtype=float).reshape(10, 2)
    predictor = DummyAnomalyPredictor()
    cad = SplitCAD(predictor, train=True, random_state=0)

    cad.fit(z=z, fit_ratio=0.6)
    preds = cad.predict(z, alpha=0.5)

    assert predictor.is_trained is True
    assert len(predictor.fit_calls) == 1
    assert preds.shape == (10,)
    assert preds.dtype == np.bool_


def test_splitcad_fit_with_pretrained_predictor_skips_training():
    z_calib = np.array([[0.0, 0.5], [0.5, 0.0], [0.1, 0.2]])
    predictor = DummyAnomalyPredictor(trained=True)
    cad = SplitCAD(predictor, train=False)

    cad.fit(z_calib=z_calib)

    assert predictor.fit_calls == []


def test_splitcad_fit_raises_with_untrained_predictor_when_train_false():
    predictor = DummyAnomalyPredictor(trained=False)
    cad = SplitCAD(predictor, train=False)

    with pytest.raises(RuntimeError):
        cad.fit(
            z_fit=np.array([[0.0, 0.0], [1.0, 1.0]]),
            z_calib=np.array([[0.0, 0.0], [1.0, 1.0]]),
        )


def test_splitcad_fit_raises_without_data():
    cad = SplitCAD(DummyAnomalyPredictor(), train=True)

    with pytest.raises(RuntimeError):
        cad.fit()


def test_splitcad_predict_raises_when_forced_to_unfit_state():
    cad = SplitCAD(DummyAnomalyPredictor(), train=True)
    cad._SplitCAD__is_fit = None

    with pytest.raises(RuntimeError):
        cad.predict(np.array([[0.0, 1.0]]), alpha=0.1)

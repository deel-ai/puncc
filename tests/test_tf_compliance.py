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

import os

import numpy as np
import pytest

from deel.puncc.backend.keras import ops
from deel.puncc.core.cross_conformal import CVPlusRegressor
from deel.puncc.regression.split import SplitConformalRegression
from tests._utils import to_numpy


@pytest.fixture(scope="module")
def keras_tf():
    if os.environ.get("KERAS_BACKEND") != "tensorflow":
        pytest.skip("Requires KERAS_BACKEND=tensorflow.")

    pytest.importorskip("tensorflow")

    import keras

    assert keras.backend.backend() == "tensorflow"
    return keras


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(96, 2)).astype(np.float32)
    y = (2.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.3, size=96)).astype(np.float32)

    return X, y


def make_keras_model(keras):
    # Reshape(()) produces scalar predictions with shape (n_samples,).
    # This is the shape required by the scalar regression methods.
    return keras.Sequential(
        [
            keras.layers.Input(shape=(2,)),
            keras.layers.Dense(8, activation="tanh"),
            keras.layers.Dense(1),
            keras.layers.Reshape(()),
        ]
    )


def test_tensorflow_split_conformal_regression(keras_tf, regression_data):
    X, y = regression_data

    X_fit, y_fit = X[:60], y[:60]
    X_calib, y_calib = X[60:84], y[60:84]
    X_test = X[84:]

    model = make_keras_model(keras_tf)
    model.compile(optimizer="adam", loss="mse")

    cp = SplitConformalRegression(model=model)

    cp.fit(
        X_fit,
        y_fit,
        epochs=3,
        batch_size=16,
        verbose=0,
    )
    cp.calibrate(X_calib, y_calib)

    assert cp.len_calibr == len(X_calib)

    expected_scores = np.abs(to_numpy(model(X_calib)) - y_calib)

    np.testing.assert_allclose(
        to_numpy(cp.nc_scores),
        expected_scores,
        rtol=1e-5,
        atol=1e-5,
    )

    result = cp.predict(X_test, alpha=0.2)

    prediction = to_numpy(result.prediction)
    intervals = to_numpy(result.prediction_set)

    assert prediction.shape == (len(X_test),)
    assert intervals.shape == (len(X_test), 2)
    assert np.all(np.isfinite(intervals))
    assert np.all(intervals[:, 0] <= intervals[:, 1])

    # Constant-width split conformal intervals are centered on
    # the base model prediction.
    np.testing.assert_allclose(
        (intervals[:, 0] + intervals[:, 1]) / 2,
        prediction,
        rtol=1e-5,
        atol=1e-5,
    )


def test_tensorflow_cvplus_clones_keras_models(keras_tf, regression_data):
    X, y = regression_data
    X_train, y_train = X[:84], y[:84]
    X_test = X[84:]

    base_model = make_keras_model(keras_tf)

    def fit_keras_model(model, X_fit, y_fit):
        # Cloned Keras models are not automatically compiled.
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            X_fit,
            y_fit,
            epochs=2,
            batch_size=16,
            verbose=0,
        )
        return model

    cp = CVPlusRegressor(
        model=base_model,
        K=3,
        random_state=42,
        fit_function=fit_keras_model,
    )

    cp.fit(X_train, y_train)

    assert cp.len_calibr == len(X_train)
    assert len(cp._conformal_predictors) == 3

    # Every fold must own an independent cloned Keras model.
    fold_models = [fold.model for fold in cp._conformal_predictors]

    assert len({id(model) for model in fold_models}) == 3
    assert all(model is not base_model for model in fold_models)

    result = cp.predict(X_test, alpha=0.2)

    prediction = to_numpy(result.prediction)
    intervals = to_numpy(result.prediction_set)

    assert prediction.shape == (len(X_test),)
    assert intervals.shape == (len(X_test), 2)

    assert np.all(np.isfinite(prediction))
    assert np.all(np.isfinite(intervals))
    assert np.all(intervals[:, 0] <= intervals[:, 1])

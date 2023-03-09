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
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical

from deel.puncc import metrics
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.classification import APS
from deel.puncc.classification import RAPS


RESULTS = {
    "aps": {"cov": 0.92, "size": 1.86},
    "raps": {"cov": 10000, "size": 10000},  # Placeholder
}


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_aps(mnist_data, alpha, random_state):

    tf.keras.utils.set_random_seed(random_state)

    # Get data
    (X_train, X_test, y_train, y_test, y_train_cat, y_test_cat) = mnist_data

    # Split fit and calib datasets
    X_fit, X_calib = X_train[:50000], X_train[50000:]
    y_fit, y_calib = y_train[:50000], y_train[50000:]
    y_fit_cat, y_calib_cat = y_train_cat[:50000], y_train_cat[50000:]

    # One hot encoding of classes
    y_fit_cat = to_categorical(y_fit)
    y_calib_cat = to_categorical(y_calib)
    y_test_cat = to_categorical(y_test)

    # Classification model
    nn_model = models.Sequential()
    nn_model.add(layers.Dense(4, activation="relu", input_shape=(28 * 28,)))
    nn_model.add(layers.Dense(10, activation="softmax"))
    compile_kwargs = {
        "optimizer": "rmsprop",
        "loss": "categorical_crossentropy",
        "metrics": [],
    }
    fit_kwargs = {"epochs": 5, "batch_size": 16, "verbose": 1}
    # Predictor wrapper
    class_predictor = BasePredictor(nn_model, is_trained=False, **compile_kwargs)

    # APS
    aps_cp = APS(class_predictor)
    aps_cp.fit(X_fit, y_fit_cat, X_calib, y_calib, **fit_kwargs)
    y_pred, set_pred = aps_cp.predict(X_test, alpha=alpha)
    assert y_pred is not None

    # Compute marginal coverage
    coverage = metrics.classification_mean_coverage(y_test, set_pred)
    width = metrics.classification_mean_size(set_pred)
    res = {"cov": np.round(coverage, 2), "size": np.round(width, 2)}
    assert RESULTS["aps"] == res


# @pytest.mark.parametrize(
#     "alpha, random_state, lambd, k_reg",
#     [(0.1, 42, 1, 2)],
# )
# def test_raps(mnist_data, alpha, random_state, lambd, k_reg):

#     tf.keras.utils.set_random_seed(random_state)

#     # Get data
#     (X_train, X_test, y_train, y_test, y_train_cat, y_test_cat) = mnist_data

#     # Split fit and calib datasets
#     X_fit, X_calib = X_train[:50000], X_train[50000:]
#     y_fit, y_calib = y_train[:50000], y_train[50000:]
#     y_fit_cat, y_calib_cat = y_train_cat[:50000], y_train_cat[50000:]

#     # One hot encoding of classes
#     y_fit_cat = to_categorical(y_fit)
#     y_calib_cat = to_categorical(y_calib)
#     y_test_cat = to_categorical(y_test)

#     # Classification model
#     nn_model = models.Sequential()
#     nn_model.add(layers.Dense(4, activation="relu", input_shape=(28 * 28,)))
#     nn_model.add(layers.Dense(10, activation="softmax"))
#     compile_kwargs = {
#         "optimizer": "rmsprop",
#         "loss": "categorical_crossentropy",
#         "metrics": [],
#     }
#     fit_kwargs = {"epochs": 5, "batch_size": 128, "verbose": 1}
#     # Predictor wrapper
#     class_predictor = BasePredictor(nn_model, is_trained=False, **compile_kwargs)

#     # RAPS
#     raps_cp = RAPS(class_predictor, k_reg=k_reg, lambd=lambd)
#     raps_cp.fit(X_fit, y_fit_cat, X_calib, y_calib, **fit_kwargs)
#     y_pred, set_pred = raps_cp.predict(X_test, alpha=alpha)
#     assert y_pred is not None

#     # Compute marginal coverage
#     coverage = metrics.classification_mean_coverage(y_test, set_pred)
#     width = metrics.classification_mean_size(set_pred)
#     res = {"cov": np.round(coverage, 2), "size": np.round(width, 2)}
#     assert RESULTS["raps"] == res

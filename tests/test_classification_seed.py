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
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from deel.puncc import ops
from deel.puncc.classification import APS, RAPS, LAC, ClassConditionalLAC
from deel.puncc.metrics import (
    classification_mean_coverage,
    classification_mean_size,
)
from tests._utils import tensor, to_numpy


class ProbabilityModel:
    """Adapt a classifier to PUNCC's predict-probabilities interface."""

    def __init__(self):
        self.estimator = LogisticRegression(max_iter=500)

    def fit(self, X, y):
        self.estimator.fit(to_numpy(X), to_numpy(y))
        return self

    def predict(self, X):
        return tensor(self.estimator.predict_proba(to_numpy(X)), "float32")


@pytest.fixture(scope="module")
def classification_data():
    X, y = make_classification(
        n_samples=360,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42,
    )
    X = X.astype(np.float32)

    X_fit, X_remaining, y_fit, y_remaining = train_test_split(
        X,
        y,
        train_size=0.5,
        stratify=y,
        random_state=42,
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.5,
        stratify=y_remaining,
        random_state=42,
    )

    return (
        (tensor(X_fit, "float32"), tensor(y_fit, "int32")),
        (tensor(X_calib, "float32"), tensor(y_calib, "int32")),
        (tensor(X_test, "float32"), tensor(y_test, "int32")),
    )


def conformal_quantile(scores, alpha):
    """Independent finite-sample quantile, with an infinite extra score."""
    values = np.sort(np.append(scores, np.inf))
    rank = int(np.ceil(len(values) * (1 - alpha))) - 1
    return values[rank]


def check_prediction_sets(result, y_test, n_classes=3):
    probabilities = to_numpy(result.prediction)
    sets = [to_numpy(s) for s in result.prediction_set]

    assert probabilities.shape == (len(y_test), n_classes)
    np.testing.assert_allclose(
        probabilities.sum(axis=-1),
        np.ones(len(y_test)),
        atol=1e-6,
    )

    assert len(sets) == len(y_test)
    for labels in sets:
        assert labels.ndim == 1
        assert np.all((labels >= 0) & (labels < n_classes))
        assert len(np.unique(labels)) == len(labels)

    coverage = classification_mean_coverage(to_numpy(y_test), sets)
    mean_size = classification_mean_size(sets)

    assert coverage >= 0.6
    assert 0 <= mean_size <= n_classes

    return probabilities, sets


def test_lac(classification_data):
    (X_fit, y_fit), (X_calib, y_calib), (X_test, y_test) = classification_data

    cp = LAC(model=ProbabilityModel())
    cp.fit(X_fit, y_fit)
    cp.calibrate(X_calib, y_calib)

    calib_prob = to_numpy(cp.model(X_calib))
    expected_scores = 1 - calib_prob[np.arange(len(y_calib)), to_numpy(y_calib)]

    np.testing.assert_allclose(
        to_numpy(cp.nc_scores),
        expected_scores,
        atol=1e-6,
    )

    alpha = 0.2
    probabilities, sets = check_prediction_sets(
        cp.predict(X_test, alpha=alpha),
        y_test,
    )

    q = conformal_quantile(expected_scores, alpha)

    for prob, labels in zip(probabilities, sets):
        np.testing.assert_array_equal(
            labels,
            np.flatnonzero(prob >= 1 - q),
        )

    # Lower miscoverage must produce supersets.
    smaller_alpha_sets = cp.predict(X_test, alpha=0.05).prediction_set

    for original, enlarged in zip(sets, smaller_alpha_sets):
        assert set(original.tolist()).issubset(set(to_numpy(enlarged).tolist()))


def test_class_conditional_lac(classification_data):
    (X_fit, y_fit), (X_calib, y_calib), (X_test, y_test) = classification_data

    cp = ClassConditionalLAC(model=ProbabilityModel())
    cp.fit(X_fit, y_fit)
    cp.calibrate(X_calib, y_calib)

    assert set(cp.classwise_calibration_contexts) == {0, 1, 2}

    alpha = 0.2
    probabilities, sets = check_prediction_sets(
        cp.predict(X_test, alpha=alpha),
        y_test,
    )

    calib_prob = to_numpy(cp.model(X_calib))

    # Each class gets its own conformity quantile.
    quantiles = []
    for label in range(3):
        mask = to_numpy(y_calib) == label
        class_scores = 1 - calib_prob[mask, label]
        quantiles.append(conformal_quantile(class_scores, alpha))

    quantiles = np.asarray(quantiles)

    for prob, labels in zip(probabilities, sets):
        np.testing.assert_array_equal(
            labels,
            np.flatnonzero(prob >= 1 - quantiles),
        )


def test_class_conditional_lac_includes_missing_classes():
    # Class 2 is never observed in calibration.
    # Its quantile is +inf, so it must remain in prediction sets.
    def fixed_probabilities(X):
        return tensor(
            np.tile(
                np.array([[0.55, 0.35, 0.10]], dtype=np.float32),
                (len(X), 1),
            )
        )

    cp = ClassConditionalLAC(model=fixed_probabilities)

    X_calib = tensor(np.arange(30, dtype=np.float32)[:, None])
    y_calib = tensor(np.tile([0, 1], 15), "int32")

    cp.calibrate(X_calib, y_calib)

    result = cp.predict(
        tensor([[0.0], [1.0]], "float32"),
        alpha=0.2,
    )

    assert all(2 in to_numpy(labels) for labels in result.prediction_set)


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda model: APS(model),
            id="aps-randomized",
        ),
        pytest.param(
            lambda model: RAPS(model, lambd=0, k_reg=1, rand=False),
            id="aps-nonrandomized",
        ),
        pytest.param(
            lambda model: RAPS(model, lambd=0.05, k_reg=1, rand=True),
            id="raps-randomized",
        ),
        pytest.param(
            lambda model: RAPS(model, lambd=0.05, k_reg=1, rand=False),
            id="raps-nonrandomized",
        ),
    ],
)
def test_aps_and_raps(classification_data, factory):
    (X_fit, y_fit), (X_calib, y_calib), (X_test, y_test) = classification_data

    cp = factory(ProbabilityModel())
    cp.fit(X_fit, y_fit)
    cp.calibrate(X_calib, y_calib)

    scores = to_numpy(cp.nc_scores)

    assert scores.shape == (len(X_calib),)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)

    # Cumulative probability is at most 1, with a maximum
    # RAPS regularization penalty of 0.05 * (3 - 1).
    assert np.all(scores <= 1.1 + 1e-6)

    check_prediction_sets(
        cp.predict(X_test, alpha=0.2),
        y_test,
    )


def test_nonrandomized_aps_scores(classification_data):
    (X_fit, y_fit), (X_calib, y_calib), _ = classification_data

    # APS without randomization is RAPS with lambda=0.
    cp = RAPS(
        model=ProbabilityModel(),
        lambd=0,
        k_reg=1,
        rand=False,
    )
    cp.fit(X_fit, y_fit)
    cp.calibrate(X_calib, y_calib)

    probabilities = to_numpy(cp.model(X_calib))
    true_probabilities = probabilities[np.arange(len(y_calib)), to_numpy(y_calib)]

    # Sum probability mass of classes ranked before the true class,
    # then add the true class probability.
    higher_ranked = probabilities > true_probabilities[:, None]
    expected = (
        np.sum(
            np.where(higher_ranked, probabilities, 0),
            axis=1,
        )
        + true_probabilities
    )

    np.testing.assert_allclose(
        to_numpy(cp.nc_scores),
        expected,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambd": -0.1},
        {"k_reg": -1},
    ],
)
def test_raps_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        RAPS(model=ProbabilityModel(), **kwargs)

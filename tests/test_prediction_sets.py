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

from deel.puncc.prediction_sets import (
    constant_bbox,
    cqr_interval,
    lac_set,
    raps_set,
    scaled_bbox,
    scaled_interval,
)
from tests._utils import tensor, to_numpy


@pytest.mark.parametrize("n_samples", [1, 10])
def test_scaled_bbox(n_samples):
    y_pred = np.repeat(
        np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        n_samples,
        axis=0,
    )
    y_pred = tensor(y_pred, "float32")
    quantile = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    quantile = tensor(quantile, "float32")

    expected = np.repeat(
        np.array(
            [
                [
                    [1.2, 0.8],
                    [2.4, 1.6],
                    [2.4, 3.6],
                    [3.2, 4.8],
                ]
            ],
            dtype=np.float32,
        ),
        n_samples,
        axis=0,
    )

    result = scaled_bbox()(y_pred, quantile)

    assert result.shape == (n_samples, 4, 2)
    np.testing.assert_allclose(to_numpy(result), expected)


def test_scaled_interval_mean_dispersion():
    # Each prediction contains [mean, dispersion].
    y_pred = np.array(
        [[10.0, 2.0], [20.0, 4.0]],
        dtype=np.float32,
    )
    y_pred = tensor(y_pred, "float32")

    result = scaled_interval()(y_pred, quantile=3.0)

    expected = np.array(
        [[4.0, 16.0], [8.0, 32.0]],
        dtype=np.float32,
    )

    np.testing.assert_allclose(to_numpy(result), expected)


def test_scaled_interval_negative_dispersion():
    y_pred = np.array(
        [[10.0, -1.0], [20.0, 2.0]],
        dtype=np.float32,
    )
    y_pred = tensor(y_pred, "float32")

    result = scaled_interval()(y_pred, quantile=3.0)

    assert result.shape == (2, 2)
    assert np.isneginf(to_numpy(result)[0, 0])
    assert np.isposinf(to_numpy(result)[0, 1])

    np.testing.assert_allclose(to_numpy(result)[1], [14.0, 26.0])


def test_constant_bbox():
    y_pred = np.array(
        [[1.0, 2.0, 3.0, 4.0]],
        dtype=np.float32,
    )
    y_pred = tensor(y_pred, "float32")
    quantile = np.array(
        [0.5, 1.0, 1.5, 2.0],
        dtype=np.float32,
    )
    quantile = tensor(quantile, "float32")

    result = constant_bbox()(y_pred, quantile)

    expected = np.array(
        [
            [
                [1.5, 0.5],
                [3.0, 1.0],
                [1.5, 4.5],
                [2.0, 6.0],
            ]
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(to_numpy(result), expected)


def test_cqr_interval():
    # Each prediction contains [lower_quantile, upper_quantile].
    y_pred = np.array(
        [[1.0, 4.0], [2.0, 6.0]],
        dtype=np.float32,
    )
    y_pred = tensor(y_pred, "float32")

    result = cqr_interval()(y_pred, quantile=0.5)

    np.testing.assert_allclose(
        to_numpy(result),
        [[0.5, 4.5], [1.5, 6.5]],
    )


def test_lac_set_with_classwise_quantiles():
    y_pred = np.array(
        [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]],
        dtype=np.float32,
    )
    y_pred = tensor(y_pred, "float32")

    # Class-specific quantiles, broadcast over samples.
    quantiles = np.array(
        [[0.2, 0.4, 0.8]],
        dtype=np.float32,
    )
    quantiles = tensor(quantiles, "float32")

    prediction_sets = lac_set()(y_pred, quantiles)

    assert len(prediction_sets) == 2
    np.testing.assert_array_equal(to_numpy(prediction_sets[0]), [])
    np.testing.assert_array_equal(to_numpy(prediction_sets[1]), [2])


def test_raps_set():
    y_pred = np.array(
        [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]],
        dtype=np.float32,
    )
    y_pred = tensor(y_pred, "float32")

    prediction_sets = raps_set(
        lambd=0.1,
        k_reg=1,
        rand=False,
    )(y_pred, quantile=0.65)

    assert len(prediction_sets) == 2
    np.testing.assert_array_equal(to_numpy(prediction_sets[0]), [0])
    np.testing.assert_array_equal(to_numpy(prediction_sets[1]), [1, 2])


@pytest.mark.parametrize(
    "parameters",
    [{"lambd": -1.0}, {"k_reg": -1}],
)
def test_raps_set_rejects_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        raps_set(**parameters)

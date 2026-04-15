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

from deel.puncc.api.prediction_sets import classwise_lac_set
from deel.puncc.api.prediction_sets import constant_bbox
from deel.puncc.api.prediction_sets import cqr_interval
from deel.puncc.api.prediction_sets import raps_set
from deel.puncc.api.prediction_sets import raps_set_builder
from deel.puncc.api.prediction_sets import scaled_bbox
from deel.puncc.api.prediction_sets import scaled_interval


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.asarray(x)


def _make_backend_data(backend_name, bbox_pred, scores_quantile):
    if backend_name == "numpy":
        return bbox_pred, scores_quantile

    if backend_name == "python":
        return bbox_pred.tolist(), scores_quantile.tolist()

    if backend_name == "pandas":
        pd = pytest.importorskip("pandas")
        return pd.DataFrame(bbox_pred), pd.Series(scores_quantile)

    if backend_name == "torch":
        torch = pytest.importorskip("torch")
        return torch.tensor(bbox_pred), torch.tensor(scores_quantile)

    if backend_name == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        return tf.convert_to_tensor(bbox_pred), tf.convert_to_tensor(scores_quantile)

    if backend_name == "jax":
        jnp = pytest.importorskip("jax.numpy")
        return jnp.asarray(bbox_pred), jnp.asarray(scores_quantile)

    raise ValueError(f"Unsupported backend test case: {backend_name}")


@pytest.mark.parametrize(
    "backend_name",
    ["numpy", "python", "pandas", "torch", "tensorflow", "jax"],
)
def test_scaled_bbox_all_backends(backend_name):
    n = 10
    bbox_pred = np.array([[1, 2, 3, 4]], dtype=float)
    scores_quantile = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)

    expected_lo = np.array([[1.2, 2.4, 2.4, 3.2]], dtype=float)
    expected_hi = np.array([[0.8, 1.6, 3.6, 4.8]], dtype=float)

    bbox_pred_n = np.repeat(bbox_pred, n, axis=0)
    expected_lo_n = np.repeat(expected_lo, n, axis=0)
    expected_hi_n = np.repeat(expected_hi, n, axis=0)

    y_pred, q = _make_backend_data(backend_name, bbox_pred, scores_quantile)
    y_lo, y_hi = scaled_bbox(y_pred, q)

    np.testing.assert_allclose(_to_numpy(y_lo), expected_lo)
    np.testing.assert_allclose(_to_numpy(y_hi), expected_hi)

    y_pred_n, q_n = _make_backend_data(backend_name, bbox_pred_n, scores_quantile)
    y_lo_n, y_hi_n = scaled_bbox(y_pred_n, q_n)

    np.testing.assert_allclose(_to_numpy(y_lo_n), expected_lo_n)
    np.testing.assert_allclose(_to_numpy(y_hi_n), expected_hi_n)


def test_scaled_bbox_invalid_shapes():
    scores_quantile = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)

    # Y_pred is not 2D
    with pytest.raises(RuntimeError):
        scaled_bbox([1, 2, 3, 4], scores_quantile)

    # Y_pred does not contain 4 bbox coordinates
    y_pred = np.array([[0, 0, 2], [1, 3, 3]], dtype=float)
    with pytest.raises(RuntimeError):
        scaled_bbox(y_pred, scores_quantile)


def test_scaled_interval_with_weights_meanvar_predictions():
    # Predictions contain [mu, sigma]
    y_pred = np.array([[10.0, 2.0], [20.0, 4.0]])
    q = 3.0
    # Use 1D per-sample weights (the expected shape used in LWCP calibrate path)
    weights = np.array([0.5, 2.0])
    eps = 1e-12

    y_lo, y_hi = scaled_interval(y_pred, q, weights=weights, eps=eps)

    expected_lo = np.array(
        [10.0 - 3.0 * (2.0 + eps) * 0.5, 20.0 - 3.0 * (4.0 + eps) * 2.0]
    )
    expected_hi = np.array(
        [10.0 + 3.0 * (2.0 + eps) * 0.5, 20.0 + 3.0 * (4.0 + eps) * 2.0]
    )

    np.testing.assert_allclose(y_lo, expected_lo)
    np.testing.assert_allclose(y_hi, expected_hi)


def test_scaled_interval_with_weights_point_predictions():
    # Predictions contain only point estimates: sigma defaults to 1.
    y_pred = np.array([10.0, 20.0])
    q = 2.0
    weights = np.array([0.5, 2.0])
    eps = 1e-12

    y_lo, y_hi = scaled_interval(y_pred, q, weights=weights, eps=eps)

    expected_lo = np.array(
        [
            10.0 - 2.0 * (1.0 + eps) * 0.5,
            20.0 - 2.0 * (1.0 + eps) * 2.0,
        ]
    )
    expected_hi = np.array(
        [
            10.0 + 2.0 * (1.0 + eps) * 0.5,
            20.0 + 2.0 * (1.0 + eps) * 2.0,
        ]
    )

    np.testing.assert_allclose(y_lo, expected_lo)
    np.testing.assert_allclose(y_hi, expected_hi)


def test_scaled_interval_negative_sigma_returns_infinite_bounds():
    y_pred = np.array([[10.0, -1.0], [20.0, 2.0]])

    y_lo, y_hi = scaled_interval(y_pred, 3.0)

    assert np.isneginf(y_lo[0])
    assert np.isposinf(y_hi[0])
    assert np.isfinite(y_lo[1])
    assert np.isfinite(y_hi[1])


def test_constant_bbox_and_cqr_interval_guards():
    bbox = np.array([[1.0, 2.0, 3.0, 4.0]])
    q_bbox = np.array([0.5, 1.0, 1.5, 2.0])
    y_lo, y_hi = constant_bbox(bbox, q_bbox)

    np.testing.assert_allclose(y_lo, np.array([[1.5, 3.0, 1.5, 2.0]]))
    np.testing.assert_allclose(y_hi, np.array([[0.5, 1.0, 4.5, 6.0]]))

    with pytest.raises(RuntimeError):
        cqr_interval(np.array([1.0, 2.0, 3.0]), 0.5)

    with pytest.raises(RuntimeError):
        constant_bbox(np.array([[1.0, 2.0, 3.0]]), q_bbox)


def test_classwise_lac_and_raps_variants():
    y_pred = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])
    classwise_sets = classwise_lac_set(y_pred, np.array([0.2, 0.4, 0.8]))[0]
    assert classwise_sets == [[], [2]]

    raps_sets = raps_set(y_pred, scores_quantile=0.65, lambd=0.1, k_reg=1, rand=False)[0]
    assert all(isinstance(pred_set, list) for pred_set in raps_sets)
    assert len(raps_sets) == len(y_pred)

    with pytest.raises(ValueError):
        raps_set_builder(lambd=-1)
    with pytest.raises(ValueError):
        raps_set_builder(k_reg=-1)

    built = raps_set_builder(lambd=0.1, k_reg=1, rand=False)
    assert built(y_pred, 0.65)[0] == raps_sets

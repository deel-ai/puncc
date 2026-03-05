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

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

from deel.puncc.api.backend import concat_columns
from deel.puncc.api.backend import get_backend
from deel.puncc.api.backend import infer_backend
from deel.puncc.api.backend import shape2
from deel.puncc.api.backend import split_columns


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


def _make_backend_array(name, arr):
    if name == "numpy":
        return np.asarray(arr)

    if name == "pandas":
        pd = pytest.importorskip("pandas")
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return pd.Series(arr)
        return pd.DataFrame(arr)

    if name == "torch":
        torch = pytest.importorskip("torch")
        return torch.tensor(np.asarray(arr))

    if name == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        return tf.convert_to_tensor(np.asarray(arr))

    if name == "jax":
        jnp = pytest.importorskip("jax.numpy")
        return jnp.asarray(np.asarray(arr))

    raise ValueError(f"Unsupported backend name: {name}")


def test_infer_backend_numpy_default():
    assert infer_backend([1, 2, 3]) == "numpy"
    assert infer_backend(np.array([1, 2, 3])) == "numpy"
    assert infer_backend(None) == "numpy"


def test_infer_backend_pandas_and_mixed_error():
    pd = pytest.importorskip("pandas")
    assert infer_backend(pd.Series([1, 2, 3])) == "pandas"

    with pytest.raises(TypeError):
        infer_backend(pd.Series([1, 2, 3]), np.array([1, 2, 3]))


@pytest.mark.parametrize("name", ["numpy", "pandas", "torch", "tensorflow", "jax"])
def test_get_backend_name(name):
    x = _make_backend_array(name, [[1.0, 2.0], [3.0, 4.0]])
    b = get_backend(x)
    assert b.name == name


@pytest.mark.parametrize("name", ["numpy", "pandas", "torch", "tensorflow", "jax"])
def test_backend_core_ops(name):
    b = get_backend(_make_backend_array(name, [1.0]))

    arr = b.asarray(_make_backend_array(name, [[1.0, -2.0], [3.0, -4.0]]))
    np.testing.assert_allclose(_to_numpy(b.abs(arr)), np.array([[1.0, 2.0], [3.0, 4.0]]))

    a = b.asarray(_make_backend_array(name, [[1.0, 5.0], [2.0, 3.0]]))
    c = b.asarray(_make_backend_array(name, [[2.0, 4.0], [2.0, 6.0]]))
    np.testing.assert_allclose(_to_numpy(b.maximum(a, c)), np.array([[2.0, 5.0], [2.0, 6.0]]))
    np.testing.assert_allclose(_to_numpy(b.minimum(a, c)), np.array([[1.0, 4.0], [2.0, 3.0]]))

    clipped = b.clip(a, 1.5, 4.5)
    np.testing.assert_allclose(_to_numpy(clipped), np.array([[1.5, 4.5], [2.0, 3.0]]))

    sq = b.sqrt(b.asarray(_make_backend_array(name, [1.0, 4.0, 9.0])))
    np.testing.assert_allclose(_to_numpy(sq), np.array([1.0, 2.0, 3.0]))

    cond = b.asarray(_make_backend_array(name, [[True, False], [False, True]]))
    w = b.where(cond, a, c)
    np.testing.assert_allclose(_to_numpy(w), np.array([[1.0, 4.0], [2.0, 3.0]]))
    assert b.any(cond)

    np.testing.assert_allclose(_to_numpy(b.sum(a, axis=1)), np.array([6.0, 5.0]))
    np.testing.assert_allclose(_to_numpy(b.mean(a, axis=0)), np.array([1.5, 4.0]))
    np.testing.assert_allclose(_to_numpy(b.max(a, axis=0)), np.array([2.0, 5.0]))
    np.testing.assert_allclose(_to_numpy(b.min(a, axis=0)), np.array([1.0, 3.0]))

    cs = b.cumsum(b.asarray(_make_backend_array(name, [[1, 2, 3], [4, 5, 6]])), axis=1)
    np.testing.assert_allclose(_to_numpy(cs), np.array([[1, 3, 6], [4, 9, 15]]))

    am = b.argmax(b.asarray(_make_backend_array(name, [[1, 9, 3], [4, 2, 8]])), axis=1)
    np.testing.assert_allclose(_to_numpy(am), np.array([1, 2]))


@pytest.mark.parametrize("name", ["numpy", "pandas", "torch", "tensorflow", "jax"])
def test_backend_shape_dtype_and_indexing_ops(name):
    b = get_backend(_make_backend_array(name, [1.0]))

    arr = b.asarray(_make_backend_array(name, [[10, 20, 30], [40, 50, 60]]))

    reshaped = b.reshape(arr, (3, 2))
    np.testing.assert_allclose(_to_numpy(reshaped), np.array([[10, 20], [30, 40], [50, 60]]))

    if name == "pandas":
        with pytest.raises(ValueError):
            b.reshape(arr, (2, 3, 1))
    else:
        squeezed = b.squeeze(b.reshape(arr, (2, 3, 1)))
        np.testing.assert_allclose(_to_numpy(squeezed), np.array([[10, 20, 30], [40, 50, 60]]))

    rng = b.arange(1, 6)
    np.testing.assert_allclose(_to_numpy(rng), np.array([1, 2, 3, 4, 5]))

    full_1d = b.full((4,), 7)
    np.testing.assert_allclose(_to_numpy(full_1d), np.array([7, 7, 7, 7]))

    full_2d = b.full((2, 3), -1.5)
    np.testing.assert_allclose(_to_numpy(full_2d), np.array([[-1.5, -1.5, -1.5], [-1.5, -1.5, -1.5]]))

    full_dtype = b.full((3,), 2, dtype="int64")
    full_dtype_np = _to_numpy(full_dtype)
    np.testing.assert_allclose(full_dtype_np, np.array([2, 2, 2], dtype=np.int64))
    if name == "jax":
        assert full_dtype_np.dtype in (np.int32, np.int64)
    else:
        assert full_dtype_np.dtype == np.int64

    eq = b.equal(b.asarray(_make_backend_array(name, [1, 2, 3])), b.asarray(_make_backend_array(name, [1, 0, 3])))
    np.testing.assert_allclose(_to_numpy(eq), np.array([True, False, True]))

    casted = b.astype(b.asarray(_make_backend_array(name, [1.2, 2.8])), "int64")
    np.testing.assert_allclose(_to_numpy(casted), np.array([1, 2]))

    ru = b.random_uniform((5,))
    ru_np = _to_numpy(ru)
    assert ru_np.shape == (5,)
    assert np.all((ru_np >= 0.0) & (ru_np <= 1.0))

    assert isinstance(b.scalar_at(b.asarray(_make_backend_array(name, [10, 20, 30])), 1), (int, float, np.integer, np.floating))

    idx = b.argsort(b.asarray(_make_backend_array(name, [[3, 1, 2], [5, 4, 6]])), axis=1, descending=False)
    np.testing.assert_allclose(_to_numpy(idx), np.array([[1, 2, 0], [1, 0, 2]]))

    ta = b.take_along_axis(
        b.asarray(_make_backend_array(name, [[10, 20, 30], [40, 50, 60]])),
        b.asarray(_make_backend_array(name, [[2, 0], [1, 1]])),
        axis=1,
    )
    np.testing.assert_allclose(_to_numpy(ta), np.array([[30, 10], [50, 50]]))


@pytest.mark.parametrize("name", ["numpy", "pandas", "torch", "tensorflow", "jax"])
def test_backend_concat_and_ones_like(name):
    b = get_backend(_make_backend_array(name, [1.0]))

    left = b.asarray(_make_backend_array(name, [[1.0, 2.0], [3.0, 4.0]]))
    right = b.ones_like(left)

    np.testing.assert_allclose(_to_numpy(right), np.array([[1.0, 1.0], [1.0, 1.0]]))

    merged = b.concat([left, right], axis=1)
    np.testing.assert_allclose(
        _to_numpy(merged),
        np.array([[1.0, 2.0, 1.0, 1.0], [3.0, 4.0, 1.0, 1.0]]),
    )

    merged_rows = b.concat([left, right], axis=0)
    np.testing.assert_allclose(
        _to_numpy(merged_rows),
        np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 1.0], [1.0, 1.0]]),
    )


@pytest.mark.parametrize("name", ["numpy", "pandas", "torch", "tensorflow", "jax"])
def test_shape2_split_concat_helpers(name):
    x = _make_backend_array(name, [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    b = get_backend(x)
    arr = b.asarray(x)

    ndim, shp = shape2(arr)
    assert ndim == 2
    assert shp[0] == 2
    assert shp[1] == 4

    c0, c1, c2, c3 = split_columns(arr, (0, 1, 2, 3), keepdims=True)
    merged = concat_columns([c0, c1, c2, c3], like=arr)

    np.testing.assert_allclose(_to_numpy(merged), np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))

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


def test_numpy_backend_accepts_pandas_inputs():
    pd = pytest.importorskip("pandas")
    from deel.puncc.api.backend import _NumpyOps

    ops = _NumpyOps()
    frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    series = pd.Series([5.0, 6.0])

    np.testing.assert_allclose(ops.asarray(frame), np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(ops.to_numpy(series), np.array([5.0, 6.0]))
    np.testing.assert_allclose(ops.to_numpy([7.0, 8.0]), np.array([7.0, 8.0]))


def test_pandas_backend_special_cases():
    pd = pytest.importorskip("pandas")

    b = get_backend(pd.Series([1.0]))
    nested = b.asarray([[1, 2], [3, 4]])
    assert isinstance(nested, pd.DataFrame)

    scalar_series = b.asarray([1, 2, 3])
    assert isinstance(scalar_series, pd.Series)

    frame = pd.DataFrame([[1.0, 3.0], [2.0, 4.0]], index=["a", "b"], columns=["x", "y"])
    cond = pd.DataFrame([[True, False], [False, True]], index=["b", "a"], columns=["y", "x"])
    other = pd.DataFrame([[10.0, 20.0], [30.0, 40.0]], index=["a", "b"], columns=["x", "y"])
    where_out = b.where(cond, frame, other)
    assert list(where_out.index) == ["a", "b"]
    assert list(where_out.columns) == ["x", "y"]

    keepdims_axis0 = b.sum(frame, axis=0, keepdims=True)
    keepdims_axis1 = b.sum(frame, axis=1, keepdims=True)
    assert keepdims_axis0.shape == (1, 2)
    assert keepdims_axis1.shape == (2, 1)

    keepdims_series = b.sum(pd.Series([1.0, 2.0, 3.0]), keepdims=True)
    assert isinstance(keepdims_series, pd.Series)
    assert keepdims_series.iloc[0] == 6.0

    reshaped = b.reshape(frame, (2, 2))
    assert list(reshaped.index) == ["a", "b"]

    with pytest.raises(ValueError):
        b.concat([], axis=1)

    full_3d = b.full((2, 2, 1), 1.0)
    assert isinstance(full_3d, np.ndarray)

    descending = b.argsort(frame, axis=1, descending=True)
    np.testing.assert_allclose(_to_numpy(descending), np.array([[1, 0], [1, 0]]))

    taken_frame = b.take_along_axis(frame, pd.DataFrame([[1], [0]], index=frame.index), axis=1)
    assert isinstance(taken_frame, pd.DataFrame)
    assert list(taken_frame.index) == ["a", "b"]

    taken_series = b.take_along_axis(pd.Series([9, 7, 8], index=["a", "b", "c"]), pd.Series([2, 0, 1]), axis=0)
    assert isinstance(taken_series, pd.Series)
    assert list(taken_series.index) == ["a", "b", "c"]


def test_pandas_backend_fallback_paths():
    pd = pytest.importorskip("pandas")
    from deel.puncc.api.backend import _PandasOps

    class NoToNumpy:
        def __array__(self):
            return np.array([1.0, 2.0, 3.0])

    class RaisingClipFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return RaisingClipFrame

        def clip(self, *args, **kwargs):
            raise TypeError("force numpy fallback")

    class RaisingCumsumFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return RaisingCumsumFrame

        def cumsum(self, *args, **kwargs):
            raise TypeError("force numpy fallback")

    class RaisingAstypeSeries(pd.Series):
        @property
        def _constructor(self):
            return RaisingAstypeSeries

        def astype(self, *args, **kwargs):
            raise TypeError("force numpy fallback")

    ops = _PandasOps()

    np.testing.assert_allclose(ops.to_numpy(NoToNumpy()), np.array([1.0, 2.0, 3.0]))

    max_mixed = ops.maximum(pd.Series([1.0, 5.0]), [2.0, 4.0])
    min_mixed = ops.minimum(pd.Series([1.0, 5.0]), [2.0, 4.0])
    np.testing.assert_allclose(_to_numpy(max_mixed), np.array([2.0, 5.0]))
    np.testing.assert_allclose(_to_numpy(min_mixed), np.array([1.0, 4.0]))
    clipped = ops.clip(RaisingClipFrame([[1.0, 5.0], [2.0, 7.0]]), [1.5, 2.5], [4.0, 6.0])
    np.testing.assert_allclose(_to_numpy(clipped), np.array([[1.5, 5.0], [2.0, 6.0]]))

    where_series = ops.where(
        pd.Series([True, False], index=["a", "b"]),
        pd.Series([1.0, 2.0], index=["a", "b"]),
        [10.0, 20.0],
    )
    np.testing.assert_allclose(
        _to_numpy(where_series), np.array([1.0, np.nan]), equal_nan=True
    )
    np.testing.assert_allclose(
        ops.where(np.array([True, False]), np.array([1.0, 2.0]), np.array([3.0, 4.0])),
        np.array([1.0, 4.0]),
    )

    csum = ops.cumsum(RaisingCumsumFrame([[1, 2], [3, 4]]), axis=1)
    np.testing.assert_allclose(_to_numpy(csum), np.array([[1, 3], [3, 7]]))

    squeezed = ops.squeeze(pd.DataFrame([[1.0], [2.0]], index=["a", "b"]))
    assert isinstance(squeezed, pd.Series)
    assert list(squeezed.index) == ["a", "b"]

    concat_mixed = ops.concat([pd.Series([1.0, 2.0]), [3.0, 4.0]], axis=0)
    np.testing.assert_allclose(_to_numpy(concat_mixed), np.array([1.0, 2.0, 3.0, 4.0]))

    casted = ops.astype(RaisingAstypeSeries([1.2, 2.8]), "int64")
    np.testing.assert_allclose(_to_numpy(casted), np.array([1, 2]))

    wrapped_index = ops._wrap_like(pd.Index(["x", "y"]), np.array([5.0, 6.0]))
    assert isinstance(wrapped_index, pd.Index)

    wrapped_other = ops._wrap_like(object(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(wrapped_other, np.array([1.0, 2.0]))

    assert ops._keepdims(pd.DataFrame([[1.0, 2.0]]), 3.0, axis=0) == 3.0
    wrapped_df_rowmatch = ops._wrap_like(
        pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["a", "b"]),
        np.array([[5.0], [6.0]]),
    )
    assert isinstance(wrapped_df_rowmatch, pd.DataFrame)
    assert list(wrapped_df_rowmatch.index) == ["a", "b"]
    wrapped_df_other = ops._wrap_like(
        pd.DataFrame([[1.0, 2.0]], index=["a"]),
        np.array([7.0, 8.0, 9.0]),
    )
    assert isinstance(wrapped_df_other, pd.Series)
    wrapped_series_rowmatch = ops._wrap_like(
        pd.Series([1.0, 2.0], index=["a", "b"]),
        np.array([[5.0], [6.0]]),
    )
    assert isinstance(wrapped_series_rowmatch, pd.DataFrame)


def test_torch_backend_branch_cases():
    torch = pytest.importorskip("torch")

    b = get_backend(torch.tensor([1.0]))
    arr = torch.tensor([[1.0, 5.0], [2.0, 3.0]])
    bool_arr = torch.tensor([[True, False], [False, True]])

    assert _to_numpy(b.max(arr)).item() == 5.0
    assert _to_numpy(b.min(arr)).item() == 1.0
    assert _to_numpy(b.argmax(arr)).item() == 1
    np.testing.assert_allclose(_to_numpy(b.argmax(bool_arr, axis=1)), np.array([0, 1]))
    np.testing.assert_allclose(_to_numpy(b.arange(4)), np.array([0, 1, 2, 3]))

    np.testing.assert_allclose(_to_numpy(b.asarray([[1, 2], [3, 4]])), np.array([[1, 2], [3, 4]]))

    pd = pytest.importorskip("pandas")
    np.testing.assert_allclose(
        _to_numpy(b.asarray(pd.DataFrame([[1, 2], [3, 4]]))),
        np.array([[1, 2], [3, 4]]),
    )
    np.testing.assert_allclose(b.to_numpy(torch.tensor([1, 2])), np.array([1, 2]))
    np.testing.assert_allclose(_to_numpy(b.to_numpy(pd.Series([1, 2]))), np.array([1, 2]))
    np.testing.assert_allclose(b.to_numpy([1, 2, 3]), np.array([1, 2, 3]))

    with pytest.raises(TypeError):
        b.full((2,), 1, dtype=object())

    with pytest.raises(TypeError):
        b.astype(torch.tensor([1.0]), object())


def test_jax_backend_branch_cases():
    jnp = pytest.importorskip("jax.numpy")
    pd = pytest.importorskip("pandas")

    b = get_backend(jnp.asarray([1.0]))
    arr = b.asarray(pd.DataFrame([[3.0, 1.0], [2.0, 4.0]]))
    np.testing.assert_allclose(_to_numpy(arr), np.array([[3.0, 1.0], [2.0, 4.0]]))
    np.testing.assert_allclose(b.to_numpy(arr), np.array([[3.0, 1.0], [2.0, 4.0]]))

    desc = b.argsort(arr, axis=1, descending=True)
    np.testing.assert_allclose(_to_numpy(desc), np.array([[0, 1], [1, 0]]))


def test_tensorflow_backend_branch_cases():
    tf = pytest.importorskip("tensorflow")
    pd = pytest.importorskip("pandas")

    b = get_backend(tf.convert_to_tensor([1.0]))
    frame = pd.DataFrame([[1, 2], [3, 4]])
    np.testing.assert_allclose(_to_numpy(b.asarray(frame)), np.array([[1, 2], [3, 4]]))
    np.testing.assert_allclose(_to_numpy(b.asarray([[5, 6], [7, 8]])), np.array([[5, 6], [7, 8]]))
    np.testing.assert_allclose(_to_numpy(b.to_numpy(frame)), np.array([[1, 2], [3, 4]]))
    np.testing.assert_allclose(b.to_numpy([1, 2, 3]), np.array([1, 2, 3]))
    np.testing.assert_allclose(_to_numpy(b.arange(4)), np.array([0, 1, 2, 3]))

    arr = tf.constant([[10, 20, 30], [40, 50, 60]])
    ind = tf.constant([[2, 0], [1, 1]])
    ta = b.take_along_axis(arr, ind, axis=1)
    np.testing.assert_allclose(_to_numpy(ta), np.array([[30, 10], [50, 50]]))


def test_get_backend_shape_and_column_helpers_branches():
    pd = pytest.importorskip("pandas")
    tf = pytest.importorskip("tensorflow")
    torch = pytest.importorskip("torch")
    jnp = pytest.importorskip("jax.numpy")
    import deel.puncc.api.backend as backend_module

    original = backend_module._BACKENDS["numpy"]
    try:
        backend_module._BACKENDS["numpy"] = None
        with pytest.raises(RuntimeError):
            get_backend(np.array([1, 2, 3]))
    finally:
        backend_module._BACKENDS["numpy"] = original

    class BrokenShape:
        def __iter__(self):
            raise TypeError("broken shape")

    class ShapeRaises:
        @property
        def shape(self):
            return BrokenShape()

        def __array__(self):
            return np.array([[1, 2], [3, 4]])

    ndim, shape = shape2(ShapeRaises())
    assert (ndim, shape) == (2, (2, 2))
    ndim_list, shape_list = shape2([[1, 2], [3, 4]])
    assert (ndim_list, shape_list) == (2, (2, 2))

    frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    c0, c1 = split_columns(frame, (0, 1), keepdims=True)
    assert isinstance(c0, pd.Series)
    merged_pd = concat_columns([c0, c1], like=frame)
    assert isinstance(merged_pd, pd.DataFrame)
    c0_nk, c1_nk = split_columns(frame, (0, 1), keepdims=False)
    assert isinstance(c0_nk, pd.Series)
    assert isinstance(c1_nk, pd.Series)

    tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    tc0, tc1 = split_columns(tensor, (0, 1), keepdims=True)
    np.testing.assert_allclose(_to_numpy(tc0), np.array([[1.0], [3.0]]))
    np.testing.assert_allclose(_to_numpy(concat_columns([tc0, tc1], like=tensor)), np.array([[1.0, 2.0], [3.0, 4.0]]))
    tc0_nk, tc1_nk = split_columns(tensor, (0, 1), keepdims=False)
    np.testing.assert_allclose(_to_numpy(tc0_nk), np.array([1.0, 3.0]))
    np.testing.assert_allclose(_to_numpy(tc1_nk), np.array([2.0, 4.0]))

    torch_cols = [torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])]
    np.testing.assert_allclose(_to_numpy(concat_columns(torch_cols, like=torch_cols[0])), np.array([[1.0, 3.0], [2.0, 4.0]]))

    jax_cols = [jnp.asarray([[1.0], [2.0]]), jnp.asarray([[3.0], [4.0]])]
    np.testing.assert_allclose(_to_numpy(concat_columns(jax_cols, like=jax_cols[0])), np.array([[1.0, 3.0], [2.0, 4.0]]))


def test_is_jax_false_when_jax_numpy_missing(monkeypatch):
    import deel.puncc.api.backend as backend_module

    monkeypatch.setitem(backend_module.sys.modules, "jax", object())
    monkeypatch.delitem(backend_module.sys.modules, "jax.numpy", raising=False)

    assert backend_module._is_jax(np.array([1.0])) is False


def test_pandas_backend_forced_mix_and_take_along_axis_branches(monkeypatch):
    pd = pytest.importorskip("pandas")
    from deel.puncc.api.backend import _PandasOps

    ops = _PandasOps()
    frame = pd.DataFrame([[1.0, 5.0], [2.0, 3.0]], index=["a", "b"])
    series = pd.Series([9.0, 7.0], index=["a", "b"], name="score")

    original_asarray = _PandasOps.asarray

    def patched_asarray(self, x):
        if isinstance(x, np.ndarray):
            return x
        return original_asarray(self, x)

    monkeypatch.setattr(_PandasOps, "asarray", patched_asarray)

    max_df = ops.maximum(frame, np.array([[2.0, 4.0], [1.0, 6.0]]))
    min_df = ops.minimum(frame, np.array([[2.0, 4.0], [1.0, 6.0]]))
    np.testing.assert_allclose(_to_numpy(max_df), np.array([[2.0, 5.0], [2.0, 6.0]]))
    np.testing.assert_allclose(_to_numpy(min_df), np.array([[1.0, 4.0], [1.0, 3.0]]))
    np.testing.assert_allclose(
        ops.where(np.array([True, False]), np.array([1.0, 2.0]), np.array([3.0, 4.0])),
        np.array([1.0, 4.0]),
    )

    monkeypatch.setattr(
        np,
        "take_along_axis",
        lambda arr, indices, axis: np.array([1.0, 2.0]),
    )
    out_frame_1d = ops.take_along_axis(frame, pd.DataFrame([[0], [0]], index=frame.index), axis=1)
    assert isinstance(out_frame_1d, pd.Series)
    assert list(out_frame_1d.index) == ["a", "b"]

    out_series_2d = ops.take_along_axis(series, pd.Series([0, 0]), axis=0)
    assert isinstance(out_series_2d, pd.Series)

    monkeypatch.setattr(
        np,
        "take_along_axis",
        lambda arr, indices, axis: np.array([[1.0], [2.0]]),
    )
    out_series_as_df = ops.take_along_axis(series, pd.DataFrame([[0], [0]], index=series.index), axis=0)
    assert isinstance(out_series_as_df, pd.DataFrame)
    assert list(out_series_as_df.index) == ["a", "b"]


def test_torch_backend_unsupported_dtype_branches():
    torch = pytest.importorskip("torch")
    from deel.puncc.api.backend import _TorchOps

    ops = _TorchOps()
    arr = torch.tensor([1.0])

    with pytest.raises(TypeError):
        ops.full((2,), 1, dtype=np.dtype("V1"))

    with pytest.raises(TypeError):
        ops.astype(arr, np.dtype("V1"))


def test_pandas_backend_take_along_axis_returns_raw_output(monkeypatch):
    pd = pytest.importorskip("pandas")
    from deel.puncc.api.backend import _PandasOps

    ops = _PandasOps()
    series = pd.Series([9.0, 7.0], index=["a", "b"], name="score")
    monkeypatch.setattr(
        np,
        "take_along_axis",
        lambda arr, indices, axis: np.array(5.0),
    )

    out = ops.take_along_axis(series, pd.Series([0, 0]), axis=0)
    assert np.array(out).shape == ()

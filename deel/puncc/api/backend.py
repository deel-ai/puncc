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
# deel/puncc/api/backend.py
"""Backend abstraction layer for array/tensor operations.

This module provides backend inference and a unified API over NumPy, pandas,
PyTorch, JAX and TensorFlow objects.
"""
# pylint: disable=C0115,C0116,C0321,C0415,R0911,C0301
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Protocol
from typing import Sequence
from typing import Tuple
from typing import runtime_checkable

import numpy as _np
import sys


# -------------------------
# Detection helpers
# -------------------------

def _is_torch(x: Any) -> bool:
    torch = sys.modules.get("torch")
    return bool(torch) and isinstance(x, torch.Tensor)

def _is_jax(x: Any) -> bool:
    jax = sys.modules.get("jax")
    jnp = sys.modules.get("jax.numpy")
    if not (jax and jnp):
        return False
    # jax.Array exists on newer JAX; keep jnp.ndarray for older
    return isinstance(x, (getattr(jax, "Array", ()), getattr(jnp, "ndarray", ())))

def _is_tf(x: Any) -> bool:
    tf = sys.modules.get("tensorflow")
    return bool(tf) and isinstance(x, (tf.Tensor, tf.Variable))

def _is_pandas(x: Any) -> bool:
    pd = sys.modules.get("pandas")
    return bool(pd) and isinstance(x, (pd.Series, pd.DataFrame, pd.Index))


def infer_backend(*xs: Any) -> str:
    """Infer backend name from one or more objects.

    Priority order is: ``torch > jax > tensorflow > pandas > numpy``.
    Mixed explicit backends are rejected.

    :param Any xs: objects used to infer the computational backend.

    :returns: inferred backend name.
    :rtype: str

    :raises TypeError: if multiple incompatible backends are mixed.
    """
    kinds = set()
    for x in xs:
        if x is None:
            continue
        if _is_torch(x):
            kinds.add("torch")
        elif _is_jax(x):
            kinds.add("jax")
        elif _is_tf(x):
            kinds.add("tensorflow")
        elif _is_pandas(x):
            kinds.add("pandas")
        else:
            # treat everything else as numpy-ish (lists, tuples, np.ndarray, scalars)
            kinds.add("numpy")

    if len(kinds) > 1:
        raise TypeError(f"Mixed backends are not supported: {sorted(kinds)}")

    return next(iter(kinds), "numpy")


# -------------------------
# Canonical ops interface
# -------------------------

@runtime_checkable
class BackendOps(Protocol):
    """Protocol describing backend operations used across the API."""

    name: str

    # conversion / construction
    def asarray(self, x: Any) -> Any: ...
    def to_numpy(self, x: Any) -> _np.ndarray: ...

    # elementwise
    def abs(self, x: Any) -> Any: ...
    def maximum(self, a: Any, b: Any) -> Any: ...
    def minimum(self, a: Any, b: Any) -> Any: ...
    def clip(self, x: Any, a_min: Any, a_max: Any) -> Any: ...
    def sqrt(self, x: Any) -> Any: ...
    def where(self, cond: Any, x: Any, y: Any) -> Any: ...
    def any(self, x: Any) -> bool: ...

    # reductions
    def sum(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any: ...
    def mean(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any: ...
    def max(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any: ...
    def min(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any: ...
    def cumsum(self, x: Any, axis: int = 0) -> Any: ...
    def argmax(self, x: Any, axis: Any = None) -> Any: ...

    # shape utilities
    def reshape(self, x: Any, shape: Tuple[int, ...]) -> Any: ...
    def squeeze(self, x: Any, axis: Any = None) -> Any: ...
    def concat(self, xs: Sequence[Any], axis: int = 0) -> Any: ...

    # creation / typing / comparisons
    def arange(self, start: int, stop: Any = None, step: int = 1) -> Any: ...
    def full(
        self, shape: Tuple[int, ...], fill_value: Any, dtype: Any = None
    ) -> Any: ...
    def ones_like(self, x: Any) -> Any: ...
    def equal(self, a: Any, b: Any) -> Any: ...
    def astype(self, x: Any, dtype: Any) -> Any: ...
    def random_uniform(self, shape: Tuple[int, ...]) -> Any: ...
    def scalar_at(self, x: Any, index: int) -> Any: ...

    # ordering / indexing (needed for RAPS-like scores, quantiles, bbox sets, etc.)
    def argsort(self, x: Any, axis: int = -1, descending: bool = False) -> Any: ...
    def take_along_axis(self, arr: Any, indices: Any, axis: int) -> Any: ...


# -------------------------
# Concrete backends
# -------------------------

@dataclass(frozen=True)
class _NumpyOps:
    name: str = "numpy"

    def asarray(self, x: Any) -> _np.ndarray:
        if _is_pandas(x):
            return x.to_numpy()
        return _np.asarray(x)

    def to_numpy(self, x: Any) -> _np.ndarray:
        if isinstance(x, _np.ndarray):
            return x
        if _is_pandas(x):
            return x.to_numpy()
        # accept list-like
        return _np.asarray(x)

    def abs(self, x): return _np.abs(x)
    def maximum(self, a, b): return _np.maximum(a, b)
    def minimum(self, a, b): return _np.minimum(a, b)
    def clip(self, x, a_min, a_max): return _np.clip(x, a_min, a_max)
    def sqrt(self, x): return _np.sqrt(x)
    def where(self, cond, x, y): return _np.where(cond, x, y)
    def any(self, x): return bool(_np.any(x))

    def sum(self, x, axis=None, keepdims=False): return _np.sum(x, axis=axis,
                                                                keepdims=keepdims)
    def mean(self, x, axis=None, keepdims=False): return _np.mean(x, axis=axis, keepdims=keepdims)
    def max(self, x, axis=None, keepdims=False): return _np.max(x, axis=axis, keepdims=keepdims)
    def min(self, x, axis=None, keepdims=False): return _np.min(x, axis=axis, keepdims=keepdims)
    def cumsum(self, x, axis=0): return _np.cumsum(x, axis=axis)
    def argmax(self, x, axis=None): return _np.argmax(x, axis=axis)

    def reshape(self, x, shape): return _np.reshape(x, shape)
    def squeeze(self, x, axis=None): return _np.squeeze(x, axis=axis)
    def concat(self, xs, axis=0): return _np.concatenate(xs, axis=axis)
    def arange(self, start, stop=None, step=1): return _np.arange(start, stop, step)
    def full(self, shape, fill_value, dtype=None): return _np.full(shape, fill_value, dtype=dtype)
    def ones_like(self, x): return _np.ones_like(x)
    def equal(self, a, b): return _np.equal(a, b)
    def astype(self, x, dtype): return _np.asarray(x).astype(dtype)
    def random_uniform(self, shape): return _np.random.uniform(size=shape)
    def scalar_at(self, x, index):
        return _np.asarray(x).reshape(-1)[index].item()

    def argsort(self, x, axis=-1, descending=False):
        idx = _np.argsort(x, axis=axis)
        if descending:
            idx = _np.flip(idx, axis=axis)
        return idx

    def take_along_axis(self, arr, indices, axis):
        return _np.take_along_axis(arr, indices, axis=axis)


@dataclass(frozen=True)
class _PandasOps:
    name: str = "pandas"

    def asarray(self, x: Any):
        import pandas as pd
        if isinstance(x, (pd.DataFrame, pd.Series, pd.Index)):
            return x
        # Heuristique minimale
        if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple, dict)):
            return pd.DataFrame(x)
        return pd.Series(x)

    def to_numpy(self, x: Any) -> _np.ndarray:
        try:
            return x.to_numpy()
        except Exception:
            return _np.asarray(x)

    # ---------- elementwise ----------
    def abs(self, x):
        x = self.asarray(x)
        return x.abs() if hasattr(x, "abs") else _np.abs(self.to_numpy(x))

    def maximum(self, a, b):
        a = self.asarray(a); b = self.asarray(b)
        import pandas as pd
        if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
            return a.where(a >= b, b)
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            return a.where(a >= b, b)
        # mix -> align via numpy, rewrap like a
        out = _np.maximum(self.to_numpy(a), self.to_numpy(b))
        return self._wrap_like(a, out)

    def minimum(self, a, b):
        a = self.asarray(a); b = self.asarray(b)
        import pandas as pd
        if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
            return a.where(a <= b, b)
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            return a.where(a <= b, b)
        out = _np.minimum(self.to_numpy(a), self.to_numpy(b))
        return self._wrap_like(a, out)

    def clip(self, x, a_min, a_max):
        x = self.asarray(x)
        if hasattr(x, "clip"):
            # pandas clip (bornes scalaires)
            try:
                return x.clip(lower=a_min, upper=a_max)
            except Exception:
                pass
        out = _np.clip(self.to_numpy(x), _np.asarray(a_min), _np.asarray(a_max))
        return self._wrap_like(x, out)

    def sqrt(self, x):
        x = self.asarray(x)
        out = _np.sqrt(self.to_numpy(x))
        return self._wrap_like(x, out)

    def where(self, cond, x, y):
        x = self.asarray(x); y = self.asarray(y)
        import pandas as pd
        if isinstance(x, pd.DataFrame):
            cond = self.asarray(cond)
            if isinstance(cond, pd.DataFrame):
                cond = cond.reindex(index=x.index, columns=x.columns)
            return x.where(cond, other=y if isinstance(y, pd.DataFrame) else self._wrap_like(x, self.to_numpy(y)))
        if isinstance(x, pd.Series):
            cond = self.asarray(cond)
            if isinstance(cond, pd.Series):
                cond = cond.reindex(x.index)
            return x.where(cond, other=y if isinstance(y, pd.Series) else self._wrap_like(x, self.to_numpy(y)))
        return _np.where(self.to_numpy(cond), self.to_numpy(x), self.to_numpy(y))
    def any(self, x):
        return bool(_np.any(self.to_numpy(self.asarray(x))))

    # ---------- reductions ----------
    def sum(self, x, axis=None, keepdims=False):
        x = self.asarray(x)
        out = x.sum(axis=axis)
        return self._keepdims(x, out, axis) if keepdims else out

    def mean(self, x, axis=None, keepdims=False):
        x = self.asarray(x)
        out = x.mean(axis=axis)
        return self._keepdims(x, out, axis) if keepdims else out

    def max(self, x, axis=None, keepdims=False):
        x = self.asarray(x)
        out = x.max(axis=axis)
        return self._keepdims(x, out, axis) if keepdims else out

    def min(self, x, axis=None, keepdims=False):
        x = self.asarray(x)
        out = x.min(axis=axis)
        return self._keepdims(x, out, axis) if keepdims else out
    def cumsum(self, x, axis=0):
        x = self.asarray(x)
        if hasattr(x, "cumsum"):
            try:
                return x.cumsum(axis=axis)
            except Exception:
                pass
        return self._wrap_like(x, _np.cumsum(self.to_numpy(x), axis=axis))
    def argmax(self, x, axis=None):
        return _np.argmax(self.to_numpy(self.asarray(x)), axis=axis)

    # ---------- shape utils ----------
    def reshape(self, x, shape: Tuple[int, ...]):
        # Pandas n’est pas un backend N-D. Politique stricte.
        # Autoriser uniquement reshape 2D DataFrame sans promesse de labels.
        x = self.asarray(x)
        out = self.to_numpy(x).reshape(shape)
        if len(shape) != 2:
            raise ValueError("pandas backend supports reshape only to 2D (DataFrame).")
        import pandas as pd
        idx = x.index if hasattr(x, "index") and shape[0] == len(x.index) else None
        return pd.DataFrame(out, index=idx)

    def squeeze(self, x, axis=None):
        # pandas: delegate to numpy then rewrap.
        x = self.asarray(x)
        out = _np.squeeze(self.to_numpy(x), axis=axis)
        return self._wrap_like(x, out)
    def concat(self, xs, axis=0):
        import pandas as pd

        if len(xs) == 0:
            raise ValueError("concat requires at least one array to concatenate.")
        if all(isinstance(x, (pd.DataFrame, pd.Series, pd.Index)) for x in xs):
            return pd.concat(xs, axis=axis)

        out = _np.concatenate([self.to_numpy(self.asarray(x)) for x in xs], axis=axis)
        return self._wrap_like(self.asarray(xs[0]), out)
    def arange(self, start, stop=None, step=1): return _np.arange(start, stop, step)
    def full(self, shape, fill_value, dtype=None):
        out = _np.full(shape, fill_value, dtype=dtype)
        import pandas as pd
        if len(shape) == 1:
            return pd.Series(out)
        if len(shape) == 2:
            return pd.DataFrame(out)
        return out
    def ones_like(self, x):
        x = self.asarray(x)
        out = _np.ones_like(self.to_numpy(x))
        return self._wrap_like(x, out)
    def equal(self, a, b):
        a = self.asarray(a)
        b = self.asarray(b)
        out = _np.equal(self.to_numpy(a), self.to_numpy(b))
        return self._wrap_like(a, out)
    def astype(self, x, dtype):
        x = self.asarray(x)
        if hasattr(x, "astype"):
            try:
                return x.astype(dtype)
            except Exception:
                pass
        return self._wrap_like(x, self.to_numpy(x).astype(dtype))
    def random_uniform(self, shape): return _np.random.uniform(size=shape)
    def scalar_at(self, x, index):
        return _np.asarray(self.to_numpy(self.asarray(x))).reshape(-1)[index].item()

    # ---------- ordering / indexing ----------
    def argsort(self, x, axis=-1, descending=False):
        x = self.asarray(x)
        arr = self.to_numpy(x)
        idx = _np.argsort(arr, axis=axis)
        if descending:
            idx = _np.flip(idx, axis=axis)
        return self._wrap_like(x, idx)

    def take_along_axis(self, arr, indices, axis: int):
        # This operation breaks the semantics of columns if axis=1 with row-wise indices.
        # We return DataFrame/Series with preserved index, columns RangeIndex.
        import pandas as pd
        arr = self.asarray(arr)
        ind = self.asarray(indices)
        out = _np.take_along_axis(self.to_numpy(arr), self.to_numpy(ind), axis=axis)

        if isinstance(arr, pd.DataFrame):
            if out.ndim == 2:
                return pd.DataFrame(out, index=arr.index)
            if out.ndim == 1:
                return pd.Series(out, index=arr.index)
        if isinstance(arr, pd.Series):
            if out.ndim == 1:
                return pd.Series(out, index=arr.index, name=arr.name)
            if out.ndim == 2:
                return pd.DataFrame(out, index=arr.index)
        return out

    # ---------- helpers ----------
    def _wrap_like(self, ref, out: _np.ndarray):
        import pandas as pd
        out = _np.asarray(out)

        if isinstance(ref, pd.DataFrame):
            if out.ndim == 2 and out.shape == ref.shape:
                return pd.DataFrame(out, index=ref.index, columns=ref.columns)
            if out.ndim == 2 and out.shape[0] == ref.shape[0]:
                return pd.DataFrame(out, index=ref.index)
            if out.ndim == 1 and out.shape[0] == ref.shape[0]:
                return pd.Series(out, index=ref.index)
            return pd.DataFrame(out) if out.ndim == 2 else pd.Series(out)

        if isinstance(ref, pd.Series):
            if out.ndim == 1 and out.shape[0] == ref.shape[0]:
                return pd.Series(out, index=ref.index, name=ref.name)
            if out.ndim == 2 and out.shape[0] == ref.shape[0]:
                return pd.DataFrame(out, index=ref.index)
            return pd.Series(out) if out.ndim == 1 else pd.DataFrame(out)

        if isinstance(ref, pd.Index):
            return pd.Index(out, name=ref.name) if out.ndim == 1 else pd.DataFrame(out)

        return out

    def _keepdims(self, ref, reduced, axis):
        import pandas as pd
        if isinstance(ref, pd.DataFrame) and isinstance(reduced, pd.Series):
            if axis in (None, 0, "index"):
                return reduced.to_frame().T
            if axis in (1, "columns"):
                return reduced.to_frame()
        if isinstance(ref, pd.Series) and not isinstance(reduced, pd.Series):
            return pd.Series([reduced], name=ref.name)
        return reduced

@dataclass(frozen=True)
class _TorchOps:
    name: str = "torch"

    @property
    def _torch(self):
        import torch
        return torch

    def asarray(self, x: Any):
        t = self._torch
        if isinstance(x, t.Tensor):
            return x
        if _is_pandas(x):
            return t.as_tensor(x.to_numpy())
        return t.as_tensor(x)

    def to_numpy(self, x: Any) -> _np.ndarray:
        t = self._torch
        if isinstance(x, t.Tensor):
            return x.detach().cpu().numpy()
        if _is_pandas(x):
            return x.to_numpy()
        return _np.asarray(x)

    def abs(self, x): return self._torch.abs(x)
    def maximum(self, a, b): return self._torch.maximum(a, b)
    def minimum(self, a, b): return self._torch.minimum(a, b)
    def clip(self, x, a_min, a_max): return self._torch.clamp(x, min=a_min, max=a_max)
    def sqrt(self, x): return self._torch.sqrt(x)
    def where(self, cond, x, y): return self._torch.where(cond, x, y)
    def any(self, x): return bool(self._torch.any(x).item())

    def sum(self, x, axis=None, keepdims=False):
        return self._torch.sum(x, dim=axis, keepdim=keepdims)
    def mean(self, x, axis=None, keepdims=False):
        return self._torch.mean(x, dim=axis, keepdim=keepdims)
    def max(self, x, axis=None, keepdims=False):
        if axis is None:
            return self._torch.max(x)
        v, _ = self._torch.max(x, dim=axis, keepdim=keepdims)
        return v
    def min(self, x, axis=None, keepdims=False):
        if axis is None:
            return self._torch.min(x)
        v, _ = self._torch.min(x, dim=axis, keepdim=keepdims)
        return v
    def cumsum(self, x, axis=0): return self._torch.cumsum(x, dim=axis)
    def argmax(self, x, axis=None):
        if hasattr(x, "dtype") and x.dtype == self._torch.bool:
            x = x.to(dtype=self._torch.int64)
        if axis is None:
            return self._torch.argmax(x)
        return self._torch.argmax(x, dim=axis)

    def reshape(self, x, shape): return x.reshape(shape)
    def squeeze(self, x, axis=None): return x.squeeze(dim=axis) if axis is not None else x.squeeze()
    def concat(self, xs, axis=0): return self._torch.cat(xs, dim=axis)
    def arange(self, start, stop=None, step=1):
        if stop is None:
            return self._torch.arange(start)
        return self._torch.arange(start, stop, step)
    def full(self, shape, fill_value, dtype=None):
        kwargs = {}
        if dtype is not None:
            torch_dtype = getattr(self._torch, str(dtype), None)
            if torch_dtype is None:
                torch_dtype = getattr(self._torch, str(_np.dtype(dtype)), None)
            if torch_dtype is None:
                raise TypeError(f"Unsupported torch dtype: {dtype}")
            kwargs["dtype"] = torch_dtype
        return self._torch.full(shape, fill_value, **kwargs)
    def ones_like(self, x): return self._torch.ones_like(x)
    def equal(self, a, b): return self._torch.eq(a, b)
    def astype(self, x, dtype):
        torch_dtype = getattr(self._torch, str(dtype), None)
        if torch_dtype is None:
            torch_dtype = getattr(self._torch, str(_np.dtype(dtype)), None)
        if torch_dtype is None:
            raise TypeError(f"Unsupported torch dtype: {dtype}")
        return x.to(dtype=torch_dtype)
    def random_uniform(self, shape): return self._torch.rand(shape)
    def scalar_at(self, x, index):
        return self.asarray(x).reshape(-1)[index].item()

    def argsort(self, x, axis=-1, descending=False):
        return self._torch.argsort(x, dim=axis, descending=descending)

    def take_along_axis(self, arr, indices, axis):
        # torch has gather; indices must be same shape as output
        return self._torch.gather(arr, dim=axis, index=indices)


@dataclass(frozen=True)
class _JaxOps:
    name: str = "jax"

    @property
    def _jnp(self):
        import jax.numpy as jnp
        return jnp

    def asarray(self, x: Any):
        if _is_pandas(x):
            return self._jnp.asarray(x.to_numpy())
        return self._jnp.asarray(x)

    def to_numpy(self, x: Any) -> _np.ndarray:
        return _np.asarray(x)

    def abs(self, x): return self._jnp.abs(x)
    def maximum(self, a, b): return self._jnp.maximum(a, b)
    def minimum(self, a, b): return self._jnp.minimum(a, b)
    def clip(self, x, a_min, a_max): return self._jnp.clip(x, a_min, a_max)
    def sqrt(self, x): return self._jnp.sqrt(x)
    def where(self, cond, x, y): return self._jnp.where(cond, x, y)
    def any(self, x): return bool(_np.asarray(self._jnp.any(x)))

    def sum(self, x, axis=None, keepdims=False): return self._jnp.sum(x, axis=axis, keepdims=keepdims)
    def mean(self, x, axis=None, keepdims=False): return self._jnp.mean(x, axis=axis, keepdims=keepdims)
    def max(self, x, axis=None, keepdims=False): return self._jnp.max(x, axis=axis, keepdims=keepdims)
    def min(self, x, axis=None, keepdims=False): return self._jnp.min(x, axis=axis, keepdims=keepdims)
    def cumsum(self, x, axis=0): return self._jnp.cumsum(x, axis=axis)
    def argmax(self, x, axis=None): return self._jnp.argmax(x, axis=axis)

    def reshape(self, x, shape): return self._jnp.reshape(x, shape)
    def squeeze(self, x, axis=None): return self._jnp.squeeze(x, axis=axis)
    def concat(self, xs, axis=0): return self._jnp.concatenate(xs, axis=axis)
    def arange(self, start, stop=None, step=1): return self._jnp.arange(start, stop, step)
    def full(self, shape, fill_value, dtype=None): return self._jnp.full(shape, fill_value, dtype=dtype)
    def ones_like(self, x): return self._jnp.ones_like(x)
    def equal(self, a, b): return self._jnp.equal(a, b)
    def astype(self, x, dtype): return x.astype(dtype)
    def random_uniform(self, shape): return self._jnp.asarray(_np.random.uniform(size=shape))
    def scalar_at(self, x, index):
        return _np.asarray(self.asarray(x).reshape(-1)[index]).item()

    def argsort(self, x, axis=-1, descending=False):
        idx = self._jnp.argsort(x, axis=axis)
        if descending:
            idx = self._jnp.flip(idx, axis=axis)
        return idx

    def take_along_axis(self, arr, indices, axis):
        return self._jnp.take_along_axis(arr, indices, axis=axis)


@dataclass(frozen=True)
class _TensorflowOps:
    name: str = "tensorflow"

    @property
    def _tf(self):
        import tensorflow as tf
        return tf

    def asarray(self, x: Any):
        tf = self._tf
        if isinstance(x, (tf.Tensor, tf.Variable)):
            return x
        if _is_pandas(x):
            return tf.convert_to_tensor(x.to_numpy())
        return tf.convert_to_tensor(x)

    def to_numpy(self, x: Any) -> _np.ndarray:
        tf = self._tf
        if isinstance(x, (tf.Tensor, tf.Variable)):
            return x.numpy()
        if _is_pandas(x):
            return x.to_numpy()
        return _np.asarray(x)

    def abs(self, x): return self._tf.abs(x)
    def maximum(self, a, b): return self._tf.maximum(a, b)
    def minimum(self, a, b): return self._tf.minimum(a, b)
    def clip(self, x, a_min, a_max): return self._tf.clip_by_value(x, a_min, a_max)
    def sqrt(self, x): return self._tf.sqrt(x)
    def where(self, cond, x, y): return self._tf.where(cond, x, y)
    def any(self, x): return bool(self.to_numpy(self._tf.reduce_any(x)))

    # TF reduction names differ -> normalize here
    def sum(self, x, axis=None, keepdims=False): return self._tf.reduce_sum(x, axis=axis, keepdims=keepdims)
    def mean(self, x, axis=None, keepdims=False): return self._tf.reduce_mean(x, axis=axis, keepdims=keepdims)
    def max(self, x, axis=None, keepdims=False): return self._tf.reduce_max(x, axis=axis, keepdims=keepdims)
    def min(self, x, axis=None, keepdims=False): return self._tf.reduce_min(x, axis=axis, keepdims=keepdims)
    def cumsum(self, x, axis=0): return self._tf.cumsum(x, axis=axis)
    def argmax(self, x, axis=None):
        return self._tf.argmax(x, axis=axis, output_type=self._tf.int64)

    def reshape(self, x, shape): return self._tf.reshape(x, shape)
    def squeeze(self, x, axis=None): return self._tf.squeeze(x, axis=axis)
    def concat(self, xs, axis=0): return self._tf.concat(xs, axis=axis)
    def arange(self, start, stop=None, step=1):
        if stop is None:
            return self._tf.range(start)
        return self._tf.range(start, limit=stop, delta=step)
    def full(self, shape, fill_value, dtype=None):
        out = self._tf.fill(shape, fill_value)
        if dtype is not None:
            out = self._tf.cast(out, self._tf.as_dtype(dtype))
        return out
    def ones_like(self, x): return self._tf.ones_like(x)
    def equal(self, a, b): return self._tf.equal(a, b)
    def astype(self, x, dtype): return self._tf.cast(x, self._tf.as_dtype(dtype))
    def random_uniform(self, shape): return self._tf.random.uniform(shape)
    def scalar_at(self, x, index):
        flat = self._tf.reshape(self.asarray(x), (-1,))
        return self.to_numpy(flat[index]).item()

    def argsort(self, x, axis=-1, descending=False):
        direction = "DESCENDING" if descending else "ASCENDING"
        return self._tf.argsort(x, axis=axis, direction=direction)

    def take_along_axis(self, arr, indices, axis):
        # TF has gather with batch_dims trick; simplest reliable: use gather_nd via one-hot-ish indices building
        # Keep minimal: require indices same shape; use tf.gather with batch_dims where possible.
        tf = self._tf
        # Move axis to last for gather, then invert permutation
        rank = tf.rank(arr)
        axis_ = axis if axis >= 0 else axis + arr.shape.rank
        perm = tf.concat([tf.range(axis_), tf.range(axis_ + 1, rank), [axis_]], axis=0)
        inv = tf.argsort(perm)
        a = tf.transpose(arr, perm)
        ind = tf.transpose(indices, perm)
        out = tf.gather(a, ind, axis=-1, batch_dims=a.shape.rank - 1)
        return tf.transpose(out, inv)


# ---------
# Registry
# ---------

_BACKENDS = {
    "numpy": _NumpyOps,
    "pandas": _PandasOps,
    "torch": _TorchOps,
    "jax": _JaxOps,
    "tensorflow": _TensorflowOps,
}


def get_backend(*xs: Any) -> BackendOps:
    """Instantiate backend operations from input objects.

    :param Any xs: objects used to infer backend.

    :returns: backend operations instance implementing :class:`BackendOps`.
    :rtype: BackendOps

    :raises RuntimeError: if inferred backend has no registered implementation.
    """
    name = infer_backend(*xs)
    cls = _BACKENDS.get(name)
    if cls is None:
        raise RuntimeError(f"Unsupported backend: {name}")
    return cls()


def shape2(x: Any) -> Tuple[int, Tuple[int, ...]]:
    """Return dimensionality and shape for backend arrays/tensors.

    :param Any x: input array-like.

    :returns: ``(ndim, shape_tuple)``.
    :rtype: Tuple[int, Tuple[int, ...]]
    """
    shape = getattr(x, "shape", None)
    if shape is None:
        arr = _np.asarray(x)
        return arr.ndim, arr.shape

    if isinstance(shape, tuple):
        return len(shape), shape

    # TensorFlow TensorShape and similar objects.
    try:
        shape_tuple = tuple(int(d) for d in shape)
        return len(shape_tuple), shape_tuple
    except Exception:
        arr = _np.asarray(x)
        return arr.ndim, arr.shape


def split_columns(x: Any, columns: Sequence[int], *, keepdims: bool = False) -> Tuple[Any, ...]:
    """Extract selected columns from a 2D backend object.

    :param Any x: backend array/tensor/dataframe with shape ``(n, d)``.
    :param Sequence[int] columns: column indices to extract.
    :param bool keepdims: if ``True``, keep extracted columns as 2D objects.
        Defaults to ``False``.

    :returns: extracted columns in the same backend.
    :rtype: Tuple[Any, ...]
    """
    b = get_backend(x)
    arr = b.asarray(x)

    if b.name == "pandas":
        # Return Series to avoid DataFrame column-label alignment issues
        # during arithmetic operations between different selected columns.
        if keepdims:
            return tuple(arr.iloc[:, c] for c in columns)
        return tuple(arr.iloc[:, c] for c in columns)

    if b.name == "tensorflow":
        import tensorflow as tf

        if keepdims:
            return tuple(tf.expand_dims(arr[:, c], axis=1) for c in columns)
        return tuple(arr[:, c] for c in columns)

    if keepdims:
        return tuple(arr[:, [c]] for c in columns)
    return tuple(arr[:, c] for c in columns)


def concat_columns(cols: Sequence[Any], *, like: Any) -> Any:
    """Concatenate column objects along axis 1 using ``like`` backend.

    :param Sequence[Any] cols: column-like objects to concatenate.
    :param Any like: reference object used to infer backend.

    :returns: concatenated object in the backend of ``like``.
    :rtype: Any
    """
    b = get_backend(like)

    if b.name == "torch":
        import torch
        return torch.cat(cols, dim=1)

    if b.name == "tensorflow":
        import tensorflow as tf
        return tf.concat(cols, axis=1)

    if b.name == "jax":
        import jax.numpy as jnp
        return jnp.concatenate(cols, axis=1)

    if b.name == "pandas":
        import pandas as pd
        return pd.concat(cols, axis=1)

    return _np.concatenate(cols, axis=1)

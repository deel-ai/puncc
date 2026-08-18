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
"""
This define gestion of interaction with keras (and its backends) for the whole library.
"""
from functools import wraps
from typing import Callable
from deel.puncc.config import is_backend_set, set_backend
import sys
from packaging.version import Version

# Requires keras >= 3.3 for numpy backend support
_MIN_KERAS = (3, 3, 0)

_BACKEND_INFERENCE_ARG_NAMES = (
    "x",
    "x1",
    "a",
    "mask",
)

def infer_backend_from_var(x) -> str:
    module = type(x).__module__

    if module.startswith("numpy"):
        return "numpy"

    if module.startswith("torch"):
        return "torch"

    if module.startswith("tensorflow"):
        return "tensorflow"

    if module.startswith("jax"):# or module.startswith("jaxlib"):
        return "jax"

    raise TypeError(
        f"Cannot infer backend from object of type "
        f"{type(x).__module__}.{type(x).__qualname__}. "
        "Please use deel.puncc.config.set_backend(...) "
        "to specify the backend that should be used."
    )

class BackendManager():
    __slots__ = ("_keras")

    def __init__(self):
        self._keras = None

    @property
    def module(self):
        return self._keras

    def _load_keras(self):
        if self._keras is not None:
            return

        if "keras" in sys.modules:
            keras = sys.modules["keras"]
            check_keras_version(keras)
            if not is_backend_set():
                set_backend(keras.backend.backend())
            
            self._keras = keras
            return

        if not is_backend_set():
            raise RuntimeError(
                "PUNCC backend has not been initialized. "
                "Call deel.puncc.config.set_backend(...) first."
            )

        import keras
        check_keras_version(keras)
        self._keras = keras
        


    def __getattr__(self, name):
        self._load_keras()
        return getattr(self.module, name)

def get_x_arg(args, kwargs):
    if len(args) > 0:
        return args[0]
    for name in _BACKEND_INFERENCE_ARG_NAMES:
        if name in kwargs:
            return kwargs[name]
    return None

def check_keras_version(keras):
    if Version(keras.__version__) < Version("3.3.0"):
        raise RuntimeError(
            f"Keras {keras.__version__} detected. "
            f"This library requires Keras >= "
            f"{'.'.join(map(str, _MIN_KERAS))}. "
            "Upgrade with: pip install -U keras"
        )

def set_backend_on_first_call(f:Callable):
    @wraps(f)
    def _f(self, *args, **kwargs):
        if not is_backend_set() and "keras" not in sys.modules:
            x = get_x_arg(args, kwargs)
            if x is None:
                raise TypeError(
                    f"Cannot infer backend for function call {f.__name__!r} "
                    f"with kwargs {kwargs}. "
                    "Please use deel.puncc.config.set_backend(...) "
                    "to specify the backend that should be used."
                )
            backend = infer_backend_from_var(x)
            set_backend(backend)
        return f(self, *args, **kwargs)
    return _f

class BackendEstimationCallback():
    __slots__ = ("name","backend_manager")
    def __init__(self, name:str, backend_manager:BackendManager):
        self.name = name
        self.backend_manager = backend_manager

    @set_backend_on_first_call
    def __call__(self, *args, **kwargs):
        return getattr(self.backend_manager, self.name)(*args, **kwargs)
    
class RandomBackendManager(BackendManager):
    @property
    def module(self):
        return self._keras.random

class OpsBackendManager(BackendManager):
    @property
    def module(self):
        return self._keras.ops

    def __getattr__(self, name):
        if (
            self._keras is None
            and not is_backend_set()
            and "keras" not in sys.modules
        ):
            return BackendEstimationCallback(
                name=name,
                backend_manager=self,
            )

        return super().__getattr__(name)

    @set_backend_on_first_call
    def flatten(self, x):
        """
        Flatten a tensor to 1D.

        Backend-agnostic equivalent of np.flatten(x).

        Args:
            x (TensorLike): Input tensor.

        Returns:
            TensorLike: A 1D tensor containing the elements of x
        """
        return self.reshape(x, (-1,))

    @set_backend_on_first_call
    def _unique_sorted(self, x):
        x = self.sort(x)
        return self.concatenate([x[:1], x[1:][self.not_equal(x[1:], x[:-1])]])

    @set_backend_on_first_call
    def where_1d(self, mask):
        """Backend-agnostic: return 1D indices where mask is True (mask must be rank-1)."""
        idx = self.where(mask)

        # numpy/jax/torch: tuple of arrays
        if isinstance(idx, tuple):
            idx = idx[0]

        # tensorflow: (n, 1) for 1D mask
        if hasattr(idx, "shape") and len(idx.shape) == 2:
            idx = self.squeeze(idx, axis=-1)
        return self.reshape(idx, (-1,))

    @set_backend_on_first_call
    def where_nd(self, mask):
        """Backend-agnostic: return indices as a 2D tensor of shape (n_true, rank(mask))."""
        idx = self.where(mask)
        if isinstance(idx, tuple):
            return self.transpose(self.stack(idx, axis=0))
        return idx

    @set_backend_on_first_call
    def setdiff1d(self, a, b, assume_unique=False):
        """
        Find the set difference of two tensors.

        Return the unique values in a that are not in b.

        Backend-agnostic equivalent of np.setdiff1d(a, b).

        Args:
            a (TensorLike): Input tensor.
            b (TensorLike): Input comparison tensor.
        
        Returns:
            TensorLike: 1D tensor of values in a that are not in b.
        
        """
        # TODO : this implementation has suboptimal complexity. It should be improved later.
        # Ensure both are 1D tensors
        a = self.reshape(a, (-1,))
        b = self.reshape(b, (-1,))


        if not assume_unique:
            a = self._unique_sorted(a)
            b = self._unique_sorted(b)

        # For each element in a, check if it exists in b
        isin = self.any(self.expand_dims(a, -1) == b, axis=-1)
        mask = self.logical_not(isin)

        # Gather the elements where mask == True
        idx = self.where_1d(mask)
        return self.take(a, idx)

    @set_backend_on_first_call
    def weighted_quantile(self, x, q, weights=None, axis=None, keepdims=False):
        q = self.cast(q, x.dtype)
        weights = self.cast(weights, x.dtype) if weights is not None else None
        if weights is None or self.all(weights == 0):
            weights = self.ones_like(x)

        if axis is None:
            x = self.flatten(x)
            weights = self.flatten(weights)
            axis = 0

        q = self.convert_to_tensor(q)
        q = self.clip(q, 0.0, 1.0)

        weights = weights / self.sum(weights, axis=axis, keepdims=True)
        sorted_indices = self.argsort(x, axis=axis)
        sorted_cumsum_weights = self.cumsum(self.take_along_axis(weights, sorted_indices, axis=axis), axis=axis)
        idx = self.sum(sorted_cumsum_weights < q, axis=axis, keepdims=keepdims)
        sorted_a = self.take_along_axis(x, sorted_indices, axis=axis)
        res = self.take_along_axis(sorted_a, self.expand_dims(idx, axis=axis), axis=axis)
        return self.squeeze(res, axis=axis)



ops = OpsBackendManager()
random = RandomBackendManager()

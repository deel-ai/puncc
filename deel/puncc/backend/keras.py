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
Keras backend integration for PUNCC.

This module provides the backend-agnostic numerical interface used throughout PUNCC.
It exposes Keras 3 operations while delaying the import and initialization of Keras until a backend can be selected.

PUNCC relies on Keras 3 as a common numerical abstraction over the NumPy, PyTorch, TensorFlow, and JAX backends.
Since the Keras backend must normally be selected before Keras is imported, this module avoids importing Keras eagerly.

The module exposes two backend managers:
- ``ops``
    Provides access to :mod:`keras.ops` together with a small number
    of PUNCC-specific backend-agnostic operations such as
    :meth:`OpsBackendManager.weighted_quantile`,
    :meth:`OpsBackendManager.setdiff1d`, and
    :meth:`OpsBackendManager.where_1d`.

- ``random``
    Provides delayed access to :mod:`keras.random` using the same
    backend initialization mechanism.

The indirection implemented here is intentionally transparent to the rest of PUNCC: numerical code can use ``ops`` and ``random`` without importing a specific tensor framework or importing Keras directly.

Notes :
    Keras >= 3.3 is required. In particular, this version is required for support of the NumPy backend.

    Backend inference is intended as a convenience.
    Applications that use multiple numerical frameworks in the same process should explicitly configure the desired backend with :func:`deel.puncc.config.set_backend` before performing numerical operations.
"""
from functools import wraps
from types import ModuleType
from typing import Any, Callable

from deel.puncc.config import get_backend, is_backend_frozen, set_backend
import sys
from packaging.version import Version

from deel.puncc.typing import TensorLike

# Requires keras >= 3.3 for numpy backend support
_MIN_KERAS = (3, 3, 0)

# arg names for tensor valued parameters given to ops methods
# Used for backend inference in case of bad configuration
_BACKEND_INFERENCE_ARG_NAMES = (
    "x",
    "x1",
    "a",
    "mask",
)

class NoBackendSpecifiedError(RuntimeError):
    """
    Error raised when no numerical backend can be selected.

    This exception is raised when PUNCC requires a numerical operation but no backend has been explicitly configured and backend inference from the current runtime context is unsuccessful.
    """
    def __init__(self):
        super().__init__(
            "PUNCC backend has not been initialized and could not be infered from context. "
            "Call deel.puncc.config.set_backend(...) first."
        )

def infer_backend_from_tensor(x:TensorLike) -> str|None:
    """
    Infer a Keras backend from a tensor-like object.

    The backend is inferred from the Python module defining the type of x.
    NumPy arrays, PyTorch tensors, TensorFlow tensors, and JAX arrays are recognized.

    Args:
        x: Tensor-like object from which to infer the numerical backend.

    Returns:
        The inferred Keras backend name among ``"numpy"``, ``"torch"``, ``"tensorflow"``, and ``"jax"``.
        Returns ``None`` if the object cannot be associated with a supported backend.

    Note:
        This function only performs inference. It does not modify the
        PUNCC backend configuration.
    """
    module = type(x).__module__

    if module.startswith("numpy"):
        return "numpy"

    if module.startswith("torch"):
        return "torch"

    if module.startswith("tensorflow"):
        return "tensorflow"

    if module.startswith("jax"):# or module.startswith("jaxlib"):
        return "jax"
    return None

def infer_backend_from_modules()->str|None:
    """
    Infer a Keras backend from frameworks already imported.

    A non-NumPy backend is inferred only when exactly one of PyTorch, TensorFlow, or JAX is detected.
    NumPy is used as a fallback only when none of these frameworks is present.

    Returns:
        The inferred backend name, or ``None`` if no supported backend can be inferred unambiguously.

    Note:
        Explicit backend configuration is preferable in applications that import several numerical frameworks.
    """
    imported_modules = {
        name.split(".", 1)[0]
        for name in sys.modules
    }
    detected_backends:set[str] = set.intersection({"torch", "tensorflow", "jax"}, imported_modules)

    if "jaxlib" in imported_modules:
        detected_backends.add("jax")

    if len(detected_backends) == 1:
        return detected_backends.pop()
    elif len(detected_backends) == 0 and "numpy" in imported_modules:
        return "numpy"

    return None

class BackendManager():
    """
    Lazily load and proxy a Keras backend module.

    ``BackendManager`` delays importing Keras until an attribute of the managed module is actually required.
    This allows PUNCC to select a Keras backend before Keras initialization occurs.

    If Keras has already been imported, its active backend is reused when compatible with the current PUNCC configuration.

    Subclasses can override ``module`` to expose a Keras submodule such as ``keras.ops`` or ``keras.random``.
    """
    __slots__ = ("_keras")

    def __init__(self):
        self._keras = None

    @property
    def module(self):
        return self._keras

    def _load_keras(self):
        """
        Load Keras after ensuring that its backend is configured.
        """
        if self._keras is not None:
            return

        if "keras" in sys.modules:
            keras = sys.modules["keras"]
            check_keras_version(keras)
            if not is_backend_frozen():
                set_backend(keras.backend.backend())
            elif get_backend() != keras.backend.backend():
                raise RuntimeError(
                    f"Keras backend ({keras.backend.backend()}) does not match the frozen backend ({get_backend()})."
                )
            self._keras = keras
            return

        if not is_backend_frozen():
            raise NoBackendSpecifiedError()

        import keras
        check_keras_version(keras)
        self._keras = keras

    def __getattr__(self, name:str):
        self._load_keras()
        return getattr(self.module, name)

def get_tensor_arg(args:Any, kwargs:Any):
    """
    Extract the tensor argument used for backend inference.

    Positional arguments take precedence. If no positional argument is available, a small set of conventional tensor parameter names is searched in ``kwargs``.

    Args:
        args: Positional arguments passed to a numerical operation.
        kwargs: Keyword arguments passed to a numerical operation.

    Returns:
        A candidate tensor argument, or ``None`` if none can be found.
    """
    if len(args) > 0:
        return args[0]
    for name in _BACKEND_INFERENCE_ARG_NAMES:
        if name in kwargs:
            return kwargs[name]
    return None

def check_keras_version(keras:ModuleType):
    """
    Check that the installed Keras version is supported.

    Args:
        keras: Imported Keras module.

    Raises:
        RuntimeError: If the installed Keras version is older than the minimum version required by PUNCC.
    """
    if Version(keras.__version__) < Version("3.3.0"):
        raise RuntimeError(
            f"Keras {keras.__version__} detected. "
            f"This library requires Keras >= "
            f"{'.'.join(map(str, _MIN_KERAS))}. "
            "Upgrade with: pip install -U keras"
        )

def set_backend_on_first_call(f:Callable[..., Any])->Callable[..., Any]:
    """
    Ensure that a backend is selected before an operation runs.

    If the backend has already been configured, or if Keras has already been imported, the operation is executed directly.

    Otherwise, PUNCC first attempts to infer the backend from the operation's tensor argument.
    If this fails, it falls back to inspecting numerical frameworks already imported in the process.

    Args:
        f: Backend-dependent operation to decorate.

    Returns:
        A wrapped callable that ensures backend initialization before executing ``f``.

    Raises:
        NoBackendSpecifiedError: If no backend was configured and backend inference fails.
    """
    @wraps(f)
    def _f(self:object, *args:Any, **kwargs:Any):
        if not is_backend_frozen() and "keras" not in sys.modules:
            x = get_tensor_arg(args, kwargs)
            backend = None
            if x is not None:
                backend = infer_backend_from_tensor(x)
            if backend is None:
                backend = infer_backend_from_modules()
            if backend is None:
                raise NoBackendSpecifiedError()
            set_backend(backend)
        return f(self, *args, **kwargs)
    return _f

class _DeferredBackendOperation():
    """
    Proxy an operation until the numerical backend is known.

    Instances are returned when an ``ops`` attribute is accessed before Keras can safely be initialized.
    The underlying Keras operation is resolved when the proxy is called.
    """
    __slots__ = ("name","backend_manager")
    def __init__(self, name:str, backend_manager:BackendManager):
        self.name = name
        self.backend_manager = backend_manager

    @set_backend_on_first_call
    def __call__(self, *args:Any, **kwargs:Any):
        return getattr(self.backend_manager, self.name)(*args, **kwargs)
    
class RandomBackendManager(BackendManager):
    """
    Provide lazily initialized access to ``keras.random``.
    """
    @property
    def module(self):
        if self._keras is None:
            raise RuntimeError("Keras has not been loaded.")
        return self._keras.random

class OpsBackendManager(BackendManager):
    """
    Provide backend-agnostic tensor operations for PUNCC.

    The manager exposes operations from ``keras.ops`` through lazy attribute resolution,
    allowing PUNCC to postpone Keras initialization until the numerical backend is known.

    In addition to native Keras operations, this class implements a small set of backend-independent utilities required by PUNCC when no directly suitable ``keras.ops`` equivalent is available.

    Attributes:
        inf: Positive infinity.
        ninf: Negative infinity.
    """
    inf = float("inf")
    ninf = float("-inf")

    @property
    def tensor_type(self) -> type[TensorLike]:
        """
        Returns:
            The concrete tensor class associated with the currently active Keras backend.

        Raises:
            NoBackendSpecifiedError: If no backend has been selected yet.
        """
        if get_backend() is None:
            raise NoBackendSpecifiedError()
        return type(self.array(0.0))

    @property
    def module(self):
        if self._keras is None:
            raise RuntimeError("Keras has not been loaded.")
        return self._keras.ops

    def __getattr__(self, name:str):
        if (
            self._keras is None
            and not is_backend_frozen()
            and "keras" not in sys.modules
        ):
            return _DeferredBackendOperation(
                name=name,
                backend_manager=self,
            )

        return super().__getattr__(name)

    @set_backend_on_first_call
    def flatten(self, x:TensorLike):
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
    def _unique_sorted(self, x:TensorLike):
        x = self.sort(x)
        return self.concatenate([x[:1], x[1:][self.not_equal(x[1:], x[:-1])]])

    @set_backend_on_first_call
    def where_1d(self, mask:TensorLike):
        """
        Return indices of true values in a one-dimensional mask.

        This helper normalizes the different output conventions of ``keras.ops.where`` across supported backends.

        Args:
            mask: One-dimensional boolean tensor.

        Returns:
            A one-dimensional tensor containing the indices for which ``mask`` is true.
        """
        idx = self.where(mask)

        # numpy/jax/torch: tuple of arrays
        if isinstance(idx, tuple):
            idx = idx[0]

        # tensorflow: (n, 1) for 1D mask
        if hasattr(idx, "shape") and len(idx.shape) == 2:
            idx = self.squeeze(idx, axis=-1)
        return self.reshape(idx, (-1,))

    @set_backend_on_first_call
    def where_nd(self, mask:TensorLike):
        """
        Return coordinates of true values in a boolean tensor.

        Args:
            mask: Boolean tensor of arbitrary rank.

        Returns:
            A two-dimensional tensor of shape ``(n_true, rank(mask))``, where each row contains the coordinates of one true element.
        """
        idx = self.where(mask)
        if isinstance(idx, tuple):
            return self.transpose(self.stack(idx, axis=0))
        return idx

    @set_backend_on_first_call
    def setdiff1d(self, a:TensorLike, b:TensorLike, assume_unique:bool=False):
        """
        Return values present in ``a`` and absent from ``b``.

        Both inputs are flattened before the set difference is computed.
        Unless ``assume_unique`` is true, duplicate values are removed
        before comparison.

        Backend-agnostic equivalent of np.setdiff1d(a, b).

        Args:
            a: Input tensor.
            b: Tensor containing values to remove from ``a``.
            assume_unique: Whether to assume that both inputs already
                contain unique values.

        Returns:
            A one-dimensional tensor containing values from ``a`` that are
            not present in ``b``.
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
    def weighted_quantile(self, 
                          x:TensorLike,
                          q:float|TensorLike,
                          weights:TensorLike=None,
                          axis:int|None=None,
                          keepdims:bool=False):
        """
        Compute weighted empirical quantiles.

        Values are sorted along the requested axis together with their associated weights.
        The returned quantile is the first sorted value whose normalized cumulative weight is greater than or equal to the requested quantile level.

        If ``weights`` is ``None``, all observations receive equal weight.

        Args:
            x: Input values.
            q: Quantile level or tensor of quantile levels.
                Values are clipped to ``[0, 1]``.
            weights: Non-negative observation weights.
                If one-dimensional and ``axis`` is specified, weights are interpreted along that axis and broadcast over the remaining dimensions.
                If ``axis`` is ``None``, weights must have the same shape as ``x``.
            axis: Axis along which to compute the quantile. If ``None``, both ``x`` and ``weights`` are flattened.
            keepdims: Whether to retain the reduced axis with length one.

        Returns:
            The weighted empirical quantile values.

        Raises:
            ValueError: If weights contain negative values, if all weights are zero, or if their shape is incompatible with ``x``.

        Note:
            This function computes an inverse weighted empirical cumulative distribution function and does not interpolate between adjacent observations.
        """
        q = self.convert_to_tensor(q)

        if weights is None:
            weights = self.ones_like(x)
        else:
            if self.any(weights < 0):
                raise ValueError("Weights must be non-negative.")
            if self.all(weights == 0):
                raise ValueError("All weights are zero. At least one weight must be positive.")

            weights = self.cast(weights, x.dtype)

            if axis is None:
                if tuple(weights.shape) != tuple(x.shape):
                    raise ValueError("Weights must have the same shape as x when axis is None.")
            elif len(weights.shape) == 1:
                if weights.shape[0] != x.shape[axis]:
                    raise ValueError(
                        "1D weights must have the same length as "
                        "x along the quantile axis."
                    )

                shape = [1] * len(x.shape)
                shape[axis] = x.shape[axis]

                weights = self.reshape(weights, shape)
                weights = self.broadcast_to(weights, self.shape(x))

        if axis is None:
            x = self.flatten(x)
            weights = self.flatten(weights)
            axis = 0

        q = self.convert_to_tensor(q)
        q = self.clip(q, 0.0, 1.0)

        weights = weights / self.sum(weights, axis=axis, keepdims=True)
        sorted_indices = self.argsort(x, axis=axis)
        sorted_cumsum_weights = self.cumsum(self.take_along_axis(weights, sorted_indices, axis=axis), axis=axis)
        idx = self.sum(sorted_cumsum_weights < q, axis=axis, keepdims=True)
        idx = self.minimum(idx,self.shape(x)[axis] - 1)
        sorted_x = self.take_along_axis(x, sorted_indices, axis=axis)
        res = self.take_along_axis(sorted_x, idx, axis=axis)
        if not keepdims:
            res = self.squeeze(res,axis=axis)
        return res

ops = OpsBackendManager()
random = RandomBackendManager()

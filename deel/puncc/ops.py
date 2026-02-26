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
Backend-agnostic operations built on top of Keras ops.

This module re-exports `keras.ops` and provides a small set of additional
utilities used throughout the library.
"""
# Keep this import to ensure all default ops are available
from deel.puncc._keras import keras
from keras.ops import *
import warnings
import numpy as np

ops = keras.ops

# Other definitions dynamically added to ops :
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    inf = ops.convert_to_tensor(np.inf)
    ninf = ops.convert_to_tensor(-np.inf)

def flatten(x):
    """
    Flatten a tensor to 1D.

    Backend-agnostic equivalent of np.flatten(x).

    Args:
        x (TensorLike): Input tensor.

    Returns:
        TensorLike: A 1D tensor containing the elements of x
    """
    return ops.reshape(x, (-1,))


def _unique_sorted(x):
    x = ops.sort(x)
    return ops.concatenate([x[:1], x[1:][ops.not_equal(x[1:], x[:-1])]])

def where_1d(mask):
    """Backend-agnostic: return 1D indices where mask is True (mask must be rank-1)."""
    idx = ops.where(mask)

    # numpy/jax/torch: tuple of arrays
    if isinstance(idx, tuple):
        idx = idx[0]

    # tensorflow: (n, 1) for 1D mask
    if hasattr(idx, "shape") and len(idx.shape) == 2:
        idx = ops.squeeze(idx, axis=-1)
    return ops.reshape(idx, (-1,))

def where_nd(mask):
    """Backend-agnostic: return indices as a 2D tensor of shape (n_true, rank(mask))."""
    idx = ops.where(mask)
    if isinstance(idx, tuple):
        return ops.transpose(ops.stack(idx, axis=0))
    return idx

def setdiff1d(a, b, assume_unique=False):
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
    a = ops.reshape(a, (-1,))
    b = ops.reshape(b, (-1,))


    if not assume_unique:
        a = _unique_sorted(a)
        b = _unique_sorted(b)

    # For each element in a, check if it exists in b
    isin = ops.any(ops.expand_dims(a, -1) == b, axis=-1)
    mask = ops.logical_not(isin)

    # Gather the elements where mask == True
    idx = where_1d(mask)
    return ops.take(a, idx)

def weighted_quantile(x, q, weights=None, axis=None, keepdims=False):
    if weights is None or ops.all(weights == 0):
        weights = ops.ones_like(x)

    if axis is None:
        x = flatten(x)
        weights = flatten(weights)
        axis = 0
    
    q = ops.convert_to_tensor(q)
    q = ops.clip(q, 0.0, 1.0)

    weights = weights / ops.sum(weights, axis=axis, keepdims=True)
    sorted_indices = ops.argsort(x, axis=axis)
    sorted_cumsum_weights = ops.cumsum(ops.take_along_axis(weights, sorted_indices, axis=axis), axis=axis)
    idx = ops.sum(sorted_cumsum_weights < q, axis=axis, keepdims=keepdims)
    sorted_a = ops.take_along_axis(x, sorted_indices, axis=axis)
    res = ops.take_along_axis(sorted_a, ops.expand_dims(idx, axis=axis), axis=axis)
    return ops.squeeze(res, axis=axis)

patchs = ["inf", "ninf", "flatten", "where_1d", "where_nd", "setdiff1d", "weighted_quantile"]
__all__ = [*dir(ops), *patchs]

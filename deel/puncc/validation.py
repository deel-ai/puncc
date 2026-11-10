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
Tools for data validation and shape checking, used by conformal prediction methods.
"""

from typing import Any

from deel.puncc.backend.keras import ops
from deel.puncc.typing import TensorLike


def check_same_shape(
    **tensors: Any,
) -> None:
    """Require tensors to have exactly the same shape."""
    if not tensors:
        return
    tensor_iter = iter(tensors.items())
    ref_name, ref_tensor = next(tensor_iter)
    ref_shape = tuple(ops.shape(ref_tensor))

    for name, tensor in tensor_iter:
        shape = tuple(ops.shape(tensor))
        if shape != ref_shape:
            raise ValueError(
                f"All tensors must have the same shape, "
                f"but {ref_name} has shape {ref_shape} and {name} has shape {shape}."
            )


def as_scalar_vector(
    value: TensorLike,
    *,
    n_samples: int | None = None,
    allow_scalar: bool = False,
) -> TensorLike:
    """
    Normalize scalar regression values to a one-dimensional tensor.

    Accept shapes (n,) and (n, 1), or a scalar when explicitly allowed.
    Reject multi-output values.
    """
    value = ops.convert_to_tensor(value)
    shape = tuple(ops.shape(value))

    if allow_scalar and shape == ():
        value = ops.reshape(value, (1,))
    elif len(shape) == 2 and shape[1] == 1:
        value = ops.squeeze(value, axis=-1)
    elif len(shape) != 1:
        raise ValueError(f"tensor must have shape (n,) or (n, 1), " f"got {shape}.")

    if n_samples is not None and len(value) != n_samples:
        raise ValueError(
            f"tensor must contain {n_samples} samples, " f"got {len(value)}."
        )

    return value

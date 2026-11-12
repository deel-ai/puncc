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
Bounding-box prediction set functions for conformal object detection.
"""


from typing import Any
from deel.puncc.typing import TensorLike, PredSetFunction
from deel.puncc import ops


def _constant_bbox(y_pred: TensorLike, quantile: TensorLike) -> TensorLike:
    direction = ops.convert_to_tensor([-1.0, -1.0, 1.0, 1.0], dtype=y_pred.dtype)

    delta = quantile * direction

    outer = y_pred + delta
    inner = y_pred - delta

    return ops.stack([inner, outer], axis=-1)


def constant_bbox() -> PredSetFunction:
    """
    Return a function constructing coordinate-wise bounding-box bounds.

    The returned function applies a conformal threshold to each bounding-box coordinate.
    For positive thresholds, minimum coordinates are shifted outward in the negative direction,
    and maximum coordinates outward in the positive direction.

    Returns:
        A function accepting bounding boxes with shape ``(..., 4)`` and thresholds,
        returning a tensor with shape ``(..., 4, 2)``.

    Notes:
        The last dimension contains the inner and outer coordinate values, in that order.
        It does not necessarily represent numerically sorted lower and upper bounds.
    """
    return _constant_bbox


def _scaled_bbox(y_pred: TensorLike, quantile: float | TensorLike) -> Any:
    dx = ops.abs(y_pred[..., 2] - y_pred[..., 0])
    dy = ops.abs(y_pred[..., 3] - y_pred[..., 1])

    scale = ops.stack([dx, dy, dx, dy], axis=-1)

    direction = ops.convert_to_tensor([-1.0, -1.0, 1.0, 1.0], dtype=y_pred.dtype)

    delta = quantile * scale * direction

    outer = y_pred + delta
    inner = y_pred - delta

    return ops.stack([inner, outer], axis=-1)


def scaled_bbox() -> PredSetFunction:
    """
    Return a function constructing size-scaled bounding-box bounds.

    Returns:
        A function accepting bounding boxes with shape ``(..., 4)`` and thresholds,
        returning coordinate bounds with shape ``(..., 4, 2)``.

    Notes:
        The last dimension contains inner and outer coordinate values, in that order.
    """
    return _scaled_bbox

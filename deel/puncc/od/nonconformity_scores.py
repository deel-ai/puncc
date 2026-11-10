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
Nonconformity score functions for conformal object detection.
"""

from deel.puncc.typing import TensorLike, NCScoreFunction
from deel.puncc import ops
from deel.puncc.validation import check_same_shape


def _boxwise_difference(y_pred: TensorLike, y_true: TensorLike) -> TensorLike:
    check_same_shape(y_pred=y_pred, y_true=y_true)
    return ops.stack(
        [
            y_pred[..., 0] - y_true[..., 0],  # xmin
            y_pred[..., 1] - y_true[..., 1],  # ymin
            y_true[..., 2] - y_pred[..., 2],  # xmax
            y_true[..., 3] - y_pred[..., 3],  # ymax
        ],
        axis=-1,
    )


def boxwise_difference() -> NCScoreFunction:
    """
    Return a coordinate-wise bounding-box nonconformity score function.

    Boxes follow the ``(xmin, ymin, xmax, ymax)`` convention.
    Signed coordinate residuals are defined as
        [
            pred_xmin - true_xmin,
            pred_ymin - true_ymin,
            true_xmax - pred_xmax,
            true_ymax - pred_ymax,
        ]

    Returns:
        A function accepting predicted and target boxes with a trailing dimension of size
        four and returning four signed coordinate scores per box.

    Notes:
        Positive scores indicate extensions required for the predicted box to contain the
        corresponding target coordinate.
    """
    return _boxwise_difference


def _scaled_bbox_difference(y_pred: TensorLike, y_true: TensorLike) -> TensorLike:
    x_min, y_min, x_max, y_max = ops.split(y_pred, 4, axis=1)
    dx = ops.abs(x_max - x_min)
    dy = ops.abs(y_max - y_min)
    diff = _boxwise_difference(y_pred, y_true)
    return diff / ops.hstack([dx, dy, dx, dy])


def scaled_bbox_difference() -> NCScoreFunction:
    """
    Return a size-normalized bounding-box nonconformity score function.

    Returns:
        A function accepting predicted and target boxes of shape ``(n, 4)`` and returning normalized
        scores of shape ``(n, 4)``.
    """
    return _scaled_bbox_difference

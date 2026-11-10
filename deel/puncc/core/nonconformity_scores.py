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
Simple nonconformity scores shared across prediction tasks.
"""

from deel.puncc.typing import TensorLike, NCScoreFunction
from deel.puncc import ops
from deel.puncc.validation import check_same_shape


def _difference(y_pred: TensorLike, y_true: TensorLike) -> TensorLike:
    check_same_shape(y_pred=y_pred, y_true=y_true)
    return y_pred - y_true


def difference() -> NCScoreFunction:
    """
    Return a signed residual nonconformity score function.

    The returned function computes ``y_pred - y_true`` elementwise.
    Both inputs must have identical shapes.

    Returns:
        A function mapping predictions and targets to signed residuals
        with the same shape as its inputs.
    """
    return _difference


def _absolute_difference(y_pred: TensorLike, y_true: TensorLike) -> TensorLike:
    check_same_shape(y_pred=y_pred, y_true=y_true)
    return ops.abs(y_pred - y_true)


def absolute_difference() -> NCScoreFunction:
    """
    Return an absolute residual nonconformity score function.

    The returned function computes ``abs(y_pred - y_true)``
    elementwise. Both inputs must have identical shapes.

    Returns:
        A function mapping predictions and targets to non-negative
        absolute residuals with the same shape as its inputs.
    """
    return _absolute_difference

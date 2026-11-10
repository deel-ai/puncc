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
Nonconformity score functions for conformal regression.
"""

from deel.puncc.typing import TensorLike, NCScoreFunction
from deel.puncc import ops
from deel.puncc.validation import check_same_shape


def _cqr_score(y_pred: TensorLike, y_true: TensorLike) -> TensorLike:
    pred_shape = tuple(ops.shape(y_pred))
    if len(pred_shape) not in (2, 3) or pred_shape[-1] != 2:
        raise ValueError(
            "CQR predictions must have shape (n, 2) or (n, d, 2), " f"got {pred_shape}."
        )

    lower_pred = ops.take(y_pred, 0, axis=-1)
    upper_pred = ops.take(y_pred, 1, axis=-1)
    check_same_shape(lower_pred=lower_pred, upper_pred=upper_pred, y_true=y_true)
    return ops.maximum(lower_pred - y_true, y_true - upper_pred)


def cqr_score() -> NCScoreFunction:
    """
    Return a Conformalized Quantile Regression (CQR) score function.

    For predicted lower and upper bounds, the score is defined as::
        max(lower_bound - y_true, y_true - upper_bound)

    The returned function expects predictions with shape ``(n, 2)``
    or ``(n, d, 2)`` and corresponding targets with shape ``(n,)``
    or ``(n, d)``.

    Returns:
        A function computing one score per target component.
    """
    return _cqr_score

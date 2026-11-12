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
Prediction interval functions for conformal regression.
"""
from deel.puncc.typing import TensorLike, PredSetFunction
from deel.puncc import ops


def _constant_interval(y_pred: TensorLike, quantile: float | TensorLike) -> TensorLike:
    lower_bounds = y_pred - quantile
    upper_bounds = y_pred + quantile
    return ops.stack([lower_bounds, upper_bounds], axis=-1)


def constant_interval() -> PredSetFunction:
    """
    Return a function constructing symmetric prediction intervals.

    Given point predictions and a nonconformity threshold ``q``,
    the returned function produces intervals ``[y_pred - q,
    y_pred + q]``.

    Returns:
        A function returning prediction intervals
    """
    return _constant_interval


def scaled_interval(eps: float = 1e-12) -> PredSetFunction:
    """
    Return a function constructing locally scaled prediction intervals.

    The returned function expects predictions containing a location estimate and a dispersion estimate along
    the last axis.
    Interval offsets are obtained by multiplying the conformal threshold by the predicted dispersion
    plus ``eps``.

    Args:
        eps:
            Additive stabilization term applied to the predicted dispersion.
            Defaults to ``1e-12``.

    Returns:
        A function returning lower and upper interval bounds along the last axis.
    """

    def _scaled_interval(
        y_pred: TensorLike, quantile: float | TensorLike
    ) -> TensorLike:
        mean_pred = ops.take(y_pred, 0, axis=-1)
        var_pred = ops.take(y_pred, 1, axis=-1)
        nonneg = var_pred + eps > 0

        y_low = ops.where(
            nonneg,
            mean_pred - quantile * (var_pred + eps),
            ops.ninf,
        )

        y_high = ops.where(
            nonneg,
            mean_pred + quantile * (var_pred + eps),
            ops.inf,
        )
        return ops.stack([y_low, y_high], axis=-1)

    return _scaled_interval


def _cqr_interval(y_pred: TensorLike, quantile: float | TensorLike) -> TensorLike:
    lower_pred = ops.take(y_pred, 0, axis=-1)
    upper_pred = ops.take(y_pred, 1, axis=-1)
    y_low = lower_pred - quantile
    y_high = upper_pred + quantile
    return ops.stack([y_low, y_high], axis=-1)


def cqr_interval() -> PredSetFunction:
    """
    Return a function constructing conformalized quantile intervals.

    Given predicted lower and upper quantiles and a calibration threshold ``q``,
    the returned function computes::

        [lower_pred - q, upper_pred + q]

    Returns:
        A function returning intervals with the lower and upper bounds along the last axis.
    """
    return _cqr_interval

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
Prediction set construction functions for conformal classification.
"""
from deel.puncc.typing import TensorLike, PredSetFunction
from deel.puncc import ops
from deel.puncc.backend.keras import random


def _lac_set(y_pred: TensorLike, quantile: float | TensorLike) -> list[TensorLike]:
    if ops.ndim(quantile) == 1:
        quantile = ops.expand_dims(quantile, axis=-1)

    mask = y_pred >= (1 - quantile)
    n_samples = len(y_pred)
    return [ops.where_1d(mask[i]) for i in range(n_samples)]


def lac_set() -> PredSetFunction:
    """
    Return a LAC prediction set function.

    The returned function includes each class whose predicted probability is at least ``1 - quantile``.

    Returns:
        A function accepting class probabilities of shape ``(n, K)`` and a calibration threshold,
        and returning one tensor of selected class indices per sample.

    Notes:
        Prediction sets may be empty.
    """
    return _lac_set


def raps_set(lambd: float = 0, k_reg: int = 1, rand: bool = True) -> PredSetFunction:
    """
    Return a Regularized Adaptive Prediction Sets (RAPS) function.

    Classes are ordered by decreasing predicted probability, with random tie-breaking.
    Their cumulative probabilities are augmented with a rank-dependent regularization penalty.

    Args:
        lambd:
            Non-negative regularization strength.
        k_reg:
            Rank threshold beyond which regularization is applied.
        rand:
            Whether to randomize inclusion of the boundary class.
            When False, boundary inclusion is deterministic.

    Returns:
        A function accepting class probabilities of shape ``(n, K)`` and a calibration threshold, and
        returning a list of tensors containing the selected class indices for each sample.

    Raises:
        ValueError:
            If ``lambd`` or ``k_reg`` is negative.

    Notes:
        Returned class indices follow decreasing predicted probability rather than their original class order.
        Ties are randomly broken regardless of ``rand``.
    """
    # TODO : I think this implementation is clearly suboptimal, see if it can be improved
    if lambd < 0:
        raise ValueError(f"`lambd` must be >= 0, got {lambd}")
    if k_reg < 0:
        raise ValueError(f"`k_reg` must be >= 0, got {k_reg}")

    def _raps_set(y_pred: TensorLike, quantile: float | TensorLike) -> list[TensorLike]:
        # Shuffle the predictions to break ties randomly
        tie_break = random.uniform(ops.shape(y_pred), dtype=y_pred.dtype)
        tie_order = ops.argsort(tie_break, axis=-1)
        shuffled_p = ops.take_along_axis(y_pred, tie_order, axis=-1)
        order = ops.argsort(-shuffled_p, axis=-1)
        sorted_index = ops.take_along_axis(tie_order, order, axis=-1)
        sorted_p = ops.take_along_axis(y_pred, sorted_index, axis=-1)

        cs = ops.cumsum(sorted_p, axis=-1)

        K = ops.shape(y_pred)[-1]
        ranks = ops.arange(1, K + 1, dtype=cs.dtype)
        penalty = lambd * ops.maximum(ranks - k_reg, 0)
        penal_cs = cs + penalty

        quantile = ops.reshape(ops.convert_to_tensor(quantile), (-1,))
        index_limit = ops.sum(penal_cs <= quantile[:, None], axis=-1) + 1

        index_limit = ops.minimum(index_limit, K)

        if rand:
            u = random.uniform(ops.shape(index_limit))

            last_pos = ops.maximum(index_limit - 1, 0)
            last_pos_exp = ops.expand_dims(last_pos, axis=-1)

            cs_at_last = ops.take_along_axis(cs, last_pos_exp, axis=-1)[..., 0]
            p_at_last = ops.take_along_axis(sorted_p, last_pos_exp, axis=-1)[..., 0]

            reg_at_L = lambd * ops.maximum(ops.cast(index_limit, cs.dtype) - k_reg, 0)

            v = (quantile - cs_at_last - reg_at_L + p_at_last) / p_at_last
            v = ops.clip(v, 0, 1)
            exclude_last = ops.where(v <= u, 1, 0)
            index_limit = ops.maximum(index_limit - exclude_last, 0)
        return [p[:lim] for p, lim in zip(sorted_index, index_limit)]

    return _raps_set


def aps_set(rand: bool = True) -> PredSetFunction:
    """
    Return an Adaptive Prediction Sets (APS) prediction set function.
    APS is equivalent to RAPS with zero regularization.

    Args:
        rand:
            Whether to randomize inclusion of the boundary class.
            Defaults to True.

    Returns:
        A function returning one tensor of selected class indices per sample.
    """
    return raps_set(lambd=0, k_reg=1, rand=rand)

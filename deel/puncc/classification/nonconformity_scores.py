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
Nonconformity score functions for conformal classification.
"""
from deel.puncc.typing import TensorLike, NCScoreFunction
from deel.puncc import ops
from deel.puncc.backend.keras import random


def _lac_score(y_pred: TensorLike, y_true: TensorLike) -> TensorLike:
    true_scores = ops.take_along_axis(y_pred, y_true[..., None], axis=-1)
    return 1 - ops.squeeze(
        true_scores,
        axis=-1,
    )


def lac_score() -> NCScoreFunction:
    """
    Return a LAC nonconformity score function.

    The returned function computes ``1 - p_true``, where ``p_true`` is the predicted probability of the true class.

    It expects class probabilities with shape ``(n, K)`` and integer class labels with shape ``(n,)``.

    Returns:
        A function returning one nonconformity score per sample.
    """
    return _lac_score


def raps_score(lambd: float = 0, k_reg: int = 1, rand: bool = True) -> NCScoreFunction:
    """
    Return a Regularized Adaptive Prediction Sets (RAPS) score function.

    Classes are ranked by decreasing predicted probability.
    The score combines the cumulative probability of classes preceding the true
    class, a contribution from its probability, and a rank-dependent regularization penalty.

    Args:
        lambd:
            Non-negative regularization strength. Defaults to zero.
        k_reg:
            Rank threshold beyond which the regularization penalty
            is applied. Defaults to one.
        rand:
            Whether to multiply the true-class probability by a
            uniform random variable. When False, its full probability
            is included.

    Returns:
        A function accepting class probabilities of shape ``(n, K)``
        and integer class labels of shape ``(n,)``, returning one
        score per sample.

    Raises:
        ValueError:
            If ``lambd`` or ``k_reg`` is negative.

    Notes:
        Equal class probabilities are randomly ordered regardless of ``rand``.
    """
    if lambd < 0:
        raise ValueError(f"`lambd` must be >= 0, got {lambd}")
    if k_reg < 0:
        raise ValueError(f"`k_reg` must be >= 0, got {k_reg}")

    def _raps_score(y_pred: TensorLike, y_true: TensorLike) -> TensorLike:
        # true proba
        true_p = ops.squeeze(
            ops.take_along_axis(y_pred, y_true[..., None], axis=-1), axis=-1
        )

        # randomization for cases where multiple classes have the same probability as the true class
        tie_break = random.uniform(ops.shape(y_pred), dtype=y_pred.dtype)
        true_tie_break = ops.squeeze(
            ops.take_along_axis(tie_break, y_true[..., None], axis=-1), axis=-1
        )

        before_true = ops.logical_or(
            y_pred > true_p[..., None],
            ops.logical_and(
                y_pred == true_p[..., None],
                tie_break < true_tie_break[..., None],
            ),
        )
        rho = ops.sum(ops.where(before_true, y_pred, 0), axis=-1)
        rank = ops.sum(ops.cast(before_true, "int32"), axis=-1) + 1
        regul = lambd * ops.maximum(ops.cast(rank, true_p.dtype) - k_reg, 0)
        if rand:
            u = random.uniform(ops.shape(true_p))
        else:
            u = ops.ones_like(true_p)
            # or u = ops.zero_like(true_p) ?
        return rho + u * true_p + regul

    return _raps_score


def aps_score(rand: bool = True) -> NCScoreFunction:
    """
    Return an Adaptive Prediction Sets (APS) score function.
    APS is equivalent to RAPS with zero regularization.

    Args:
        rand:
            Whether to randomize the true-class probability
            contribution. Defaults to True.

    Returns:
        An APS nonconformity score function.

    Notes:
        Equal class probabilities are randomly ordered even when ``rand`` is False.
    """
    return raps_score(lambd=0, k_reg=1, rand=rand)

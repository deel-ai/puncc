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
This module provides prediction sets for conformal prediction. To be used when
building a ConformalPredictor.
"""
from typing import Any
from deel.puncc.api.utils import logit_normalization_check
from deel.puncc.typing import TensorLike, PredSetFunction
from deel.puncc import ops
from deel.puncc._keras import random


def _constant_interval(y_pred:TensorLike, quantile:float|TensorLike) -> Any:
    lower_bounds = y_pred - quantile
    upper_bounds = y_pred + quantile
    return ops.stack([lower_bounds, upper_bounds], axis=-1)

def constant_interval()->PredSetFunction:
    return _constant_interval

def scaled_interval(eps:float=1e-12)->PredSetFunction:
    def _scaled_interval(y_pred:TensorLike, quantile:float|TensorLike) -> Any:
        mean_pred = ops.take(y_pred, 0, axis=-1)
        var_pred = ops.take(y_pred, 1, axis=-1)
        
        y_low = ops.zeros_like(mean_pred)
        y_high = ops.zeros_like(mean_pred)

        nonneg = var_pred + eps > 0

        y_low[nonneg] = mean_pred[nonneg] - quantile * (var_pred[nonneg] + eps)
        y_high[nonneg] = mean_pred[nonneg] + quantile * (var_pred[nonneg] + eps)
        y_low[~nonneg] = ops.ninf
        y_high[~nonneg] = ops.inf
        return ops.stack([y_low, y_high], axis=-1)
    return _scaled_interval

def _cqr_interval(y_pred:TensorLike, quantile:float|TensorLike) -> Any:
    lower_pred = y_pred[:, 0]
    upper_pred = y_pred[:, 1]
    y_low = lower_pred - quantile
    y_high = upper_pred + quantile
    return ops.stack([y_low, y_high], axis=-1)

def cqr_interval()->PredSetFunction:
    return _cqr_interval

def _lac_set(y_pred:TensorLike, quantile:float|TensorLike) -> Any:
    # FIXME : see the [-1] indexation for cases where y_pred.ndim > 2, change the use of .where
    return [ops.where(pred >= 1 - quantile)[-1] for pred in y_pred]

def lac_set()->PredSetFunction:
    return _lac_set

def raps_set(lambd:float=0, k_reg:int=1, rand:bool=False)->PredSetFunction:
    # TODO : I think this implementation is clearly suboptimal, see if it can be improved
    if lambd < 0:
        raise ValueError(f"`lambd` must be >= 0, got {lambd}")
    if k_reg < 0:
        raise ValueError(f"`k_reg` must be >= 0, got {k_reg}")
    def _raps_set(y_pred:TensorLike, quantile:float|TensorLike) -> Any:
        sorted_index = ops.argsort(-y_pred, axis=-1)
        sorted_p = ops.take_along_axis(y_pred, sorted_index, axis=-1)
        cs = ops.cumsum(sorted_p, axis=-1)

        # FIXME : lol, i completely forgot to add the regularization term here
        K = ops.shape(y_pred)[-1]
        ranks = ops.arange(1, K + 1, dtype=cs.dtype)
        penalty = lambd * ops.maximum(ranks - k_reg, 0)
        penal_cs = cs + penalty


        index_limit = ops.sum(penal_cs < quantile, axis=-1) + 1
        index_limit = ops.minimum(index_limit, K)

        if rand:
            u = random.uniform(ops.shape(index_limit))

            last_pos = ops.maximum(index_limit - 1, 0)
            last_pos_exp = ops.expand_dims(last_pos, axis=-1)

            cs_at_last = ops.take_along_axis(cs, last_pos_exp, axis=-1)[..., 0]
            p_at_last = ops.take_along_axis(sorted_p, last_pos_exp, axis=-1)[..., 0]

            reg_at_L = lambd * ops.maximum(index_limit - k_reg, 0)
            proba_excess = (cs_at_last + reg_at_L) - quantile

            indic_L_greater_kreg = ops.where(index_limit > k_reg, 1, 0)
            denom = p_at_last + lambd * indic_L_greater_kreg

            v = proba_excess / denom
            v = ops.clip(v, 0, 1)
            exclude_last = ops.where(u <= v, 1, 0)
            index_limit = ops.maximum(index_limit - exclude_last, 0)
        return [p[:lim] for p, lim in zip(sorted_index, index_limit)]
    return _raps_set

def aps_set(rand:bool=False)->PredSetFunction:
    return raps_set(lambd=1, k_reg=1, rand=rand)

def _constant_bbox(y_pred:TensorLike, quantile:float|TensorLike) -> Any:
    x_min, y_min, x_max, y_max = ops.split(y_pred, 4, axis=1)
    # Coordinates of covering bbox (upperbounds)
    x_min_lo, y_min_lo = x_min - quantile[0], y_min - quantile[1]
    x_max_hi, y_max_hi = x_max + quantile[2], y_max + quantile[3]
    Y_pred_hi = ops.hstack([x_min_lo, y_min_lo, x_max_hi, y_max_hi])

    # Coordinates of included bbox (lowerbounds)
    x_min_hi, y_min_hi = x_min + quantile[0], y_min + quantile[1]
    x_max_lo, y_max_lo = x_max - quantile[2], y_max - quantile[3]
    Y_pred_lo = ops.hstack([x_min_hi, y_min_hi, x_max_lo, y_max_lo])
    return ops.stack([Y_pred_lo, Y_pred_hi], axis=-1)

def constant_bbox()->PredSetFunction:
    return _constant_bbox

def _scaled_bbox(y_pred:TensorLike, quantile:float|TensorLike) -> Any:
    print("quantile : ", quantile)
    x_min, y_min, x_max, y_max = ops.split(y_pred, 4, axis=1)
    dx = ops.abs(x_max - x_min)
    dy = ops.abs(y_max - y_min)
    qd = [quantile[0] * dx, quantile[1] * dy, quantile[2] * dx, quantile[3] * dy]
    # Coordinates of covering bbox (upperbounds)
    x_min_lo = x_min - qd[0]
    y_min_lo = y_min - qd[1]
    x_max_hi = x_max + qd[2]
    y_max_hi = y_max + qd[3]

    Y_pred_outer = ops.hstack([x_min_lo, y_min_lo, x_max_hi, y_max_hi])

    # Coordinates of included bbox (lowerbounds)
    x_min_hi, y_min_hi = (
        x_min + qd[0],
        y_min + qd[1],
    )
    x_max_lo, y_max_lo = (
        x_max - qd[2],
        y_max - qd[3],
    )
    Y_pred_inner = ops.hstack([x_min_hi, y_min_hi, x_max_lo, y_max_lo])

    return ops.stack([Y_pred_inner, Y_pred_outer], axis=-1)

def scaled_bbox():
    return _scaled_bbox


def _classwise_lac_set(
    Y_pred, scores_quantile
):
    """Classwise LAC prediction set.

    For each sample i and class c, include c in the prediction set if:
        Y_pred[i, c] >= 1 - scores_quantile[c]

    :param Iterable Y_pred:
        :math:`Y_{\\text{pred}} = (P_{\\text{C}_1}, ..., P_{\\text{C}_n})`
        where :math:`P_{\\text{C}_i}` is logit associated to class i.

    :param ndarray scores_quantile: per-class quantiles of nonconformity scores
        computed on a calibration set for a given :math:`\\alpha`.
        Shape: (n_classes,)

    :returns: Classwise LAC prediction sets.
    :rtype: List

    """
    # Check if logits sum is close to one
    logit_normalization_check(Y_pred)

    #logger.debug(f"Shape of Y_pred: {Y_pred.shape}")

    # Threshold per class: 1 - quantile[c]
    thresholds = 1 - scores_quantile  # shape: (n_classes,)

    # Build prediction sets: include class c if Y_pred[i, c] >= threshold[c]
    prediction_sets = [
        ops.where(Y_pred[i] >= thresholds)[0].tolist() for i in range(Y_pred.shape[0])
    ]

    return (prediction_sets,)

def classwise_lac_set()->PredSetFunction:
    return _classwise_lac_set
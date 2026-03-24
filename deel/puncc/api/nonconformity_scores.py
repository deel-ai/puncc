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
This module provides nonconformity scores for conformal prediction. To be used
when building a ConformalPredictor
"""
from collections.abc import Sequence
import warnings
from deel.puncc.api.utils import logit_normalization_check, supported_types_check
from deel.puncc.typing import TensorLike, NCScoreFunction
from deel.puncc import ops
from deel.puncc._keras import random

def _difference(y_pred:TensorLike, y_true:TensorLike) -> Sequence[float]:
    return y_pred - y_true

def difference()->NCScoreFunction:
    return _difference

def _absolute_difference(y_pred:TensorLike, y_true:TensorLike) -> Sequence[float]:
    return ops.abs(y_pred - y_true)

def absolute_difference()->NCScoreFunction:
    return _absolute_difference

def scaled_ad(eps:float=1e-12)-> NCScoreFunction:
    def _scaled_ad(y_pred:TensorLike, y_true:TensorLike) -> Sequence[float]:
        mean_pred = ops.take(y_pred, 0, axis=-1)
        var_pred = ops.take(y_pred, 1, axis=-1)
        mean_abs_dev = ops.abs(mean_pred - y_true)
        if ops.any(var_pred + eps <= 0):
            warnings.warn("Warning: calibration points with MAD predictions below -eps won't be used for calibration.",
                RuntimeWarning,
                stacklevel=2,
            )
        nonneg = var_pred + eps > 0
        return mean_abs_dev[nonneg] / (var_pred[nonneg] + eps)
    return _scaled_ad

def _cqr_score(y_pred:TensorLike, y_true:TensorLike) -> Sequence[float]:
    lower_pred = ops.take(y_pred, 0, axis=-1)
    upper_pred = ops.take(y_pred, 1, axis=-1)
    return ops.maximum(lower_pred - y_true, y_true - upper_pred)

def cqr_score()->NCScoreFunction:
    return _cqr_score

def _scaled_bbox_difference(y_pred:TensorLike, y_true:TensorLike) -> Sequence[float]:
    x_min, y_min, x_max, y_max = ops.split(y_pred, 4, axis=1)
    dx = ops.abs(x_max - x_min)
    dy = ops.abs(y_max - y_min)
    return (y_pred - y_true) / ops.hstack([dx, dy, dx, dy])

def scaled_bbox_difference()->NCScoreFunction:
    return _scaled_bbox_difference

def _lac_score(y_pred:TensorLike, y_true:TensorLike) -> Sequence[float]:
    return 1 - y_pred[ops.arange(ops.shape(y_true)[0]), y_true]

def lac_score()->NCScoreFunction:
    return _lac_score

def raps_score(lambd:float=0, k_reg:int=1, rand:bool=True)->NCScoreFunction:
    def _raps_score(y_pred:TensorLike, y_true:TensorLike) -> Sequence[float]:
        condition = y_pred>=ops.take_along_axis(y_pred, y_true[..., None], axis=-1)
        s = ops.sum(ops.where(condition, y_pred, 0), axis=-1)
        nb_cum_elems = ops.sum(condition, axis=-1)
        regul = lambd * ops.maximum(nb_cum_elems - k_reg, ops.zeros_like(nb_cum_elems, dtype=s.dtype))
        rand_correction = 0
        if rand:
            u = random.uniform(ops.shape(s))
            rand_correction = u * ops.squeeze(ops.take_along_axis(y_pred, y_true[..., None], axis=-1), axis=-1)
        regul = ops.cast(regul, dtype=s.dtype)
        return s + regul - rand_correction
    return _raps_score

def aps_score(rand:bool=True)->NCScoreFunction:
    return raps_score(lambd=1, k_reg=1, rand=rand)

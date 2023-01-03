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
This module implements the core Calibrator and provides a collection of non-conformity scores and computations of the prediction sets.
"""
from abc import ABC
from abc import abstractmethod
from typing import Iterable
from typing import Optional
from typing import Union

import numpy as np

from deel.puncc.api.utils import check_alpha_calib
from deel.puncc.api.utils import EPSILON
from deel.puncc.api.utils import quantile
from deel.puncc.api.utils import supported_types_check


def abs_nonconf_score(y_pred, y_true):
    supported_types_check(y_pred, y_true)
    diff = y_pred - y_true
    if isinstance(y_pred, np.ndarray):
        return np.absolute(diff)
    elif isinstance(y_pred, pd.DataFrame):
        return diff.abs()
    elif isinstance(y_pred, tf.Tensor):
        return tf.math.abs(diff)
    elif isinstance(y_pred, torch.Tensor):
        return torch.abs(diff)
    else:  # Sanity check, this should never happen.
        raise RuntimeError("Fatal Error. Type check failed !")


def constant_interval_pred(y_pred, scores_quantiles):
    supported_types_check(y_pred)
    y_lo, y_hi = y_pred - scores_quantiles, y_pred + scores_quantiles
    return y_lo, y_hi


class BaseCalibrator:
    def __init__(self, *, nonconf_score_func, pred_set_func):
        self.nonconf_score_func = nonconf_score_func
        self.pred_set_func = pred_set_func
        self._len_calib = 0
        self._residuals = None

    def fit(
        self,
        *,
        y_true: Iterable,
        y_pred: Iterable,
    ) -> None:
        """Compute and store nonconformity scores on the calibration set.

        :param ndarray|DataFrame|Tensor y_true: true values.
        :param ndarray|DataFrame|Tensor y_pred: predicted values.

        """
        # TODO check structure match in supported types
        self._residuals = self.nonconf_score_func(y_true=y_true, y_pred=y_pred)
        self._len_calib = len(self._residuals)

    def calibrate(
        self,
        alpha: float,
        y_pred: Iterable,
        weights: Optional[Iterable] = None,
    ) -> tuple[Iterable, Iterable,]:
        """Compute calibrated prediction intervals for new examples.

        :param float alpha: significance level (max miscoverage target).
        :param ndarray|DataFrame|Tensor y_pred: predicted values.
        :param Iterable weights: weights to be associated to the non-conformity
                                 scores. Defaults to None when all the scores
                                 are equiprobable.

        :returns: y_lower, y_upper.
        :rtype: tuple[ndarray|DataFrame|Tensor, ndarray|DataFrame|Tensor]

        :raises RuntimeError: :meth:`calibrate` called before :meth:`fit`.
        :raise ValueError: failed check on :data:`alpha` w.r.t size of the calibration set.

        """
        if self._residuals is None:
            raise RuntimeError("Run `fit` method before calling `calibrate`.")

        # Check consistency of alpha w.r.t the size of calibration data
        check_alpha_calib(alpha=alpha, n=self._len_calib)

        residuals_Qs = list()

        # Fix to factorize the loop on weights:
        # Enables the loop even when the weights are not provided
        if weights is None:
            it_weights = [None]
        else:
            it_weights = weights

        for w in it_weights:
            infty_array = np.array([np.inf])
            lemma_residuals = np.concatenate((self._residuals, infty_array))
            residuals_Q = quantile(
                lemma_residuals,
                1 - alpha,
                w=w,
            )
            residuals_Qs.append(residuals_Q)
        residuals_Qs = np.array(residuals_Qs)

        y_lo, y_hi = self.pred_set_func(y_pred=y_pred, scores_quantiles=residuals_Qs)
        return y_lo, y_hi

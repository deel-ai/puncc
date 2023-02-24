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
This module provides prediction sets for conformal prediction.
"""
import numpy as np
import pandas as pd

from deel.puncc.api.utils import supported_types_check


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Classification ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def raps_set(lambd=0, k_reg=1):
    # @TODO: type checking and co
    def _RAPS_SET(Y_pred, scores_quantiles):
        pred_len = len(Y_pred)
        # Generate u randomly from a uniform distribution
        u = np.random.uniform(size=pred_len)
        # 1-alpha th empirical quantile of conformity scores
        tau = scores_quantiles
        # Rank classes by their probability
        idx_class_pred_ranking = np.argsort(-Y_pred, axis=1)
        sorted_proba = -np.sort(-Y_pred, axis=1)
        sorted_cum_mass = sorted_proba.cumsum(axis=1)
        # First sorted class for which the cumulative probability mass
        # exceeds the threshold "tau"
        penal_proba = [
            sorted_proba[i]
            + lambd
            * np.maximum(
                np.arange(1, len(sorted_cum_mass[i]) + 1) - k_reg,
                0,  # Because classes are ranked by probability, o_x(y) corresponds [1, 2, 3, ...]
            )
            for i in range(pred_len)
        ]
        penal_cum_proba = [
            sorted_cum_mass[i]
            + lambd
            * np.maximum(
                np.arange(1, len(sorted_cum_mass[i]) + 1) - k_reg,
                0,  # Because classes are ranked by probability, o_x(y) corresponds [1, 2, 3, ...]
            )
            for i in range(pred_len)
        ]
        L = [len(np.where(penal_cum_proba[i] <= tau)[-1]) + 1 for i in range(pred_len)]
        proba_excess = [
            -sorted_cum_mass[i, L[i] - 1]
            - lambd * np.maximum(L[i] - k_reg, 0)
            + sorted_proba[i, L[i] - 1]
            for i in range(pred_len)
        ]
        v = [
            (tau - proba_excess[i]) / sorted_proba[i, L[i] - 1]
            if (L[i] - 1) >= 0
            else np.inf
            for i in range(pred_len)
        ]

        # Build prediction set
        prediction_sets = list()

        for i in range(pred_len):

            if v[i] <= u[i]:
                cut_i = L[i] - 1
            else:
                cut_i = L[i]

            if cut_i < 0:
                prediction_sets.append([])
            else:
                prediction_sets.append(list(idx_class_pred_ranking[i, :cut_i]))

        return (prediction_sets,)

    return _RAPS_SET


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Regression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def constant_interval(y_pred, scores_quantiles):
    supported_types_check(y_pred)
    y_lo = y_pred - scores_quantiles
    y_hi = y_pred + scores_quantiles
    return y_lo, y_hi


def scaled_interval(Y_pred, scores_quantiles):
    supported_types_check(Y_pred)

    if Y_pred.shape[1] != 2:  # check Y_pred contains two predictions
        raise RuntimeError(
            f"Each Y_pred must contain a point prediction and a dispersion estimation."
        )

    if isinstance(Y_pred, pd.DataFrame):
        y_pred, sigma_pred = Y_pred.iloc[:, 0], Y_pred.iloc[:, 1]
    else:
        y_pred, sigma_pred = Y_pred[:, 0], Y_pred[:, 1]

    y_lo = y_pred - scores_quantiles * sigma_pred
    y_hi = y_pred + scores_quantiles * sigma_pred
    return y_lo, y_hi


def cqr_interval(Y_pred, scores_quantiles):
    supported_types_check(Y_pred)

    if Y_pred.shape[1] != 2:  # check Y_pred contains two predictions
        raise RuntimeError(
            f"Each Y_pred must contain lower and higher quantiles predictions, respectively."
        )

    if isinstance(Y_pred, pd.DataFrame):
        q_lo, q_hi = Y_pred.iloc[:, 0], Y_pred.iloc[:, 1]
    else:
        q_lo, q_hi = Y_pred[:, 0], Y_pred[:, 1]

    y_lo = q_lo - scores_quantiles
    y_hi = q_hi + scores_quantiles
    return y_lo, y_hi

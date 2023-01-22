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
This module provides nonconformity scores for conformal prediction.
"""
import numpy as np
import pandas as pd

from deel.puncc.api.utils import EPSILON
from deel.puncc.api.utils import supported_types_check


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Classification ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def raps_score(lambd=0, k_reg=1):
    def _RAPS_SCORE(Y_pred, y_true):
        """APS nonconformity score.
        Given :math:`n` the number of classes,
            :math:`Y_\text{{pred}} = (P_{\text{C}_1}, ..., P_{\text{C}_n})`

        Refer to https://arxiv.org/abs/2009.14193 for details.

        :param ndarray|DataFrame|Tensor Y_pred: :math:`Y_\text{{pred}} = (P_{\text{C}_1}, ..., P_{\text{C}_n})`
            where :math:`P_{\text{C}_i}` is logit associated to class i.
        :param ndarray|DataFrame|Tensor y_true:

        :returns: APS nonconformity scores.
        :rtype: ndarray|DataFrame|Tensor
        """

        supported_types_check(Y_pred, y_true)

        if not isinstance(Y_pred, np.ndarray):
            raise NotImplementedError(
                "RAPS nonconformity scores only implemented for ndarrays"
            )
        # Generate rand randomly from a uniform distribution
        rand = np.random.uniform(size=len(y_true))
        # Sort classes by descending probability order
        class_ranking = np.argsort(-Y_pred, axis=1)
        sorted_proba = -np.sort(-Y_pred, axis=1)
        # Cumulative probability mass (given the previous class ranking)
        sorted_cum_mass = sorted_proba.cumsum(axis=1)
        # Locate position of true label in the classes
        # sequence ranked by decreasing probability
        L = [
            np.where(class_ranking[i] == y_true[i])[0][0]
            for i in range(y_true.shape[0])
        ]
        # Threshold of cumulative probability mass to include the real class
        E = [sorted_cum_mass[i, L[i]] for i in range(y_true.shape[0])]
        E = [
            E[i]
            + (rand[i] - 1) * sorted_proba[i, L[i]]
            + lambd * np.maximum((L[i] - k_reg + 1), 0)
            for i in range(y_true.shape[0])
        ]
        return np.array(E)

    return _RAPS_SCORE


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Regression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def mad(y_pred, y_true):
    """Mean Absolute Deviation (MAD).

    .. math::
        R = |y_{\text{true}}-y_\text{{pred}}|

    :param ndarray|DataFrame|Tensor y_pred:
    :param ndarray|DataFrame|Tensor y_true:

    :returns: mean absolute deviation.
    :rtype: ndarray|DataFrame|Tensor
    """
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


def scaled_mad(Y_pred, y_true):
    """Scaled Mean Absolute Deviation (MAD).

    .. math::
        R = \frac{|Y_{\text{true}}-Y_\text{{pred}}|}{\sigma_\text{{pred}}}

    :param ndarray|DataFrame|Tensor Y_pred: :math:`Y_\text{{pred}}=(y_\text{{pred}}, \sigma_\text{{pred}})`
    :param ndarray|DataFrame|Tensor y_true: point observation.

    :returns: scaled mean absolute deviation.
    :rtype: ndarray|DataFrame|Tensor
    """
    supported_types_check(Y_pred, y_true)

    if Y_pred.shape[1] != 2:  # check Y_pred contains two predictions
        raise RuntimeError(
            f"Each Y_pred must contain a point prediction and a dispersion estimation."
        )

    # check y_true is a collection of point observations
    if len(y_true.shape) != 1:
        raise RuntimeError(f"Each y_pred must contain a point observation.")

    if isinstance(Y_pred, pd.DataFrame):
        y_pred, var_pred = Y_pred.iloc[:, 0], Y_pred.iloc[:, 1]
    else:
        y_pred, var_pred = Y_pred[:, 0], Y_pred[:, 1]

    # MAD then Scaled MAD and computed
    mean_absolute_deviation = mad(y_pred, y_true)
    return mean_absolute_deviation / (var_pred + EPSILON)


def cqr_score(Y_pred, y_true):
    """CQR nonconformity score.
    Considering :math:`Y_\text{{pred}} = (q_{\text{lo}}, q_{\text{hi}})`
    .. math::
        R = max\{q_{\text{lo}} - y_{\text{true}}, y_{\text{true}} - q_{\text{hi}}\}

    :param ndarray|DataFrame|Tensor Y_pred: :math:`Y_\text{{pred}} = (q_{\text{lo}}, q_{\text{hi}})`
        where :math:`q_{\text{lo}}` (resp. :math:`q_{\text{hi}}` is the lower (resp. higher) quantile prediction.
    :param ndarray|DataFrame|Tensor y_true:

    :returns: CQR nonconformity scores.
    :rtype: ndarray|DataFrame|Tensor
    """
    supported_types_check(Y_pred, y_true)

    if Y_pred.shape[1] != 2:  # check Y_pred contains two predictions
        raise RuntimeError(
            f"Each Y_pred must contain lower and higher quantiles estimations."
        )

    # check y_true is a collection of point observations
    if len(y_true.shape) != 1:
        raise RuntimeError(f"Each y_pred must contain a point observation.")

    if isinstance(Y_pred, pd.DataFrame):
        q_lo, q_hi = Y_pred.iloc[:, 0], Y_pred.iloc[:, 1]
    else:
        q_lo, q_hi = Y_pred[:, 0], Y_pred[:, 1]

    diff_lo = q_lo - y_true
    diff_hi = y_true - q_hi

    if isinstance(diff_lo, np.ndarray):
        return np.maximum(diff_lo, diff_hi)
    elif isinstance(diff_lo, pd.DataFrame):
        raise NotImplementedError(
            "CQR score not implemented for DataFrames. Please provide ndarray or tensors."
        )
    elif isinstance(diff_lo, tf.Tensor):
        return tf.math.maximum(diff_lo, diff_hi)
    elif isinstance(diff_lo, torch.Tensor):
        return torch.maximum(diff_lo, diff_hi)
    else:  # Sanity check, this should never happen.
        raise RuntimeError("Fatal Error. Type check failed !")

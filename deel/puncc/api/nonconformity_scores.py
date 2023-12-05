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
when building a :ref:`calibrator <calibration>`.
"""
import pkgutil
from typing import Callable
from typing import Iterable

import numpy as np

from deel.puncc.api.utils import EPSILON
from deel.puncc.api.utils import logit_normalization_check
from deel.puncc.api.utils import supported_types_check

if pkgutil.find_loader("pandas") is not None:
    import pandas as pd

if pkgutil.find_loader("tensorflow") is not None:
    import tensorflow as tf

if pkgutil.find_loader("torch") is not None:
    import torch


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Classification ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def raps_score(
    Y_pred: Iterable,
    y_true: Iterable,
    lambd: float = 0,
    k_reg: int = 1,
    rand: bool = True,
) -> Iterable:
    """RAPS nonconformity score.

    .. warning::

        This signature is incompatible with the interface of calibrators.
        **Use** :func:`raps_score_builder` **to properly initialize**
        :class:`deel.puncc.api.calibration.BaseCalibrator`.

    :param Iterable Y_pred:
        :math:`Y_{\\text{pred}} = (P_{\\text{C}_1}, ..., P_{\\text{C}_n})`
        where :math:`P_{\\text{C}_i}` is logit associated to class i.
    :param Iterable y_true: true labels.
    :param float lambd: positive weight associated to the regularization term
        that encourages small set sizes. If :math:`\\lambda = 0`, there is no
        regularization and the implementation identifies with **APS**.
    :param float k_reg: class rank (ordered by descending probability) starting
        from which the regularization is applied. For example,
        if :math:`k_{reg} = 3`, then the fourth most likely estimated class has
        an extra penalty of size :math:`\\lambda`.
    : param bool rand: turn on or off the randomization term that smoothes the
        discrete probability mass jump when including a new class.


    :returns: RAPS nonconformity scores.
    :rtype: Iterable

    """
    supported_types_check(Y_pred, y_true)

    # Check if logits sum is close to one
    logit_normalization_check(Y_pred)

    if not isinstance(Y_pred, np.ndarray):
        raise NotImplementedError(
            "RAPS/APS nonconformity scores only implemented for ndarrays"
        )

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

    if rand:
        # Generate u randomly from a uniform distribution
        u = np.random.uniform(size=len(y_true))
        E = [
            E[i]
            + lambd * np.maximum((L[i] + 1 - k_reg), 0)
            - u[i] * sorted_proba[i, L[i]]
            for i in range(y_true.shape[0])
        ]
    else:
        E = [
            E[i] + lambd * np.maximum((L[i] + 1 - k_reg), 0)
            for i in range(y_true.shape[0])
        ]

    return np.array(E)


def raps_score_builder(
    lambd: float = 0, k_reg: int = 1, rand: bool = True
) -> Callable:
    """RAPS nonconformity score builder. When called, returns a RAPS
    nonconformity score function :func:`raps_score` with given initialitation
    of regularization hyperparameters.

    :param float lambd: positive weight associated to the regularization term
        that encourages small set sizes. If :math:`\\lambda = 0`, there is no
        regularization and the implementation identifies with **APS**.
    :param float k_reg: class rank (ordered by descending probability) starting
        from which the regularization is applied. For example, if
        :math:`k_{reg} = 3`, then the fourth most likely estimated class has
        an extra penalty of size :math:`\\lambda`.
    : param bool rand: turn on or off the randomization term that smoothes the
        discrete probability mass jump when including a new class.

    :returns: RAPS nonconformity score function that takes two parameters:
        `Y_pred` and `y_true`.
    :rtype: Callable

    """

    def _raps_score_function(Y_pred: Iterable, y_true: Iterable) -> np.ndarray:
        return raps_score(Y_pred, y_true, lambd, k_reg, rand)

    return _raps_score_function


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Regression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def mad(y_pred: Iterable, y_true: Iterable) -> Iterable:
    """Mean Absolute Deviation (MAD).

    .. math::

        R = |y_{\\text{true}}-y_{\\text{pred}}|

    :param Iterable y_pred: predictions.
    :param Iterable y_true: true labels.

    :returns: mean absolute deviation.
    :rtype: Iterable

    :raises TypeError: unsupported data types.
    """
    supported_types_check(y_pred, y_true)

    if pkgutil.find_loader("torch") is not None and isinstance(
        y_pred, torch.Tensor
    ):
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        return abs(np.squeeze(y_pred) - np.squeeze(y_true))

    return abs(y_pred - y_true)


def scaled_mad(Y_pred: Iterable, y_true: Iterable) -> Iterable:
    """Scaled Mean Absolute Deviation (MAD). Considering
    :math:`Y_{\\text{pred}} = (\mu_{\\text{pred}}, \sigma_{\\text{pred}})`:

    .. math::

        R = \\frac{|y_{\\text{true}}-\mu_{\\text{pred}}|}{\sigma_{\\text{pred}}}

    :param Iterable Y_pred:
        :math:`Y_{\\text{pred}}=(y_{\\text{pred}}, \sigma_{\\text{pred}})`
    :param Iterable y_true: true labels.

    :returns: scaled mean absolute deviation.
    :rtype: Iterable

    :raises TypeError: unsupported data types.
    """
    supported_types_check(Y_pred, y_true)

    if Y_pred.shape[1] != 2:  # check Y_pred contains two predictions
        raise RuntimeError(
            "Each Y_pred must contain a point prediction and a dispersion estimation."
        )

    # check y_true is a collection of point observations
    if len(y_true.shape) != 1:
        raise RuntimeError("Each y_pred must contain a point observation.")

    if pkgutil.find_loader("pandas") is not None and isinstance(
        Y_pred, pd.DataFrame
    ):
        y_pred, sigma_pred = Y_pred.iloc[:, 0], Y_pred.iloc[:, 1]
    else:
        y_pred, sigma_pred = Y_pred[:, 0], Y_pred[:, 1]

    # MAD then Scaled MAD and computed
    mean_absolute_deviation = mad(y_pred, y_true)
    if np.any(sigma_pred < 0):
        raise RuntimeError("All MAD predictions should be positive.")
    return mean_absolute_deviation / (sigma_pred + EPSILON)


def cqr_score(Y_pred: Iterable, y_true: Iterable) -> Iterable:
    """CQR nonconformity score. Considering
    :math:`Y_{\\text{pred}} = (q_{\\text{lo}}, q_{\\text{hi}})`:

    .. math::

        R = max\{q_{\\text{lo}} - y_{\\text{true}}, y_{\\text{true}} - q_{\\text{hi}}\}

    where :math:`q_{\\text{lo}}` (resp. :math:`q_{\\text{hi}}`) is the lower
    (resp. higher) quantile prediction

    :param Iterable Y_pred: predicted quantiles couples.
    :param Iterable y_true: true quantiles couples.

    :returns: CQR nonconformity scores.
    :rtype: Iterable
    """
    supported_types_check(Y_pred, y_true)

    if Y_pred.shape[1] != 2:  # check Y_pred contains two predictions
        raise RuntimeError(
            "Each Y_pred must contain lower and higher quantiles estimations."
        )

    # check y_true is a collection of point observations
    if len(y_true.shape) != 1:
        raise RuntimeError("Each y_pred must contain a point observation.")

    if pkgutil.find_loader("pandas") is not None and isinstance(
        Y_pred, pd.DataFrame
    ):
        q_lo, q_hi = Y_pred.iloc[:, 0], Y_pred.iloc[:, 1]
    else:
        q_lo, q_hi = Y_pred[:, 0], Y_pred[:, 1]

    diff_lo = q_lo - y_true
    diff_hi = y_true - q_hi

    if isinstance(diff_lo, np.ndarray):
        return np.maximum(diff_lo, diff_hi)

    if pkgutil.find_loader("pandas") is not None and isinstance(
        diff_lo, (pd.DataFrame, pd.Series)
    ):
        return (pd.concat([diff_lo, diff_hi]).groupby(level=0)).max()
        # raise NotImplementedError(
        #     "CQR score not implemented for DataFrames. Please provide ndarray or tensors."
        # )

    if pkgutil.find_loader("tensorflow") is not None and isinstance(
        diff_lo, tf.Tensor
    ):
        return tf.math.maximum(diff_lo, diff_hi)

    # if pkgutil.find_loader("torch") is not None and isinstance(
    #     diff_lo, torch.Tensor
    # ):
    #     return torch.maximum(diff_lo, diff_hi)

    raise RuntimeError("Fatal Error. Type check failed !")

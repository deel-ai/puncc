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
building a :ref:`calibrator <calibration>`.
"""
import logging
import pkgutil
from typing import Callable
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np

from deel.puncc.api.utils import logit_normalization_check
from deel.puncc.api.utils import supported_types_check

if pkgutil.find_loader("pandas") is not None:
    import pandas as pd

logger = logging.getLogger(__name__)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Classification ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def raps_set(
    Y_pred, scores_quantile, lambd: float = 0, k_reg: int = 1, rand: bool = True
) -> List:
    """RAPS prediction set.

    .. warning::

        This signature is incompatible with the interface of calibrators.
        **Use** :func:`raps_set_builder` **to properly initialize**
        :class:`deel.puncc.api.calibration.BaseCalibrator`.

    :param Iterable Y_pred:
        :math:`Y_{\\text{pred}} = (P_{\\text{C}_1}, ..., P_{\\text{C}_n})`
        where :math:`P_{\\text{C}_i}` is logit associated to class i.
    :param ndarray scores_quantile: quantile of nonconformity scores computed
        on a calibration set for a given :math:`\\alpha`
    :param float lambd: positive weight associated to the regularization term
        that encourages small set sizes. If :math:`\\lambda = 0`, there is no
        regularization and the implementation identifies with **APS**.
    :param float k_reg: class rank (ordered by descending probability) starting
        from which the regularization is applied. For example, if
        :math:`k_{reg} = 3`, then the fourth most likely estimated class has an
        extra penalty of size :math:`\\lambda`.
    : param bool rand: turn on or off the randomization term that smoothes the
        discrete probability mass jump when including a new class.


    :returns: RAPS prediction sets.
    :rtype: Iterable

    """
    # Check if logits sum is close to one
    logit_normalization_check(Y_pred)

    pred_len = len(Y_pred)

    logger.debug(f"Shape of Y_pred: {Y_pred.shape}")

    if rand:
        # Generate u randomly from a uniform distribution
        u = np.random.uniform(size=pred_len)

    # 1-alpha th empirical quantile of conformity scores
    tau = scores_quantile

    # Rank classes by their probability
    idx_class_pred_ranking = np.argsort(-Y_pred, axis=-1)
    sorted_proba = -np.sort(-Y_pred, axis=-1)
    sorted_cum_mass = sorted_proba.cumsum(axis=-1)

    # Cumulative probability mass of (sorted by descending probability) classes
    # penalized by the regularization term
    penal_cum_proba = sorted_cum_mass + lambd * np.maximum(
        np.arange(1, sorted_cum_mass.shape[-1] + 1) - k_reg, 0
    )

    # L is the number of classes (+1) for which the cumulative probability mass
    # is below the threshold "tau"
    # The minimum is used in case the Y_pred logits are not well normalized
    L = np.minimum(np.sum(penal_cum_proba < tau, axis=-1) + 1, pred_len)

    if not rand:
        # Build prediction set
        prediction_sets = [
            list(idx_class_pred_ranking[i, : L[i]]) for i in range(pred_len)
        ]
        return (prediction_sets,)

    ## The following code runs only when rand argument is True
    # For indexing, use L-1 to denote the L-th element
    # Residual of cumulative probability mass (regularized) and tau
    proba_excess = (
        sorted_cum_mass[np.arange(pred_len), L - 1]
        + lambd * np.maximum(L - k_reg, 0)
        - tau
    )

    # Indicator when regularization rank is exceeded
    indic_L_greater_kreg = np.where(L > k_reg, 1, 0)

    # Normalized probability mass excess
    v = proba_excess / (
        sorted_proba[np.arange(pred_len), L - 1] + lambd * indic_L_greater_kreg
    )

    # Least likely class in the prediction set is removed
    # with a probability of norm_proba_excess
    indic_excess_proba = np.where(u <= v, 1, 0)
    L = L - indic_excess_proba

    # Build prediction set
    prediction_sets = [
        list(idx_class_pred_ranking[i, : L[i]]) for i in range(pred_len)
    ]

    return (prediction_sets,)


def raps_set_builder(
    lambd: float = 0, k_reg: int = 1, rand: bool = True
) -> Callable:
    """RAPS prediction set builder. When called, returns a RAPS prediction set
    function :func:`raps_set` with given initialitation of regularization
    hyperparameters.

    :param float lambd: positive weight associated to the regularization term
        that encourages small set sizes. If :math:`\\lambda = 0`, there is no
        regularization and the implementation identifies with **APS**.
    :param float k_reg: class rank (ordered by descending probability) starting
        from which the regularization is applied. For example, if
        :math:`k_{reg} = 3`, then the fourth most likely estimated class has an
        extra penalty of size :math:`\\lambda`.
    : param bool rand: turn on or off the randomization term that smoothes the
        discrete probability mass jump when including a new class.

    :returns: RAPS prediction set function that takes two parameters:
        `Y_pred` and `scores_quantile`.
    :rtype: Callable

    :raises ValueError: incorrect value of lambd or k_reg.
    :raises TypeError: unsupported data types.
    """
    if lambd < 0:
        raise ValueError(
            f"Argument `lambd` has to be positive, provided: {lambd} < 0"
        )
    if k_reg < 0:
        raise ValueError(
            f"Argument `k_reg` has to be positive, provided: {k_reg} < 0"
        )
    # @TODO: type checking and co

    def _raps_set_function(Y_pred, scores_quantile):
        return raps_set(
            Y_pred, scores_quantile, lambd=lambd, k_reg=k_reg, rand=rand
        )

    return _raps_set_function


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Regression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def constant_interval(
    y_pred: Iterable, scores_quantile: np.ndarray
) -> Tuple[np.ndarray]:
    """Constant prediction interval centered on `y_pred`. The size of the
    margin is `scores_quantile` (noted :math:`\gamma_{\\alpha}`).

    .. math::

        I = [y_{\\text{pred}} - \gamma_{\\alpha}, y_{\\text{pred}} +
        \gamma_{\\alpha}]

    :param Iterable y_pred: predictions.
    :param ndarray scores_quantile: quantile of nonconformity scores computed
        on a calibration set for a given :math:`\\alpha`.

    :returns: prediction intervals :math:`I`.
    :rtype: Tuple[ndarray]

    :raises TypeError: unsupported data types.
    """
    supported_types_check(y_pred)
    y_lo = y_pred - scores_quantile
    y_hi = y_pred + scores_quantile
    return y_lo, y_hi


def scaled_interval(
    Y_pred: Iterable, scores_quantile: np.ndarray, eps: float = 1e-12
) -> Tuple[np.ndarray]:
    """Scaled prediction interval centered on `y_pred`. Considering
    :math:`Y_{\\text{pred}} = (\mu_{\\text{pred}}, \sigma_{\\text{pred}})`,
    the size of the margin is proportional to `scores_quantile`
    :math:`\gamma_{\\alpha}`.

    .. math::

        I = [\mu_{\\text{pred}} - \gamma_{\\alpha} \cdot \sigma_{\\text{pred}},
        y_{\\text{pred}} + \gamma_{\\alpha} \cdot \sigma_{\\text{pred}}]

    :param Iterable y_pred: predictions.
    :param ndarray scores_quantile: quantile of nonconformity scores computed
        on a calibration set for a given :math:`\\alpha`.
    :param float eps: small positive value to avoid singleton sets.

    :returns: scaled prediction intervals :math:`I`.
    :rtype: Tuple[ndarray]

    :raises TypeError: unsupported data types.
    """
    supported_types_check(Y_pred)

    if Y_pred.shape[1] != 2:  # check Y_pred contains two predictions
        raise RuntimeError(
            "Each Y_pred must contain a point prediction and a dispersion estimation."
        )

    if pkgutil.find_loader("pandas") is not None and isinstance(
        Y_pred, pd.DataFrame
    ):
        y_pred, sigma_pred = Y_pred.iloc[:, 0], Y_pred.iloc[:, 1]
    else:
        y_pred, sigma_pred = Y_pred[:, 0], Y_pred[:, 1]

    if np.any(sigma_pred + eps <= 0):
        print("Warning: test points with MAD predictions below -eps"
              " will have infinite sized prediction intervals.")

    fints = sigma_pred + eps > 0
    y_lo, y_hi = np.zeros_like(y_pred), np.zeros_like(y_pred)
    y_lo[fints] = y_pred[fints] - scores_quantile * (sigma_pred[fints] + eps)
    y_hi[fints] = y_pred[fints] + scores_quantile * (sigma_pred[fints] + eps)
    y_lo[~fints] = -np.inf
    y_hi[~fints] = np.inf
    return y_lo, y_hi


def cqr_interval(
    Y_pred: Iterable, scores_quantile: np.ndarray
) -> Tuple[np.ndarray]:
    """CQR prediction interval. Considering
    :math:`Y_{\\text{pred}} = (q_{\\text{lo}}, q_{\\text{hi}})`, the prediction
    interval is built from the upper and lower quantiles predictions and
    `scores_quantile` :math:`\gamma_{\\alpha}`.

    .. math::

        I = [q_{\\text{lo}} - \gamma_{\\alpha}, q_{\\text{lo}} +
        \gamma_{\\alpha}]

    :param Iterable y_pred: predictions.
    :param ndarray scores_quantile: quantile of nonconformity scores computed
        on a calibration set for a given :math:`\\alpha`.

    :returns: scaled prediction intervals :math:`I`.
    :rtype: Tuple[ndarray]

    :raises TypeError: unsupported data types.
    """
    supported_types_check(Y_pred)

    if Y_pred.shape[1] != 2:  # check Y_pred contains two predictions
        raise RuntimeError(
            "Each Y_pred must contain lower and higher quantiles predictions, respectively."
        )

    if pkgutil.find_loader("pandas") is not None and isinstance(
        Y_pred, pd.DataFrame
    ):
        q_lo, q_hi = Y_pred.iloc[:, 0], Y_pred.iloc[:, 1]
    else:
        q_lo, q_hi = Y_pred[:, 0], Y_pred[:, 1]

    y_lo = q_lo - scores_quantile
    y_hi = q_hi + scores_quantile
    return y_lo, y_hi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Object Detection ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def constant_bbox(Y_pred: np.ndarray, scores_quantile: np.ndarray):
    """
    Generate the upper and lower bounds of the bounding box coordinates for
    a given prediction.

    :param np.ndarray Y_pred: the predicted bounding box coordinates.
    :param np.ndarray scores_quantile: quantile of nonconformity scores computed
        on a calibration set for a given :math:`\\alpha`.

    :return: the lower bound and upper bound coordinates of the bounding box.
    :rtype: Tuple[np.ndarray, np.ndarray]

    :raises TypeError: unsupported data types.
    """

    if not isinstance(Y_pred, np.ndarray):
        raise TypeError(
            f"Unsupported data type {type(Y_pred)}."
            + "Please provide a numpy ndarray"
        )

    if Y_pred.shape[1] != 4:  # check Y_pred contains four coordinates
        raise RuntimeError("Each Y_pred must contain 4 bbox coordinates.")

    # Recover each coordinate
    x_min, y_min, x_max, y_max = np.hsplit(Y_pred, 4)

    # Coordinates of covering bbox (upperbounds)
    x_min_lo, y_min_lo = x_min - scores_quantile[0], y_min - scores_quantile[1]
    x_max_hi, y_max_hi = x_max + scores_quantile[2], y_max + scores_quantile[3]
    Y_pred_hi = np.hstack([x_min_lo, y_min_lo, x_max_hi, y_max_hi])

    # Coordinates of included bbox (lowerbounds)
    x_min_hi, y_min_hi = x_min + scores_quantile[0], y_min + scores_quantile[1]
    x_max_lo, y_max_lo = x_max - scores_quantile[2], y_max - scores_quantile[3]
    Y_pred_lo = np.hstack([x_min_hi, y_min_hi, x_max_lo, y_max_lo])

    return Y_pred_lo, Y_pred_hi


def scaled_bbox(Y_pred: np.ndarray, scores_quantile: np.ndarray):
    """
    Scaled upper and lower bounds of the bounding box coordinates
    for a given prediction coordinates.

    :param np.ndarray Y_pred: the predicted bounding box coordinates.
    :param np.ndarray scores_quantile: quantile of nonconformity scores computed
        on a calibration set for a given :math:`\\alpha`.

    :return: The coordinates of the inner and outer bounding boxes.
    :rtype: Tuple[np.ndarray, np.ndarray]

    :raises TypeError: unsupported data types.
    """
    if not isinstance(Y_pred, np.ndarray):
        raise TypeError(
            f"Unsupported data type {type(Y_pred)}."
            + "Please provide a numpy ndarray"
        )

    if Y_pred.shape[1] != 4:  # check Y_pred contains four coordinates
        raise RuntimeError("Each Y_pred must contain 4 bbox coordinates.")

    # Recover each coordinate
    x_min, y_min, x_max, y_max = np.hsplit(Y_pred, 4)

    # Compute width and height of predicted bbox
    dx = np.abs(x_max - x_min)
    dy = np.abs(y_max - y_min)

    # Coordinates of covering bbox (upperbounds)
    x_min_lo = x_min - scores_quantile[0] * dx
    y_min_lo = y_min - scores_quantile[1] * dy
    x_max_hi = x_max + scores_quantile[2] * dx
    y_max_hi = y_max + scores_quantile[3] * dy

    Y_pred_outer = np.hstack([x_min_lo, y_min_lo, x_max_hi, y_max_hi])

    # Coordinates of included bbox (lowerbounds)
    x_min_hi, y_min_hi = (
        x_min + scores_quantile[0] * dx,
        y_min + scores_quantile[1] * dy,
    )
    x_max_lo, y_max_lo = (
        x_max - scores_quantile[2] * dx,
        y_max - scores_quantile[3] * dy,
    )
    Y_pred_inner = np.hstack([x_min_hi, y_min_hi, x_max_lo, y_max_lo])

    return Y_pred_inner, Y_pred_outer

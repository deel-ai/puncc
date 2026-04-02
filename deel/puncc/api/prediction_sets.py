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
from typing import Callable
from typing import Iterable
from typing import List

from deel.puncc.api.backend import concat_columns
from deel.puncc.api.backend import get_backend
from deel.puncc.api.backend import shape2
from deel.puncc.api.backend import split_columns
from deel.puncc.api.utils import logit_normalization_check
from deel.puncc.api.utils import supported_types_check
from deel.puncc.api.utils import supported_meanvar_models_shape_check

logger = logging.getLogger(__name__)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Classification ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def lac_set(
        Y_pred, scores_quantile
) -> List:
    """LAC prediction set.

    :param Iterable Y_pred:
        :math:`Y_{\\text{pred}} = (P_{\\text{C}_1}, ..., P_{\\text{C}_n})`
        where :math:`P_{\\text{C}_i}` is logit associated to class i.

    :param ndarray scores_quantile: quantile of nonconformity scores computed
        on a calibration set for a given :math:`\\alpha`


    :returns: LAC prediction sets.
    :rtype: Iterable

    """
    # Check if logits sum is close to one
    logit_normalization_check(Y_pred)

    b = get_backend(Y_pred)
    yp = b.asarray(Y_pred)

    pred_len = len(yp)
    logger.debug(f"Shape of Y_pred: {shape2(yp)[1]}")

    cond = yp >= (1 - scores_quantile)
    cond_np = b.to_numpy(cond)

    prediction_sets = [
        [j for j, is_in in enumerate(cond_np[i]) if is_in]
        for i in range(pred_len)
    ]

    return (prediction_sets,)


def classwise_lac_set(
    Y_pred, scores_quantile
) -> List:
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

    b = get_backend(Y_pred, scores_quantile)
    yp = b.asarray(Y_pred)
    sq = b.asarray(scores_quantile)
    _, shape = shape2(yp)
    n_test = shape[0]

    logger.debug(f"Shape of Y_pred: {shape}")

    # Threshold per class: 1 - quantile[c]
    thresholds = 1 - b.squeeze(sq)  # shape: (n_classes,)

    # Build prediction sets: include class c if Y_pred[i, c] >= threshold[c].
    cond_np = b.to_numpy(yp >= thresholds)
    prediction_sets = [
        [j for j, is_in in enumerate(cond_np[i]) if is_in]
        for i in range(n_test)
    ]

    return (prediction_sets,)


def raps_set(
    Y_pred, scores_quantile, lambd: float = 0, k_reg: int = 1, rand: bool = True
) -> Iterable:
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

    b = get_backend(Y_pred)
    yp = b.asarray(Y_pred)

    _, shape = shape2(yp)
    pred_len = shape[0]
    n_classes = shape[-1]

    logger.debug(f"Shape of Y_pred: {shape}")

    if rand:
        # Generate u randomly from a uniform distribution
        u = b.random_uniform((pred_len,))

    # 1-alpha th empirical quantile of conformity scores
    tau = scores_quantile

    # Rank classes by their probability
    idx_class_pred_ranking = b.argsort(yp, axis=-1, descending=True)
    sorted_proba = b.take_along_axis(yp, idx_class_pred_ranking, axis=-1)
    sorted_cum_mass = b.cumsum(sorted_proba, axis=-1)

    # Cumulative probability mass of (sorted by descending probability) classes
    # penalized by the regularization term
    rank = b.arange(1, n_classes + 1)
    rank_zero = rank - rank
    penal_cum_proba = sorted_cum_mass + lambd * b.maximum(rank - k_reg, rank_zero)

    # L is the number of classes (+1) for which the cumulative probability mass
    # is below the threshold "tau"
    # The minimum is used in case the Y_pred logits are not well normalized
    below_tau = b.astype(penal_cum_proba < tau, "int64")
    L = b.minimum(b.sum(below_tau, axis=-1) + 1, pred_len)

    if not rand:
        # Build prediction set
        idx_np = b.to_numpy(idx_class_pred_ranking)
        l_np = [int(v) for v in b.to_numpy(L).tolist()]
        prediction_sets = [list(idx_np[i, : l_np[i]]) for i in range(pred_len)]
        return (prediction_sets,)

    ## The following code runs only when rand argument is True
    # For indexing, use L-1 to denote the L-th element
    l_minus_1 = L - 1
    l_col = b.reshape(l_minus_1, (-1, 1))

    # Residual of cumulative probability mass (regularized) and tau
    cum_at_l = b.squeeze(b.take_along_axis(sorted_cum_mass, l_col, axis=-1))
    proba_at_l = b.squeeze(b.take_along_axis(sorted_proba, l_col, axis=-1))
    reg_at_l = b.maximum(L - k_reg, L - L)
    proba_excess = cum_at_l + lambd * reg_at_l - tau

    # Indicator when regularization rank is exceeded
    indic_L_greater_kreg = b.where(L > k_reg, 1, 0)

    # Normalized probability mass excess
    v = proba_excess / (proba_at_l + lambd * indic_L_greater_kreg)

    # Least likely class in the prediction set is removed
    # with a probability of norm_proba_excess
    indic_excess_proba = b.where(u <= v, 1, 0)
    L = L - indic_excess_proba

    # Build prediction set
    idx_np = b.to_numpy(idx_class_pred_ranking)
    l_np = [int(v) for v in b.to_numpy(L).tolist()]
    prediction_sets = [list(idx_np[i, : l_np[i]]) for i in range(pred_len)]

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

    def _raps_set_function(Y_pred, scores_quantile):
        return raps_set(
            Y_pred, scores_quantile, lambd=lambd, k_reg=k_reg, rand=rand
        )

    return _raps_set_function


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Regression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def constant_interval(
    y_pred: Iterable, scores_quantile
):
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
    b = get_backend(y_pred)
    y_pred = b.asarray(y_pred)
    q = b.asarray(scores_quantile)
    y_lo = y_pred - q
    y_hi = y_pred + q
    return y_lo, y_hi


def scaled_interval(
    Y_pred: Iterable, scores_quantile, weights=1, eps: float = 1e-12
):
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
    :param ndarray weights: weights to apply to the size of the interval. 
    :param float eps: small positive value to avoid singleton sets.

    :returns: scaled prediction intervals :math:`I`.
    :rtype: Tuple[ndarray]

    :raises TypeError: unsupported data types.
    """

    b = get_backend(Y_pred)
    Yp = b.asarray(Y_pred)

    try:
        supported_meanvar_models_shape_check(Y_pred)
        y_pred, sigma_pred = split_columns(Yp, (0, 1))

    except Exception:
        supported_types_check(Y_pred)
        y_pred, sigma_pred = Yp, b.ones_like(Yp)

    if b.any(sigma_pred + eps <= 0):
        print(
            "Warning: test points with MAD predictions below -eps"
            " will have infinite sized prediction intervals."
        )

    fints = sigma_pred + eps > 0
    q = b.asarray(scores_quantile)
    y_lo = b.where(fints, y_pred - q * (sigma_pred + eps) * weights, -float("inf"))
    y_hi = b.where(fints, y_pred + q * (sigma_pred + eps) * weights, float("inf"))
    return y_lo, y_hi

    # supported_types_check(Y_pred)

    # b = get_backend(Y_pred)
    # yp = b.asarray(Y_pred)

    # ndim, shape = shape2(yp)
    # if ndim != 2 or shape[1] != 2:
    #     raise RuntimeError(
    #         "Each Y_pred must contain a point prediction and a dispersion estimation."
    #     )

    # y_pred, sigma_pred = split_columns(yp, (0, 1))

    # if b.any(sigma_pred + eps <= 0):
    #     print(
    #         "Warning: test points with MAD predictions below -eps"
    #         " will have infinite sized prediction intervals."
    #     )

    # fints = sigma_pred + eps > 0
    # q = b.asarray(scores_quantile)
    # y_lo = b.where(fints, y_pred - q * (sigma_pred + eps), -float("inf"))
    # y_hi = b.where(fints, y_pred + q * (sigma_pred + eps), float("inf"))
    # return y_lo, y_hi

def cqr_interval(
    Y_pred: Iterable, scores_quantile
):
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

    b = get_backend(Y_pred)
    yp = b.asarray(Y_pred)

    ndim, shape = shape2(yp)
    if ndim != 2 or shape[1] != 2:
        raise RuntimeError(
            "Each Y_pred must contain lower and higher quantiles predictions, respectively."
        )

    q_lo, q_hi = split_columns(yp, (0, 1))
    q = b.asarray(scores_quantile)

    y_lo = q_lo - q
    y_hi = q_hi + q
    return y_lo, y_hi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Object Detection ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def constant_bbox(Y_pred, scores_quantile):
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
    b = get_backend(Y_pred)
    yp = b.asarray(Y_pred)

    ndim, shape = shape2(yp)
    if ndim != 2 or shape[1] != 4:
        raise RuntimeError("Each Y_pred must contain 4 bbox coordinates.")

    x_min, y_min, x_max, y_max = split_columns(yp, (0, 1, 2, 3), keepdims=True)

    q0 = b.scalar_at(scores_quantile, 0)
    q1 = b.scalar_at(scores_quantile, 1)
    q2 = b.scalar_at(scores_quantile, 2)
    q3 = b.scalar_at(scores_quantile, 3)
    x_min_lo, y_min_lo = x_min - q0, y_min - q1
    x_max_hi, y_max_hi = x_max + q2, y_max + q3
    y_pred_hi = concat_columns([x_min_lo, y_min_lo, x_max_hi, y_max_hi], like=yp)

    x_min_hi, y_min_hi = x_min + q0, y_min + q1
    x_max_lo, y_max_lo = x_max - q2, y_max - q3
    y_pred_lo = concat_columns([x_min_hi, y_min_hi, x_max_lo, y_max_lo], like=yp)

    return y_pred_lo, y_pred_hi


def scaled_bbox(Y_pred, scores_quantile):
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
    b = get_backend(Y_pred)
    yp = b.asarray(Y_pred)

    ndim, shape = shape2(yp)
    if ndim != 2 or shape[1] != 4:
        raise RuntimeError("Each Y_pred must contain 4 bbox coordinates.")

    # Recover each coordinate
    x_min, y_min, x_max, y_max = split_columns(yp, (0, 1, 2, 3), keepdims=True)

    # Compute width and height of predicted bbox
    dx = b.abs(x_max - x_min)
    dy = b.abs(y_max - y_min)

    # Coordinates of covering bbox (upperbounds)
    q0 = b.scalar_at(scores_quantile, 0)
    q1 = b.scalar_at(scores_quantile, 1)
    q2 = b.scalar_at(scores_quantile, 2)
    q3 = b.scalar_at(scores_quantile, 3)
    x_min_lo = x_min - q0 * dx
    y_min_lo = y_min - q1 * dy
    x_max_hi = x_max + q2 * dx
    y_max_hi = y_max + q3 * dy

    y_pred_outer = concat_columns([x_min_lo, y_min_lo, x_max_hi, y_max_hi], like=yp)

    # Coordinates of included bbox (lowerbounds)
    x_min_hi = x_min + q0 * dx
    y_min_hi = y_min + q1 * dy
    x_max_lo = x_max - q2 * dx
    y_max_lo = y_max - q3 * dy
    y_pred_inner = concat_columns([x_min_hi, y_min_hi, x_max_lo, y_max_lo], like=yp)

    return y_pred_inner, y_pred_outer

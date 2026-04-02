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
from typing import Callable
from typing import Iterable

from deel.puncc.api.backend import concat_columns, get_backend, shape2, split_columns

from deel.puncc.api.utils import supported_types_check
from deel.puncc.api.utils import supported_dual_models_shape_check
from deel.puncc.api.utils import supported_meanvar_models_shape_check

from deel.puncc.api.utils import supported_bbox_shape_check
from deel.puncc.api.utils import logit_normalization_check


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Classification ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def lac_score(
    Y_pred: Iterable,
    y_true: Iterable,
) -> Iterable:
    """LAC nonconformity score.

    :param Iterable Y_pred:
        :math:`Y_{\\text{pred}} = (P_{\\text{C}_1}, ..., P_{\\text{C}_n})`
        where :math:`P_{\\text{C}_i}` is logit associated to class i.
    :param Iterable y_true: true labels.

    :returns: LAC nonconformity scores.
    :rtype: Iterable

    :raises TypeError: unsupported data types.
    """
    supported_types_check(Y_pred, y_true)

    # Check if logits sum is close to one
    logit_normalization_check(Y_pred)

    b = get_backend(Y_pred, y_true)
    yp = b.asarray(Y_pred)
    yt = b.astype(b.squeeze(b.asarray(y_true)), "int64")

    # Gather predicted probability at true-label index for each sample.
    p_true = b.take_along_axis(yp, b.reshape(yt, (-1, 1)), axis=1)
    return 1 - b.squeeze(p_true)


def classwise_lac_score(
    Y_pred: Iterable,
    y_true: Iterable,
) -> Iterable:
    """Classwise LAC nonconformity score.

    Computes nonconformity scores for classwise conformal prediction.
    For each sample, the score is stored only for its true class,
    with NaN values for other classes. This allows computing
    per-class quantiles during calibration.

    :param Iterable Y_pred:
        :math:`Y_{\\text{pred}} = (P_{\\text{C}_1}, ..., P_{\\text{C}_n})`
        where :math:`P_{\\text{C}_i}` is logit associated to class i.
    :param Iterable y_true: true labels.

    :returns: Classwise LAC nonconformity scores of shape (n_samples, n_classes).
        Entry [i, c] contains `1 - Y_pred[i, c]` if `y_true[i] == c`, else NaN.
    :rtype: Iterable

    :raises TypeError: unsupported data types.
    """
    supported_types_check(Y_pred, y_true)

    # Check if logits sum is close to one
    logit_normalization_check(Y_pred)

    b = get_backend(Y_pred, y_true)
    yp = b.asarray(Y_pred)
    yt = b.astype(b.squeeze(b.asarray(y_true)), "int64")

    _, shape = shape2(yp)
    n_samples = shape[0]
    n_classes = shape[1]

    # Build a boolean mask selecting the true class per sample.
    yt_col = b.reshape(yt, (-1, 1))
    class_ids = b.reshape(b.arange(n_classes), (1, n_classes))
    true_class_mask = b.equal(yt_col, class_ids)

    # Keep 1 - p only on the true class column, NaN elsewhere.
    nan_matrix = b.full((n_samples, n_classes), float("nan"))
    return b.where(true_class_mask, 1 - yp, nan_matrix)


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

    Backend-agnostic implementation relying on backend abstractions.

    """
    supported_types_check(Y_pred, y_true)

    # Check if logits sum is close to one
    logit_normalization_check(Y_pred)

    b = get_backend(Y_pred, y_true)
    yp = b.asarray(Y_pred)
    yt = b.astype(b.squeeze(b.asarray(y_true)), "int64")

    # Sort classes by descending probability order.
    class_ranking = b.argsort(yp, axis=1, descending=True)
    sorted_proba = b.take_along_axis(yp, class_ranking, axis=1)

    # Cumulative probability mass (given the previous class ranking).
    sorted_cum_mass = b.cumsum(sorted_proba, axis=1)

    # Locate the rank position of the true class for each sample.
    yt_col = b.reshape(yt, (-1, 1))
    matches = b.equal(class_ranking, yt_col)
    L = b.argmax(matches, axis=1)

    # Threshold of cumulative probability mass to include the real class.
    e_col = b.take_along_axis(sorted_cum_mass, b.reshape(L, (-1, 1)), axis=1)
    E = b.squeeze(e_col)

    # Regularization term max(L + 1 - k_reg, 0) in backend-native form.
    reg_rank = L + 1 - k_reg
    zero_like = reg_rank - reg_rank
    reg_term = b.maximum(reg_rank, zero_like)

    if rand:
        n_samples = shape2(yp)[1][0]
        u = b.random_uniform((n_samples,))
        p_col = b.take_along_axis(sorted_proba, b.reshape(L, (-1, 1)), axis=1)
        p_at_l = b.squeeze(p_col)
        E = E + lambd * reg_term - u * p_at_l
    else:
        E = E + lambd * reg_term

    return E


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
    :param bool rand: turn on or off the randomization term that smoothes the
        discrete probability mass jump when including a new class.

    :returns: RAPS nonconformity score function that takes two parameters:
        `Y_pred` and `y_true`.
    :rtype: Callable

    :raises TypeError: unsupported data types.
    """

    def _raps_score_function(Y_pred: Iterable, y_true: Iterable) -> Iterable:
        return raps_score(Y_pred, y_true, lambd, k_reg, rand)

    return _raps_score_function


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Regression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def difference(y_pred: Iterable, y_true: Iterable) -> Iterable:
    """Elementwise difference.

    .. math::

        R = y_{\\text{pred}}-y_{\\text{true}}

    :param Iterable y_pred: predictions.
    :param Iterable y_true: true labels.

    :returns: coordinatewise difference.
    :rtype: Iterable

    :raises TypeError: unsupported data types.
    """
    supported_types_check(y_pred, y_true)
    b = get_backend(y_pred, y_true)
    yp = b.asarray(y_pred)
    yt = b.asarray(y_true)
    return b.squeeze(yp) - b.squeeze(yt)


def absolute_difference(y_pred: Iterable, y_true: Iterable) -> Iterable:
    """Absolute Deviation.

    .. math::

        R = |y_{\\text{true}}-y_{\\text{pred}}|

    :param Iterable y_pred: predictions.
    :param Iterable y_true: true labels.

    :returns: mean absolute deviation.
    :rtype: Iterable

    :raises TypeError: unsupported data types.
    """
    b = get_backend(y_pred, y_true)
    return b.abs(difference(y_pred, y_true))


def scaled_ad(
    Y_pred: Iterable, y_true: Iterable, eps: float = 1e-12
) -> Iterable:
    """Scaled Absolute Deviation, normalized by an estimation of the conditional
    mean absolute deviation (conditional MAD). Considering
    :math:`Y_{\\text{pred}} = (\mu_{\\text{pred}}, \sigma_{\\text{pred}})`:

    .. math::

        R = \\frac{|y_{\\text{true}}-\mu_{\\text{pred}}|}{\sigma_{\\text{pred}}}

    :param Iterable Y_pred: point and conditional MAD predictions.
        :math:`Y_{\\text{pred}}=(y_{\\text{pred}}, \sigma_{\\text{pred}})`
    :param Iterable y_true: true labels.
    :param float eps: small positive value to avoid division by negative or zero.

    :returns: scaled absolute deviation.
    :rtype: Iterable

    :raises TypeError: unsupported data types.
    """
    supported_dual_models_shape_check(Y_pred, y_true)

    b = get_backend(Y_pred, y_true)
    Yp = b.asarray(Y_pred)
    yt = b.asarray(y_true)

    y_pred, sigma_pred = split_columns(Yp, (0, 1))
    mad = b.abs(y_pred - yt)
    if b.any(sigma_pred + eps <= 0):
        print(
            "Warning: calibration points with MAD predictions below -eps "
            " won't be used for calibration."
        )

    nonneg = sigma_pred + eps > 0
    return mad[nonneg] / (sigma_pred[nonneg] + eps)

def weighted_scaled_ad(
    X, Y_pred: Iterable, y_true: Iterable, weight_func: Callable,
    eps: float = 1e-12
) -> Iterable:
    """Weighted Scaled Absolute Deviation. Similar to :func:`scaled_ad`
    but with an extra weighting of the nonconformity scores by a function of X.
    This allows to give more importance to calibration points, as measured by
    weight function.
    Considering :math:`Y_{\\text{pred}} = (\mu_{\\text{pred}}, \sigma_{\\text{pred}})`:

    .. math::
        R = \\frac{|y_{\\text{true}}-\mu_{\\text{pred}}|}{\sigma_{\\text{pred}}} *
        w(X)

        where :math:`w(X)` is the weight function applied to the inputs.

    :param Iterable Y_pred: point predictions. Optionally with conditional MAD
        predictions if available.
    :param Iterable y_true: true labels.
    :param Callable weight_func: function of X that computes the weighted scores.
    :param float eps: small positive value to avoid division by negative or zero.

    :returns: weighted scaled absolute deviation.
    :rtype: Iterable

    :raises TypeError: unsupported data types.
    """
    # Models should either predict only the point estimate or both the point
    # estimate and the conditional MAD.

    b = get_backend(Y_pred, y_true)
    yt = b.asarray(y_true)
    Yp = b.asarray(Y_pred)

    try:
        supported_meanvar_models_shape_check(Y_pred, y_true)
        y_pred, sigma_pred = split_columns(Yp, (0, 1))

    except Exception:
        supported_types_check(Y_pred, y_true)
        y_pred, sigma_pred = Yp, b.ones_like(Yp)
    mad = b.abs(y_pred - yt)
    if b.any(sigma_pred + eps <= 0):
        print(
            "Warning: calibration points with MAD predictions below -eps "
            " won't be used for calibration."
        )

    nonneg = sigma_pred + eps > 0
    weights = weight_func(X[nonneg]) if weight_func is not None else 1
    return mad[nonneg] / (sigma_pred[nonneg] + eps) * weights

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

    :raises TypeError: unsupported data types.
    """
    supported_dual_models_shape_check(Y_pred, y_true)

    b = get_backend(Y_pred, y_true)
    Yp = b.asarray(Y_pred)
    yt = b.asarray(y_true)

    q_lo, q_hi = split_columns(Yp, (0, 1))
    diff_lo = q_lo - yt
    diff_hi = yt - q_hi
    return b.maximum(diff_lo, diff_hi)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Object Detection ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def scaled_bbox_difference(Y_pred: Iterable, Y_true: Iterable):
    """Object detection scaled nonconformity score. Considering
    :math:`Y_{\\text{pred}} = (\hat{x}_1, \hat{y}_1, \hat{x}_2, \hat{y}_2)` and
    :math:`Y_{\\text{true}} = (x_1, y_1, x_2, y_2)`:

    .. math::

        R = (\\frac{\hat{x}_1-x_1}{\Delta x}, \\frac{\hat{y}_1-y_1}{\Delta y},
        \\frac{\hat{x}_2-x_2}{\Delta x}, \\frac{\hat{y}_1-y_1}{\Delta y})

    where :math:`\Delta x = |\hat{x}_2 - \hat{x}_1|` and
    :math:`\Delta y = |\hat{y}_2 - \hat{y}_1|`.

    :param Iterable Y_pred: predicted coordinates of the bounding box.
    :param Iterable y_true: true coordinates of the bounding box.

    :returns: scaled object detection nonconformity scores.
    :rtype: Iterable

    :raises TypeError: unsupported data types.
    """
    supported_bbox_shape_check(Y_pred, Y_true)
    b = get_backend(Y_pred, Y_true)
    yp = b.asarray(Y_pred)
    yt = b.asarray(Y_true)

    x_min, y_min, x_max, y_max = split_columns(yp, (0, 1, 2, 3), keepdims=True)
    dx = b.abs(x_max - x_min)
    dy = b.abs(y_max - y_min)
    scale = concat_columns([dx, dy, dx, dy], like=yp)

    return (yp - yt) / scale

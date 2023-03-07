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
This module implements utility functions.
"""
import sys
from typing import Iterable
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

# import tensorflow as tf
# import torch

EPSILON = sys.float_info.min  # small value to avoid underflow


def dual_predictor_check(l, name, type):
    if (not isinstance(l, list)) or len(l) != 2:
        raise TypeError(f"Argument `{name}` should be a list of two {type}.")


def supported_types_check(y_pred, y_true=None):

    if y_true is not None and (type(y_pred) != type(y_true)):
        raise TypeError(
            f"elements do not have the same type: {type(y_pred)} vs {type(y_true)}."
        )

    if isinstance(y_pred, np.ndarray):
        pass
    else:
        import pandas as pd

        if isinstance(y_pred, pd.DataFrame):
            pass

        else:
            import tensorflow as tf
            import torch

            if isinstance(y_pred, tf.Tensor) or isinstance(y_pred, torch.Tensor):
                pass

            else:
                raise TypeError(
                    "Unsupported data type. Please provide a numpy ndarray, a dataframe or a tensor (TensorFlow|torch)."
                )


def check_alpha_calib(alpha: float, n: int, complement_check: bool = False):
    """Check if the value of alpha :math:`\\alpha` is consistent with the size of calibration set :math:`n`.

    The quantile order is inflated by a factor :math:`(1+1/n)` and has to be in the interval (0,1). From this, we derive the condition:

    .. math::

        0 < (1-\\alpha)\cdot(1+1/n) < 1 \implies 1 > \\alpha > 1/(n+1)

    If complement_check is set, we consider an additional condition:

    .. math::

        0 < \\alpha \cdot (1+1/n) < 1 \implies 0 < \\alpha < n/(n+1)

    :param float alpha: target quantile order.
    :param int n: size of the calibration dataset.
    :param bool complement_check: complementary check to compute the :math:`\\alpha \cdot (1+1/n)`-th quantile, required by some methods such as jk+.

    :raises ValueError: when :math:`\\alpha` is inconsistent with the size of calibration set.
    """

    if alpha < 1 / (n + 1):
        raise ValueError(
            f"Alpha is too small: (alpha={alpha}, n={n}) {alpha} < {1/(n+1)}. "
            + "Increase alpha or the size of the calibration set."
        )

    if alpha >= 1:
        raise ValueError(
            f"Alpha={alpha} is too large. Decrease alpha such that alpha < 1."
        )

    if complement_check and alpha > n / (n + 1):
        raise ValueError(
            f"Alpha is too large: (alpha={alpha}, n={n}) {alpha} < {n/(n+1)}. "
            + "Decrease alpha or increase the size of the calibration set."
        )

    if complement_check and alpha <= 0:
        raise ValueError(
            f"Alpha={alpha} is too small. Increase alpha such that alpha > 0."
        )


def get_min_max_alpha_calib(
    n: int, two_sided_conformalization: bool = False
) -> Optional[Tuple[float, float]]:
    """Get the greatest alpha in (0,1) that is consistent with the size of calibration
    set. Recall that while true Conformal Prediction (CP, Vovk et al 2005) is defined only on (0,1)
    (bounds not included), there maybe cases where we must handle alpha=0 or 1
    (e.g. Adaptive Conformal Inference, Gibbs & Candès 2021).

    This function computes the admissible range of alpha for split CP (Papadopoulos et al 2002,
    Lei et al 2018) and the Jackknife-plus (Barber et al 2019).
    For more CP method, see the literature and update this function.

    In Split Conformal prediction we take the index := ceiling( (1-alpha)(n+1) )-th element
    from the __sorted__ nonconformity scores as the margin of error.
    We must have 1 <= index <= n ( = len(calibration_set)) which boils down to: 1/(n+1) <= alpha < 1.

    In the case of cv-plus and jackknife-plus (Barber et al 2019), we need:
        1. ceiling( (1-alpha)(n+1) )-th element of scores
        2. floor(     (alpha)(n+1) )-th element of scores

    The argument two_sided_conformalization=True, ensures the indexes are within
    range for this special case: 1/(n+1) <= alpha <= n/(n+1)

    :param int n: size of the calibration dataset.
    :param bool two_sided_conformalization: alpha threshold for two-sided quantile of jackknife+ (Barber et al 2021).

    :returns: lower and upper bounds for alpha (miscoverage probability)
    :rtype: Tuple[float, float]

    :raises ValueError: must have integer n, boolean two_sided_conformalization and n>=1
    """

    if n < 1:
        raise ValueError(f"Invalid input: you need n>=1 but received n={n}")

    if two_sided_conformalization is True:
        return (1 / (n + 1), n / (n + 1))
    # Base case: split conformal prediction (e.g. Lei et al 2019)
    else:
        return (1 / (n + 1), 1)


def quantile(a: Iterable, q: float, w: np.ndarray = None):  # type: ignore
    """Estimate the q-th empirical weighted quantiles.

    :param ndarray|DataFrame|Tensor a: collection of n samples
    :param float q: target quantile order. Must be in the open interval (0, 1).
    :param ndarray w: vector of size n. By default, w is None and equal weights (:math:`1/n`) are associated.

    :returns: weighted empirical quantiles.
    :rtype: ndarray

    :raises NotImplementedError: `a` must be unidimensional.
    """
    # type checks:
    supported_types_check(a)

    if isinstance(a, np.ndarray):
        pass
    else:
        import pandas as pd

        if isinstance(a, pd.DataFrame):
            a = a.to_numpy()
        else:
            import tensorflow as tf
            import torch

            if isinstance(a, tf.Tensor):
                a = a.numpy()
            elif isinstance(a, torch.Tensor):
                a = a.cpu().detach().numpy()
            else:
                raise RuntimeError("Fatal error.")

    # Sanity checks
    if q <= 0 or q >= 1:
        raise ValueError("q must be in the open interval (0, 1).")

    if a.ndim > 1:
        raise NotImplementedError(f"a dimension {a.ndim} should be 1.")

    if w is not None and w.ndim > 1:
        raise NotImplementedError(f"w dimension {w.ndim} should be 1.")

    # Case of None weights
    if w is None:
        return np.quantile(a, q=q, method="inverted_cdf")
        ## An equivalent method would be to assign equal values to w
        ## and carry on with the computations.
        ## np.quantile is however more optimized.
        # w = np.ones_like(a) / len(a)

    # Sanity check
    if len(w) != len(a):
        error = "a and w must have the same shape:" + f"{len(a)} != {len(w)}"
        raise RuntimeError(error)

    # Normalization check
    norm_condition = np.isclose(np.sum(w, axis=-1), 1, atol=1e-6)
    if ~np.all(norm_condition):
        error = (
            f"W is not normalized. Sum of weights on" + f"rows is {np.sum(w, axis=-1)}"
        )
        raise RuntimeError(error)

    # Empirical Weighted Quantile
    sorted_idxs = np.argsort(a)  # rows are sorted in ascending order
    sorted_cumsum_w = np.cumsum(w[sorted_idxs])
    weighted_quantile_idxs = sorted_idxs[sorted_cumsum_w >= q][0]
    return a[weighted_quantile_idxs]


#
# ========================= Aggregation functions =========================
#


def agg_list(a: np.ndarray):
    """Ancillary function to aggregate array following the axis 0.

    :param ndarray a: array.

    :returns: the concatenated array or None
    :rtype: ndarray or None
    """

    try:
        return np.concatenate(a, axis=0)
    except ValueError:
        return None

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
import logging
import pkgutil
import sys
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Tuple

import numpy as np

if pkgutil.find_loader("pandas") is not None:
    import pandas as pd

if pkgutil.find_loader("tensorflow") is not None:
    import tensorflow as tf

if pkgutil.find_loader("torch") is not None:
    import torch

logger = logging.getLogger(__name__)

EPSILON = sys.float_info.min  # small value to avoid underflow


def dual_predictor_check(l: Any, name: str, ltype: str):
    """Check if properties of `DualPredictor` come in
    couples.

    :param Any l: property of `DualPredictor`.
    :param str name: name of the property.
    :param str ltype: type of property.

    :raises TypeError: when the properties count is not two.

    """
    if (not isinstance(l, list)) or len(l) != 2:
        raise TypeError(
            f"Argument `{name}` should be a list of two {ltype}. Provided {l}."
        )


def logit_normalization_check(y: Iterable):
    """Check if provided logits sum is close to one.

    :param Iterable y: logits array. Rows correspond to samples and columns to
        classes.

    :raises ValueError: when logits sum is different than one within a tolerance
        value of 1e-5.
    """
    logits_sum = np.sum(np.array(list(y)), -1)
    if np.any(np.abs(logits_sum - 1) > 1e-5):
        raise ValueError(f"Logits must some to 1. Provided logit array {logits_sum}")


def sample_len_check(a: Iterable, b: Iterable):
    """Check if arguments type have the same length.

    :param Iterable a: iterable whose type is supported.
    :param Iterable b: iterable whose type is supported.

    :raises TypeError: unsupported data types or elements have inconsistent types.
    :raises ValueError: arguments contain different number of samples.
    """
    supported_types_check(a, b)
    if a.shape[0] != b.shape[0]:
        raise ValueError("Iterables must contain the same number of samples.")


def features_len_check(a: Iterable, b: Iterable):
    """Check if arguments have the same number of features,
    that is their last axes have the same length.

    :param Iterable a: iterable whose type is supported.
    :param Iterable b: iterable whose type is supported.

    :raises TypeError: unsupported data types or elements have inconsistent types.
    :raises ValueError: arguments have different number of features
    """

    supported_types_check(a, b)

    if a.shape[-1] != b.shape[-1]:
        raise ValueError("X_fit and X_calib must contain the same number of features.")


def supported_types_check(*data: Iterable):
    """Check if arguments' types are supported.

    :param Iterable data: iterable(s) to be checked.

    :raises TypeError: unsupported data types.
    """

    for a in data:
        if isinstance(a, np.ndarray):
            pass

        elif pkgutil.find_loader("pandas") is not None and isinstance(
            a, (pd.DataFrame, pd.Series)
        ):
            pass
        elif pkgutil.find_loader("tensorflow") is not None and isinstance(a, tf.Tensor):
            pass
        elif pkgutil.find_loader("torch") is not None and isinstance(a, torch.Tensor):
            pass
        else:
            raise TypeError(
                f"Unsupported data type {type(a)}. Please provide a numpy ndarray, "
                "a dataframe or a tensor."
            )


def alpha_calib_check(alpha: float, n: int, complement_check: bool = False):
    """Check if the value of alpha :math:`\\alpha` is consistent with the size
    of calibration set :math:`n`.

    The quantile order is inflated by a factor :math:`(1+1/n)` and has to be in
    the interval (0,1). From this, we derive the condition:

    .. math::

        0 < (1-\\alpha)\cdot(1+1/n) < 1 \implies 1 > \\alpha > 1/(n+1)

    If complement_check is set, we consider an additional condition:

    .. math::

        0 < \\alpha \cdot (1+1/n) < 1 \implies 0 < \\alpha < n/(n+1)

    :param float alpha: target quantile order.
    :param int n: size of the calibration dataset.
    :param bool complement_check: complementary check to compute the
        :math:`\\alpha \cdot (1+1/n)`-th quantile, required by some methods
        such as jk+.

    :raises ValueError: when :math:`\\alpha` is inconsistent with the size of
        calibration set.
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
) -> Optional[Tuple[float]]:
    """Get the greatest alpha in (0,1) that is consistent with the size of
    calibration set. Recall that while true Conformal Prediction
    (CP, Vovk et al 2005) is defined only on (0,1) (bounds not included), there
    maybe cases where we must handle alpha=0 or 1 (e.g. Adaptive Conformal
    Inference, Gibbs & Candès 2021).

    This function computes the admissible range of alpha for split CP
    (Papadopoulos et al 2002, Lei et al 2018) and the Jackknife-plus
    (Barber et al 2019). For more CP method, see the literature and update this
    function.

    In Split Conformal prediction we take the index
    := ceiling( (1-alpha)(n+1) )-th element from the __sorted__ nonconformity
    scores as the margin of error. We must have
    1 <= index <= n ( = len(calibration_set)) which boils down to:
    1/(n+1) <= alpha < 1.

    In the case of cv-plus and jackknife-plus (Barber et al 2019), we need:
        1. ceiling( (1-alpha)(n+1) )-th element of scores
        2. floor(     (alpha)(n+1) )-th element of scores

    The argument two_sided_conformalization=True, ensures the indexes are within
    range for this special case: 1/(n+1) <= alpha <= n/(n+1)

    :param int n: size of the calibration dataset.
    :param bool two_sided_conformalization: alpha threshold for two-sided
        quantile of jackknife+ (Barber et al 2021).

    :returns: lower and upper bounds for alpha (miscoverage probability)
    :rtype: Tuple[float]

    :raises ValueError: must have integer n, boolean two_sided_conformalization
        and n>=1
    """

    if n < 1:
        raise ValueError(f"Invalid input: you need n>=1 but received n={n}")

    if two_sided_conformalization is True:
        return (1 / (n + 1), n / (n + 1))

    # Base case: split conformal prediction (e.g. Lei et al 2019)
    return (1 / (n + 1), 1)


def quantile(
    a: Iterable, q: float or np.ndarray, w: np.ndarray = None
) -> np.ndarray:  # type: ignore
    """Estimate the columnwise p-th empirical weighted quantiles.

    :param Iterable a: collection of n samples
    :param float or np.ndarray q: q-th quantiles to compute. All elements must be in (0, 1).
    :param ndarray w: vector of size n. By default, w is None and equal weights
        (:math:`1/n`) are associated.

    :returns: weighted empirical quantiles.
    :rtype: ndarray
    """
    # type checks:
    supported_types_check(a)

    if isinstance(a, np.ndarray):
        pass
    elif pkgutil.find_loader("pandas") is not None and isinstance(a, pd.DataFrame):
        a = a.to_numpy()
    elif pkgutil.find_loader("tensorflow") is not None and isinstance(a, tf.Tensor):
        a = a.numpy()
    # elif pkgutil.find_loader("torch") is not None:
    #     if isinstance(a, torch.Tensor):
    #         a = a.cpu().detach().numpy()
    else:
        raise RuntimeError(
            "Unsupported data type for argument [a].\n"
            "Please provide a numpy ndarray, a pandas dataframe or a tensorflow tensor."
        )

    # Sanity checks
    if np.any(q <= 0) or np.any(q >= 1):
        raise ValueError("All coordinates of [q] must be in the open interval (0, 1).")

    # Dim checks for a and q
    if a.ndim > 2:
        raise ValueError("[a] must be a 1D or 2D array.")

    if a.ndim == 1 and isinstance(q, np.ndarray):
        raise ValueError("[q] must be a scalar when a is a 1D array.")

    if a.ndim == 1 and w is None:
        return quantile_unidim(a, q)

    if a.ndim == 1 and w.shape != a.shape:
        raise ValueError("[a] and [w] must have the same shape when [a] is a 1D array.")

    if a.ndim == 1:
        return quantile_unidim_weighted(a, q, w)

    if a.ndim == 2 and isinstance(q, float):
        q = np.repeat(q, a.shape[1])

    if a.ndim == 2 and q.shape != (a.shape[1],):
        raise ValueError(
            "[q] must be a float or have the same shape as the number of columns of [a]."
        )

    if a.ndim == 2 and w is None:
        return quantile_multidim(a, q)

    if a.ndim == 2 and w.shape == (a.shape[0],):
        w = np.tile(w, (a.shape[1], 1)).T
        print("shape of a is", a.shape)
        print("shape of w is", w.shape)

    if a.ndim == 2 and w.shape != a.shape:
        raise ValueError(
            "weights must be of shape (n,) or (n, m) where (n,m) is the shape of a."
        )

    return quantile_multidim_weighted(a, q, w)


def quantile_unidim(a: np.ndarray, q: float) -> float:
    """Estimate the q-th empirical quantile of a 1D array.

    :param ndarray a: collection of n samples
    :param float q: q-th quantile to compute. Must be in the open interval (0, 1).

    :returns: empirical quantile.
    :rtype: float
    """
    return np.quantile(a, q, method="inverted_cdf")


def quantile_unidim_weighted(a: np.ndarray, q: float, w: np.ndarray) -> float:
    """Estimate the q-th empirical weighted quantile of a 1D array.

    :param ndarray a: collection of n samples
    :param float q: q-th quantile to compute. Must be in the open interval (0, 1).
    :param ndarray w: vector of size n.

    :returns: weighted empirical quantile.
    :rtype: float
    """

    # Normalization check
    norm_condition = np.isclose(np.sum(w) - 1, 0, atol=1e-14)
    if ~norm_condition:
        error = f"w is not normalized. Sum of weights is {np.sum(w)}"
        raise RuntimeError(error)

    # Values are sorted in ascending order
    sorted_idxs = np.argsort(a)
    logger.debug("Sorted indices: %s", sorted_idxs)

    # Reorder weights (ascending column values of a) and compute cumulative sum
    sorted_cumsum_weights = np.cumsum(w[sorted_idxs])
    logger.debug("Sorted weights cumulative sum: %s", sorted_cumsum_weights)

    # Get the smallest index for which the cumulative sum of weights exceeds p
    min_idx_reaching_q = np.sum(sorted_cumsum_weights < q)
    logger.debug(
        "First index per column where cumsum exceeds q: %s", min_idx_reaching_q
    )

    # Collect the p-th quantile (first value whose probability mass exceeds p)
    quantile_res = a[sorted_idxs[min_idx_reaching_q]]
    logger.debug("Quantiles array: %s", quantile_res)

    return quantile_res


def quantile_multidim(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Estimate the columnwise q-th empirical quantiles of a 2D array.

    :param ndarray a: collection of n samples
    :param ndarray q: q-th quantiles to compute, array of shape (a[1],). All elements must be in the open interval (0, 1).

    :returns: empirical quantiles.
    :rtype: ndarray
    """
    return np.array(
        [np.quantile(col, q=proba, method="inverted_cdf") for col, proba in zip(a.T, q)]
    )


def quantile_multidim_weighted(
    a: np.ndarray, q: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Estimate the columnwise q-th empirical weighted quantiles of a 2D array.

    :param ndarray a: collection of n samples
    :param ndarray q: q-th quantiles to compute, array of shape (a[1],). All elements must be in the open interval (0, 1).
    :param ndarray w: matrix of the same shape as a.

    :returns: weighted empirical quantiles.
    :rtype: ndarray
    """

    # Normalization check
    norm_condition = np.isclose(np.sum(w, axis=0) - 1, 0, atol=1e-14)
    if ~np.all(norm_condition):
        error = (
            "w is not normalized. Sum of weights on"
            + f"columns is {np.sum(w, axis=0)}"
        )
        raise RuntimeError(error)

    # Empirical Weighted Quantile
    return np.array(
        [
            quantile_unidim_weighted(col, proba, weight)
            for col, proba, weight in zip(a.T, q, w.T)
        ]
    )

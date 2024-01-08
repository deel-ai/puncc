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
This module provides conformal prediction metrics.
"""
from typing import Tuple

import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Classification ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def classification_mean_coverage(
    y_true: np.ndarray, set_pred: Tuple[np.ndarray]
) -> float:
    """Compute empirical coverage of the prediction sets.

    Given the :math:`i`-th prediction set :math:`S(X_i)`, the coverage is:

    * :math:`cov(X_i) = 1` if :math:`y_{true} \in S(X_i)`
    * :math:`cov(X_i) = 0` otherwise

    With N the number of examples, the average coverage is
    :math:`1/N \sum_{i=1}^{N} cov(X_i)`.

    :param np.ndarray y_true: Observed label
    :param Tuple[np.ndarray] set_pred: label prediction set

    :returns: average coverage, indicating the proportion of instances that are
        correctly covered.
    :rtype: float
    """
    counter = 0
    for y, S in zip(y_true, set_pred):
        if (S != []) and (y in S):
            counter += 1
    return counter / len(y_true)


def classification_mean_size(set_pred: Tuple[np.ndarray]) -> float:
    """Compute average size of the prediction sets.

    :param Tuple[np.ndarray] set_pred: label prediction set

    :returns: Average size of the prediction sets
    :rtype: float
    """
    return np.mean([len(s) for s in set_pred])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Regression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def regression_mean_coverage(y_true, y_pred_lower, y_pred_upper) -> float:
    """Compute average coverage on several prediction intervals.

    Given the :math:`i`-th prediction interval :math:`C(X_i)`, the coverage is:

        * :math:`cov(X_i) = 1` if :math:`y_{true} \in C(X_i)`
        * :math:`cov(X_i) = 0` otherwise

    With N the number of examples, the average coverage is
    :math:`1/N \sum_{i=1}^{N} cov(X_i)`.

    :param ndarray y_true: label true values.
    :param ndarray y_pred_lower: lower bounds of the prediction intervals.
    :param ndarray y_pred_upper: upper bounds of the prediction intervals.

    :returns: average coverage, indicating the proportion of instances that are
        correctly covered.
    :rtype: float
    """
    return ((y_true >= y_pred_lower) & (y_true <= y_pred_upper)).mean()


def regression_ace(y_true, y_pred_lower, y_pred_upper, alpha) -> float:
    """Compte the Average Coverage Error (ACE).

    :param ndarray y_true: label true values.
    :param ndarray y_pred_lower: lower bounds of the prediction intervals.
    :param ndarray y_pred_upper: upper bounds of the prediction intervals.
    :param float alpha: significance level (max miscoverage target).

    .. NOTE::

        The ACE is the distance between the nominal coverage :math:`1-\\alpha`
        and the empirical average coverage :math:`AC` such that
        :math:`ACE = AC - (1-\\alpha)`.

        If the ACE is strictly negative, the prediction intervals are
        marginally undercovering. If the ACE is strictly positive, the
        prediction intervals are maginally conservative.

    :returns: the average coverage error (ACE).
    :rtype: float
    """
    cov = regression_mean_coverage(y_true, y_pred_lower, y_pred_upper)
    return cov - (1 - alpha)


def regression_sharpness(y_pred_lower, y_pred_upper) -> float:
    """Compute the average absolute width of the prediction intervals.

    :param ndarray y_pred_lower: lower bounds of the prediction intervals.
    :param ndarray y_pred_upper: upper bounds of the prediction intervals.

    :returns: average absolute width of the prediction intervals.
    :rtype: float
    """
    return (np.abs(y_pred_upper - y_pred_lower)).mean()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Object Detection ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def object_detection_mean_coverage(
    y_pred_outer: np.ndarray, y_true: np.ndarray
):
    """
    Calculate the mean coverage of conformal object detection predictions.
    For each instance, coverage is defined as the true bounding box being inside
    the predicted outer bounding box.

    :param np.ndarray y_pred: array of predicted outer bounding boxes with shape (n, 4).
    :type y_pred: np.ndarray
    :param np.ndarray y_true: array of true bounding boxes with shape (n, 4).

    :return: average coverage, indicating the proportion of objects that are
        correctly covered.
    :rtype: float

    """
    x_min_true, y_min_true, x_max_true, y_max_true = np.hsplit(y_true, 4)
    x_min, y_min, x_max, y_max = np.hsplit(y_pred_outer, 4)
    cov = (
        (x_min <= x_min_true)
        * (y_min <= y_min_true)
        * (x_max >= x_max_true)
        * (y_max >= y_max_true)
    )
    return np.mean(cov)


def object_detection_mean_area(y_pred: np.ndarray):
    """
    Calculate the mean area of object bounding predictions.

    :param np.ndarray y_pred: array of predicted bounding boxes with shape (n, 4).

    :return: average area of the bounding boxes
    :rtype: float

    """
    x_min, y_min, x_max, y_max = np.hsplit(y_pred, 4)
    return np.mean((x_max - x_min) * (y_max - y_min))

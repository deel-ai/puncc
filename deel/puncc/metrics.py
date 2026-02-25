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
from typing import Iterable
from deel.puncc.typing import TensorLike
from deel.puncc import ops

def classification_mean_coverage(
    y_true: TensorLike, set_pred: Iterable[Iterable]
) -> float:
    """
    Compute empirical coverage of the prediction sets.

    Given the :math:`i`-th prediction set :math:`S(X_i)`, the coverage is:

    * :math:`cov(X_i) = 1` if :math:`y_{true} \in S(X_i)`
    * :math:`cov(X_i) = 0` otherwise

    With N the number of examples, the average coverage is
    :math:`1/N \sum_{i=1}^{N} cov(X_i)`.


    Args:
        y_true (TensorLike): Observed label
        set_pred (Iterable[Iterable]): label prediction set

    Returns:
        float: average coverage, indicating the proportion of instances that are
        correctly covered.
    """
    return ops.sum(ops.array([y in s for y, s in zip(y_true, set_pred)])) / len(y_true)

def classification_mean_size(set_pred: Iterable[TensorLike]) -> float:
    """
    Compute average size of the prediction sets.

    Args:
        set_pred (Iterable[TensorLike]): label prediction set

    Returns:
        float: Average size of the prediction sets
    """
    return ops.mean(ops.array([len(s) for s in set_pred]))

def regression_mean_coverage(y_true:TensorLike, y_pred_lower:TensorLike, y_pred_upper:TensorLike) -> float:
    """
    Compute average coverage on several prediction intervals.

    Given the :math:`i`-th prediction interval :math:`C(X_i)`, the coverage is:

        * :math:`cov(X_i) = 1` if :math:`y_{true} \in C(X_i)`
        * :math:`cov(X_i) = 0` otherwise

    With N the number of examples, the average coverage is
    :math:`1/N \sum_{i=1}^{N} cov(X_i)`.

    Args:
        y_true (TensorLike): label true values.
        y_pred_lower (TensorLike): lower bounds of the prediction intervals.
        y_pred_upper (TensorLike): upper bounds of the prediction intervals.

    Returns:
        float: average coverage, indicating the proportion of instances that are
        correctly covered.
    """
    return ops.mean(ops.logical_and(y_true >= y_pred_lower, y_true <= y_pred_upper))

def regression_ace(y_true:TensorLike, y_pred_lower:TensorLike, y_pred_upper:TensorLike, alpha:float) -> float:
    """
    Compute the Average Coverage Error (ACE).

    Args:
        y_true (TensorLike): label true values.
        y_pred_lower (TensorLike): lower bounds of the prediction intervals.
        y_pred_upper (TensorLike): lower bounds of the prediction intervals.
        alpha (float): significance level (max miscoverage target).

    .. NOTE::

        The ACE is the distance between the nominal coverage :math:`1-\\alpha`
        and the empirical average coverage :math:`AC` such that
        :math:`ACE = AC - (1-\\alpha)`.

        If the ACE is strictly negative, the prediction intervals are
        marginally undercovering. If the ACE is strictly positive, the
        prediction intervals are maginally conservative.

    Returns:
        float:  the average coverage error (ACE).
    """
    cov = regression_mean_coverage(y_true, y_pred_lower, y_pred_upper)
    return cov - (1 - alpha)

def regression_sharpness(y_pred_lower:TensorLike, y_pred_upper:TensorLike) -> float:
    """
    Compute the average absolute width of the prediction intervals.

    Args:
        y_pred_lower (TensorLike): lower bounds of the prediction intervals.
        y_pred_upper (TensorLike): upper bounds of the prediction intervals.

    Returns:
        float: average absolute width of the prediction intervals.
    """
    return ops.mean(ops.abs(y_pred_upper - y_pred_lower))

def object_detection_mean_coverage(
    y_pred_outer: TensorLike, y_true: TensorLike
)->float:
    """
    Calculate the mean coverage of conformal object detection predictions.
    For each instance, coverage is defined as the true bounding box being inside
    the predicted outer bounding box.

    Args:
        y_pred_outer (TensorLike): array of predicted outer bounding boxes with shape (n, 4).
        y_true (TensorLike): array of true bounding boxes with shape (n, 4).

    Returns:
        float: average coverage, indicating the proportion of objects that are
        correctly covered.
    """
    x_min_true, y_min_true, x_max_true, y_max_true = ops.split(y_true, 4, axis=1)
    x_min, y_min, x_max, y_max = ops.split(y_pred_outer, 4, axis=1)
    cov = (
        (x_min <= x_min_true)
        * (y_min <= y_min_true)
        * (x_max >= x_max_true)
        * (y_max >= y_max_true)
    )
    return ops.mean(cov)


def object_detection_mean_area(y_pred: TensorLike)->float:
    """
    Calculate the mean area of object bounding predictions.

    Args:
        y_pred (TensorLike): array of predicted bounding boxes with shape (n, 4).

    Returns:
        float: average area of the bounding boxes
    """
    x_min, y_min, x_max, y_max = ops.split(y_pred, 4, axis=1)
    return ops.mean((x_max - x_min) * (y_max - y_min))


def iou(bboxes1: TensorLike, bboxes2: TensorLike) -> TensorLike:
    """
    Calculates the Intersection over Union (IoU) between two sets of 
    bounding boxes. The IoU is calculated as the ratio between the area of 
    intersection and the area of union between two bounding boxes.

    Args:
        bboxes1 (TensorLike): array of shape (N, 4) representing the 
        coordinates of N bounding boxes in the format 
        [x_min, y_min, x_max, y_max].

        bboxes2 (TensorLike): array of shape (N, 4) representing the 
        coordinates of N bounding boxes in the format 
        [x_min, y_min, x_max, y_max].

    Returns:
        TensorLike: Array of shape (N, ) representing the IoU 
        between each pair of bounding boxes.
    """
    x1_min, y1_min, x1_max, y1_max = ops.split(bboxes1, 4, axis=1)
    x2_min, y2_min, x2_max, y2_max = ops.split(bboxes2, 4, axis=1)

    inter_x_min = ops.maximum(x1_min, ops.transpose(x2_min))
    inter_y_min = ops.maximum(y1_min, ops.transpose(y2_min))
    inter_x_max = ops.minimum(x1_max, ops.transpose(x2_max))
    inter_y_max = ops.minimum(y1_max, ops.transpose(y2_max))

    inter_width = ops.maximum(inter_x_max - inter_x_min + 1, 0)
    inter_height = ops.maximum(inter_y_max - inter_y_min + 1, 0)
    inter_area = inter_width * inter_height

    box1_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    box2_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

    result = inter_area / (box1_area + ops.transpose(box2_area) - inter_area)
    return result

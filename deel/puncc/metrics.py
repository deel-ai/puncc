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
"""This module provides conformal prediction metrics."""

from typing import Tuple

import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Classification ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def classification_mean_coverage(
    y_true: np.ndarray, set_pred: Tuple[np.ndarray]
) -> float:
    """Compute empirical coverage of the prediction sets.

        Given the $i$-th prediction set $S(X_i)$, the coverage is:

        * $cov(X_i) = 1$ if $y_{true} \\in S(X_i)$
        * $cov(X_i) = 0$ otherwise

        With N the number of examples, the average coverage is
        $1/N \\sum_{i=1}^{N} cov(X_i)$.

    Args:
        y_true (np.ndarray): Observed label
        set_pred (Tuple[np.ndarray]): label prediction set

    Returns:
        float: average coverage, indicating the proportion of instances that are correctly covered.
    """
    counter = 0
    for y, S in zip(y_true, set_pred):
        if (S != []) and (y in S):
            counter += 1
    return counter / len(y_true)


def classification_mean_size(set_pred: Tuple[np.ndarray]) -> float:
    """Compute average size of the prediction sets.

    Args:
        set_pred (Tuple[np.ndarray]): label prediction set

    Returns:
        float: Average size of the prediction sets"""
    return np.mean([len(s) for s in set_pred])


def classification_classwise_coverage(
    y_true: np.ndarray, set_pred: Tuple[np.ndarray], n_classes: int
) -> np.ndarray:
    """Compute classwise coverage of the prediction sets.

        Given the $i$-th prediction set $S(X_i)$, the coverage is:

        * $cov(X_i) = 1$ if $y_{true} \\in S(X_i)$
        * $cov(X_i) = 0$ otherwise

        With N the number of examples, the classwise coverage for class k is
        $1/N_k \\sum_{i:y_i=k}^{N} cov(X_i)$, where $N_k$ is the number
        of examples of class k.

    Args:
        y_true (np.ndarray): Observed label
        set_pred (Tuple[np.ndarray]): label prediction set
        n_classes (int): number of classes

    Returns:
        np.ndarray: classwise coverage, indicating the proportion of instances that are correctly covered for each class.
    """
    coverage = np.zeros(n_classes)
    counts = np.zeros(n_classes)

    for y, S in zip(y_true, set_pred):
        counts[y] += 1
        if (S != []) and (y in S):
            coverage[y] += 1

    # Avoid division by zero
    for k in range(n_classes):
        if counts[k] > 0:
            coverage[k] /= counts[k]
        else:
            raise ValueError(f"No instances found for class {k}.")

    return coverage


def classification_classwise_size(
    y_true: np.ndarray, set_pred: Tuple[np.ndarray], n_classes: int
) -> np.ndarray:
    """Compute classwise average size of the prediction sets.

    Args:
        set_pred (Tuple[np.ndarray]): label prediction set
        n_classes (int): number of classes

    Returns:
        np.ndarray: classwise average size of the prediction sets."""
    size = np.zeros(n_classes)
    counts = np.zeros(n_classes)

    for y, S in zip(y_true, set_pred):
        counts[y] += 1
        size[y] += len(S)

    # Avoid division by zero
    for k in range(n_classes):
        if counts[k] > 0:
            size[k] /= counts[k]
        else:
            raise ValueError(f"No instances found for class {k}.")

    return size


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Regression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def regression_mean_coverage(y_true, y_pred_lower, y_pred_upper) -> float:
    """Compute average coverage on several prediction intervals.

        Given the $i$-th prediction interval $C(X_i)$, the coverage is:

            * $cov(X_i) = 1$ if $y_{true} \\in C(X_i)$
            * $cov(X_i) = 0$ otherwise

        With N the number of examples, the average coverage is
        $1/N \\sum_{i=1}^{N} cov(X_i)$.

    Args:
        y_true (ndarray): label true values.
        y_pred_lower (ndarray): lower bounds of the prediction intervals.
        y_pred_upper (ndarray): upper bounds of the prediction intervals.

    Returns:
        float: average coverage, indicating the proportion of instances that are correctly covered.
    """
    return ((y_true >= y_pred_lower) & (y_true <= y_pred_upper)).mean()


def regression_ace(y_true, y_pred_lower, y_pred_upper, alpha) -> float:
    """Compte the Average Coverage Error (ACE).

    Args:
        y_true (ndarray): label true values.
        y_pred_lower (ndarray): lower bounds of the prediction intervals.
        y_pred_upper (ndarray): upper bounds of the prediction intervals.
        alpha (float): significance level (max miscoverage target).

        !!! note
            The ACE is the distance between the nominal coverage $1-\\alpha$
            and the empirical average coverage $AC$ such that
            $ACE = AC - (1-\\alpha)$.

            If the ACE is strictly negative, the prediction intervals are
            marginally undercovering. If the ACE is strictly positive, the
            prediction intervals are maginally conservative.

    Returns:
        float: the average coverage error (ACE)."""
    cov = regression_mean_coverage(y_true, y_pred_lower, y_pred_upper)
    return cov - (1 - alpha)


def regression_sharpness(y_pred_lower, y_pred_upper) -> float:
    """Compute the average absolute width of the prediction intervals.

    Args:
        y_pred_lower (ndarray): lower bounds of the prediction intervals.
        y_pred_upper (ndarray): upper bounds of the prediction intervals.

    Returns:
        float: average absolute width of the prediction intervals."""
    return (np.abs(y_pred_upper - y_pred_lower)).mean()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Object Detection ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def object_detection_mean_coverage(y_pred_outer: np.ndarray, y_true: np.ndarray):
    """Calculate the mean coverage of conformal object detection predictions.
        For each instance, coverage is defined as the true bounding box being inside
        the predicted outer bounding box.

    Args:
        y_pred (np.ndarray): array of predicted outer bounding boxes with shape (n, 4).
        y_true (np.ndarray): array of true bounding boxes with shape (n, 4).

    Returns:
        float: average coverage, indicating the proportion of objects that are correctly covered.
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
    """Calculate the mean area of object bounding predictions.

    Args:
        y_pred (np.ndarray): array of predicted bounding boxes with shape (n, 4).

    Returns:
        float: average area of the bounding boxes"""
    x_min, y_min, x_max, y_max = np.hsplit(y_pred, 4)
    return np.mean((x_max - x_min) * (y_max - y_min))


# Calculate Intersection over Union (IOU) between two bounding boxes
def iou(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Calculates the Intersection over Union (IoU) between two sets of
        bounding boxes. The IoU is calculated as the ratio between the area of
        intersection and the area of union between two bounding boxes.

    Args:
        bboxes1 (np.ndarray): array of shape (N, 4) representing the  coordinates of N bounding boxes in the format [x_min, y_min, x_max, y_max].
        bboxes2 (np.ndarray): array of shape (N, 4) representing the  coordinates of N bounding boxes in the format [x_min, y_min, x_max, y_max].

    Returns:
        np.ndarray: iou (numpy.ndarray): Array of shape (N, ) representing the IoU  between each pair of bounding boxes.
    """

    x1_min, y1_min, x1_max, y1_max = np.split(bboxes1, 4, axis=1)
    x2_min, y2_min, x2_max, y2_max = np.split(bboxes2, 4, axis=1)

    inter_x_min = np.maximum(x1_min, np.transpose(x2_min))
    inter_y_min = np.maximum(y1_min, np.transpose(y2_min))
    inter_x_max = np.minimum(x1_max, np.transpose(x2_max))
    inter_y_max = np.minimum(y1_max, np.transpose(y2_max))

    inter_width = np.maximum(inter_x_max - inter_x_min + 1, 0)
    inter_height = np.maximum(inter_y_max - inter_y_min + 1, 0)
    inter_area = inter_width * inter_height

    box1_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    box2_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

    result = inter_area / (box1_area + np.transpose(box2_area) - inter_area)
    return result

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

"""Metrics for object detection."""

from __future__ import annotations
from collections.abc import Iterable

from deel.puncc.backend import ops
from deel.puncc.od.base import ODPrediction, ODTarget
from deel.puncc.od.matching import (
    AssignmentResult,
    check_assignment,
)
from deel.puncc.typing import TensorLike


def _nan() -> TensorLike:
    return ops.array(float("nan"), dtype="float32")


def detection_recall(
    y_pred: ODPrediction,
    y_true: ODTarget,
    assignment: AssignmentResult,
) -> TensorLike:
    """Compute the fraction of target objects assigned to a prediction.

    Args:
        y_pred: Object detection predictions.
        y_true: Object detection targets.
        assignment: Target-to-prediction assignment.

    Returns:
        Detection recall. Returns NaN if there are no target objects.
    """
    if len(y_true) == 0:
        return _nan()

    assignment = check_assignment(assignment)

    return ops.array(
        len(set(assignment.assigned_true_indices)) / len(y_true),
        dtype="float32",
    )


def detection_precision(
    y_pred: ODPrediction,
    y_true: ODTarget,
    assignment: AssignmentResult,
) -> TensorLike:
    """Compute the fraction of predictions assigned to a target object.

    Args:
        y_pred: Object detection predictions.
        y_true: Object detection targets.
        assignment: Prediction-to-target assignment.

    Returns:
        Detection precision. Returns NaN if there are no predictions.
    """
    if len(y_pred) == 0:
        return _nan()

    assignment = check_assignment(assignment)

    return ops.array(
        len(set(assignment.assigned_pred_indices)) / len(y_pred),
        dtype="float32",
    )


def mean_iou(
    y_pred: ODPrediction,
    y_true: ODTarget,
    assignment: AssignmentResult,
) -> TensorLike:
    """Compute the mean IoU of matched prediction-target pairs.

    Unassigned objects are ignored.

    Args:
        y_pred: Object detection predictions.
        y_true: Object detection targets.
        assignment: Target-to-prediction assignment.

    Returns:
        Mean IoU over matched pairs. Returns NaN if no pair is matched.
    """
    assignment = check_assignment(assignment)

    if not assignment.pairs:
        return _nan()

    ious = y_pred.pairwise_iou(y_true)

    return ops.mean(
        ops.stack(
            [
                ops.take(
                    ops.take(ious, pred_idx, axis=0),
                    true_idx,
                    axis=0,
                )
                for true_idx, pred_idx in assignment.pairs
            ]
        )
    )


def _mean_gt_coverage(
    n_true: int,
    assignment: AssignmentResult,
    pair_coverages: Iterable[TensorLike],
) -> TensorLike:
    """Compute mean GT coverage, keeping the best matched prediction.

    Unmatched ground truths contribute zero coverage.
    """
    if n_true == 0:
        return _nan()

    coverages = [ops.array(0.0, dtype="float32") for _ in range(n_true)]

    for (true_idx, _), coverage in zip(assignment.pairs, pair_coverages, strict=True):
        coverages[true_idx] = ops.maximum(
            coverages[true_idx],
            ops.cast(coverage, "float32"),
        )

    return ops.mean(ops.stack(coverages))


def box_coverage(
    y_pred: ODPrediction,
    y_true: ODTarget,
    assignment: AssignmentResult,
) -> TensorLike:
    """Compute the fraction of target boxes fully covered by their prediction.

    Unassigned target objects count as uncovered.

    Args:
        y_pred: Object detection predictions.
        y_true: Object detection targets.
        assignment: Target-to-prediction assignment.

    Returns:
        Fraction of fully covered target boxes. Returns NaN if there are no
        target objects.
    """
    if len(y_true) == 0:
        return _nan()

    assignment = check_assignment(assignment)

    return _mean_gt_coverage(
        len(y_true),
        assignment,
        (
            y_pred[pred_idx].contains(y_true[true_idx])
            for true_idx, pred_idx in assignment.pairs
        ),
    )


def pixel_coverage(
    y_pred: ODPrediction,
    y_true: ODTarget,
    assignment: AssignmentResult,
) -> TensorLike:
    """Compute the mean fraction of target area covered by assigned predictions.

    Coverage corresponds to intersection over target area. Unassigned target
    objects therefore contribute zero coverage.

    Args:
        y_pred: Object detection predictions.
        y_true: Object detection targets.
        assignment: Target-to-prediction assignment.

    Returns:
        Mean target-area coverage. Returns NaN if there are no target objects.
    """
    assignment = check_assignment(assignment)

    if len(y_true) == 0:
        return _nan()

    ioas = y_pred.pairwise_ioa(y_true)

    return _mean_gt_coverage(
        len(y_true),
        assignment,
        (
            ops.take(
                ops.take(ioas, pred_idx, axis=0),
                true_idx,
                axis=0,
            )
            for true_idx, pred_idx in assignment.pairs
        ),
    )


def classification_coverage(
    y_pred: ODPrediction,
    y_true: ODTarget,
    assignment: AssignmentResult,
) -> TensorLike:
    """Compute coverage of target labels by prediction sets.

    Unassigned target objects count as uncovered.

    Args:
        y_pred: Object detection predictions.
        y_true: Object detection targets.
        assignment: Target-to-prediction assignment.

    Returns:
        Classification coverage. Returns NaN if there are no target objects.
    """
    assignment = check_assignment(assignment)
    prediction_sets = y_pred.prediction_sets

    return _mean_gt_coverage(
        len(y_true),
        assignment,
        (
            ops.any(
                prediction_sets[pred_idx] == ops.take(y_true.labels, true_idx, axis=0)
            )
            for true_idx, pred_idx in assignment.pairs
        ),
    )


def mean_prediction_set_size(
    y_pred: ODPrediction,
) -> TensorLike:
    """Compute the mean number of classes in prediction sets.

    Args:
        y_pred: Object detection predictions.

    Returns:
        Mean prediction-set size. Returns NaN if there are no predictions.
    """
    if len(y_pred) == 0:
        return _nan()
    return ops.mean(
        ops.stack(
            [
                ops.cast(ops.size(prediction_set), "float32")
                for prediction_set in y_pred.prediction_sets
            ]
        )
    )


def joint_coverage(
    y_pred: ODPrediction,
    y_true: ODTarget,
    assignment: AssignmentResult,
) -> TensorLike:
    """Compute joint localization and classification coverage.

    A target object is covered when it is assigned to a prediction, its box is
    entirely contained in the predicted box, and its label belongs to the
    associated prediction set.

    Unassigned target objects count as uncovered.

    Args:
        y_pred: Object detection predictions.
        y_true: Object detection targets.
        assignment: Target-to-prediction assignment.

    Returns:
        Joint coverage. Returns NaN if there are no target objects.
    """
    assignment = check_assignment(assignment)
    prediction_sets = y_pred.prediction_sets

    return _mean_gt_coverage(
        len(y_true),
        assignment,
        (
            ops.logical_and(
                y_pred[pred_idx].contains(y_true[true_idx]),
                ops.any(
                    prediction_sets[pred_idx]
                    == ops.take(y_true.labels, true_idx, axis=0)
                ),
            )
            for true_idx, pred_idx in assignment.pairs
        ),
    )


def mean_box_area(
    y_pred: ODPrediction,
) -> TensorLike:
    """Compute the mean area of predicted boxes.

    Args:
        y_pred: Object detection predictions.

    Returns:
        Mean predicted box area. Returns NaN if there are no predictions.
    """
    if len(y_pred) == 0:
        return _nan()

    return ops.mean(y_pred.area)

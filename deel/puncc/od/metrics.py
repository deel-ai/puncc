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

from deel.puncc.backend import ops
from deel.puncc.od.base import ODPrediction, ODTarget
from deel.puncc.od.matching import (
    AssignmentResult,
    MatchingDirection,
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

    assignment = check_assignment(
        assignment,
        direction=MatchingDirection.TRUE_TO_PRED,
    )

    return ops.array(
        len(assignment.matched_pairs) / len(y_true),
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

    assignment = check_assignment(
        assignment,
        direction=MatchingDirection.PRED_TO_TRUE,
    )

    return ops.array(
        len(assignment.matched_pairs) / len(y_pred),
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
    assignment = check_assignment(
        assignment,
        direction=MatchingDirection.TRUE_TO_PRED,
    )

    if not assignment.matched_pairs:
        return _nan()

    ious = y_pred.pairwise_iou(y_true)

    return ops.mean(
        ops.stack(
            [
                ious[pred_idx, true_idx]
                for true_idx, pred_idx in assignment.matched_pairs
            ]
        )
    )


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

    assignment = check_assignment(
        assignment,
        direction=MatchingDirection.TRUE_TO_PRED,
    )

    covered = ops.array(0.0)

    for true_idx, pred_idx in assignment.matched_pairs:
        covered += ops.cast(
            y_pred[pred_idx].contains(y_true[true_idx]),
            "float32",
        )

    return covered / len(y_true)


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
    if len(y_true) == 0:
        return _nan()

    assignment = check_assignment(
        assignment,
        direction=MatchingDirection.TRUE_TO_PRED,
    )

    ioas = y_pred.pairwise_ioa(y_true)
    coverage = ops.array(0.0)

    for true_idx, pred_idx in assignment.matched_pairs:
        coverage += ioas[pred_idx, true_idx]

    return coverage / len(y_true)


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
    if len(y_true) == 0:
        return _nan()

    assignment = check_assignment(
        assignment,
        direction=MatchingDirection.TRUE_TO_PRED,
    )

    prediction_sets = y_pred.prediction_sets
    covered = ops.array(0.0)

    for true_idx, pred_idx in assignment.matched_pairs:
        covered += ops.cast(
            ops.any(
                prediction_sets[pred_idx]
                == y_true.labels[true_idx]
            ),
            "float32",
        )

    return covered / len(y_true)


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
        ops.stack([ops.cast(ops.size(prediction_set),"float32") 
                   for prediction_set in y_pred.prediction_sets])
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
    if len(y_true) == 0:
        return _nan()

    assignment = check_assignment(
        assignment,
        direction=MatchingDirection.TRUE_TO_PRED,
    )

    prediction_sets = y_pred.prediction_sets
    covered = ops.array(0.0)

    for true_idx, pred_idx in assignment.matched_pairs:
        localization_covered = y_pred[pred_idx].contains(
            y_true[true_idx]
        )

        classification_covered = ops.any(
            prediction_sets[pred_idx]
            == y_true.labels[true_idx]
        )

        covered += ops.cast(
            ops.logical_and(
                localization_covered,
                classification_covered,
            ),
            "float32",
        )

    return covered / len(y_true)


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

    return ops.mean(y_pred.areas)
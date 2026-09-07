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
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import singledispatchmethod

from deel.puncc.backend.keras import ops
from deel.puncc.typing import TensorLike
from deel.puncc.od.base import ODPrediction, ODTarget
from deel.puncc.od.matching import (
    AssignmentResult,
    AsymmetricHausdorffDistance,
    DistanceMetric,
    MatchingDirection,
)


def check_assignment(
    assignment: AssignmentResult | None,
    *,
    direction: MatchingDirection | None = None,
) -> AssignmentResult:
    if assignment is None:
        raise ValueError(
            "This loss requires an AssignmentResult."
        )

    if direction is not None and assignment.matching_direction != direction:
        raise ValueError(
            f"This loss requires a {direction.value} assignment, "
            f"got {assignment.matching_direction.value}."
        )

    return assignment


class ODLoss(ABC):
    upper_bound: float = 1.0

    @singledispatchmethod
    def __call__(
        self,
        y_pred:ODPrediction|Sequence[ODPrediction],
        y_true:ODTarget|Sequence[ODTarget],
        assignment:AssignmentResult|None=None,
    ) -> TensorLike:
        raise TypeError(
            f"Unsupported prediction type: {type(y_pred).__name__}"
        )

    @__call__.register
    def _(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> TensorLike:
        return self.compute(
            y_pred,
            y_true,
            assignment,
        )

    @__call__.register
    def _(
        self,
        y_pred: Sequence[ODPrediction],
        y_true: Sequence[ODTarget],
        assignment: Sequence[AssignmentResult | None] | None = None,
    ) -> TensorLike:
        if len(y_pred) != len(y_true):
            raise ValueError(
                "y_pred and y_true must have the same length."
            )

        assignments = (
            [None] * len(y_pred)
            if assignment is None
            else assignment
        )

        return ops.stack(
            [
                self.compute(
                    pred,
                    target,
                    assign,
                )
                for pred, target, assign in zip(
                    y_pred,
                    y_true,
                    assignments,
                )
            ]
        )

    @abstractmethod
    def compute(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> TensorLike:
        ...


class ConfidenceLoss(ODLoss):
    pass


class LocalizationLoss(ODLoss):
    pass


class ClassificationLoss(ODLoss):
    pass


class BoxCountThresholdLoss(ConfidenceLoss):
    """
    Binary loss indicating whether fewer boxes are predicted than expected.
    """

    def compute(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> TensorLike:
        return ops.array(
            float(len(y_pred) < len(y_true))
        )


class BoxCountTwoSidedLoss(ConfidenceLoss):
    """
    Binary loss based on the difference between the number of predicted
    and target boxes.
    """

    def __init__(
        self,
        threshold: int = 3,
    ):
        self.threshold = threshold

    def compute(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> TensorLike:
        if len(y_true) == 0:
            return ops.array(0.0)

        return ops.array(
            float(
                abs(len(y_true) - len(y_pred))
                > self.threshold
            )
        )


class BoxCountRecallLoss(ConfidenceLoss):
    """
    Count-based approximation of detection recall.
    """

    def compute(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> TensorLike:
        if len(y_true) == 0:
            return ops.array(0.0)

        return ops.array(
            max(
                0.0,
                (len(y_true) - len(y_pred))
                / len(y_true),
            )
        )


class DetectionRecallLoss(ConfidenceLoss):
    """
    Fraction of target objects left unassigned.
    """

    def compute(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> TensorLike:
        if len(y_true) == 0:
            return ops.array(0.0)

        assignment = check_assignment(
            assignment,
            direction=MatchingDirection.TRUE_TO_PRED,
        )

        return ops.array(
            len(assignment.unassigned_source_indices)
            / len(y_true)
        )


class ThresholdedDistanceConfidenceLoss(ConfidenceLoss):
    """
    Fraction of target objects whose closest prediction is farther
    than a given distance threshold.
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        distance_metric: DistanceMetric = AsymmetricHausdorffDistance(),
    ):
        self.distance_threshold = distance_threshold
        self.distance_metric = distance_metric

    def compute(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> TensorLike:
        if len(y_true) == 0:
            return ops.array(0.0)

        if len(y_pred) == 0:
            return ops.array(1.0)

        distances = self.distance_metric.cost_matrix(
            y_pred,
            y_true,
        )

        shortest_distances = ops.min(
            distances,
            axis=1,
        )

        return ops.mean(
            ops.cast(
                shortest_distances > self.distance_threshold,
                "float32",
            )
        )


class ClassificationCoverageLoss(ClassificationLoss):
    """
    Fraction of target objects whose class is not covered by the
    corresponding prediction set.

    Unassigned target objects count as errors.
    """

    def compute(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> TensorLike:
        if len(y_true) == 0:
            return ops.array(0.0)

        assignment = check_assignment(
            assignment,
            direction=MatchingDirection.TRUE_TO_PRED,
        )

        loss = ops.array(
            float(len(assignment.unassigned_source_indices))
        )

        prediction_sets = y_pred.prediction_sets

        for true_idx, pred_idx in assignment.matched_pairs:
            loss += ops.cast(
                ops.logical_not(
                    ops.any(
                        prediction_sets[pred_idx]
                        == y_true.labels[true_idx]
                    )
                ),
                "float32",
            )

        return loss / len(y_true)



class BoxCoverageLoss(LocalizationLoss):
    """
    Fraction of target boxes not entirely covered by their assigned
    predicted box.

    Unassigned target objects count as errors.
    """

    def compute(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> TensorLike:
        if len(y_true) == 0:
            return ops.array(0.0)

        assignment = check_assignment(
            assignment,
            direction=MatchingDirection.TRUE_TO_PRED,
        )

        loss = ops.array(
            float(len(assignment.unassigned_source_indices))
        )

        for true_idx, pred_idx in assignment.matched_pairs:
            loss += ops.cast(
                ops.logical_not(
                    y_pred[pred_idx].contains(
                        y_true[true_idx]
                    )
                ),
                "float32",
            )

        return loss / len(y_true)


class PixelCoverageLoss(LocalizationLoss):
    """
    Mean fraction of target area not covered by the assigned
    predicted boxes.

    Unassigned target objects have zero covered area.
    """

    def compute(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> TensorLike:
        if len(y_true) == 0:
            return ops.array(0.0)

        assignment = check_assignment(
            assignment,
            direction=MatchingDirection.TRUE_TO_PRED,
        )

        covered_area = ops.array(0.0)

        for true_idx, pred_idx in assignment.matched_pairs:
            true_box = y_true[true_idx]
            pred_box = y_pred[pred_idx]

            covered_area += (
                true_box.intersection(pred_box).area
                / true_box.area
            )

        return (
            ops.array(1.0)
            - covered_area / len(y_true)
        )


class ThresholdedRecallLoss(LocalizationLoss):
    """
    Binary loss indicating whether a localization loss exceeds beta.
    """

    def __init__(
        self,
        beta: float = 0.25,
        base_loss: LocalizationLoss | None = None,
    ):
        self.beta = beta
        self.base_loss = (
            BoxCoverageLoss()
            if base_loss is None
            else base_loss
        )

    def compute(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> TensorLike:
        loss = self.base_loss.compute(
            y_pred,
            y_true,
            assignment,
        )

        return ops.cast(
            loss > self.beta,
            "float32",
        )


class BoxPrecisionLoss(LocalizationLoss):
    """
    Fraction of predicted boxes not entirely contained in their
    assigned target box.

    Unassigned predictions count as errors.
    """

    def compute(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> TensorLike:
        if len(y_pred) == 0:
            return ops.array(0.0)

        assignment = check_assignment(
            assignment,
            direction=MatchingDirection.PRED_TO_TRUE,
        )

        loss = ops.array(
            float(len(assignment.unassigned_source_indices))
        )

        for pred_idx, true_idx in assignment.matched_pairs:
            loss += ops.cast(
                ops.logical_not(
                    y_true[true_idx].contains(
                        y_pred[pred_idx]
                    )
                ),
                "float32",
            )

        return loss / len(y_pred)


class IoUThresholdLoss(LocalizationLoss):
    """
    Fraction of target boxes whose assigned prediction has an IoU
    below the given threshold.

    Unassigned targets count as errors.
    """

    def __init__(
        self,
        iou_threshold: float = 0.9,
    ):
        self.iou_threshold = iou_threshold

    def compute(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> TensorLike:
        if len(y_true) == 0:
            return ops.array(0.0)

        assignment = check_assignment(
            assignment,
            direction=MatchingDirection.TRUE_TO_PRED,
        )

        loss = ops.array(
            float(len(assignment.unassigned_source_indices))
        )

        for true_idx, pred_idx in assignment.matched_pairs:
            loss += ops.cast(
                y_true[true_idx].iou(
                    y_pred[pred_idx]
                ) < self.iou_threshold,
                "float32",
            )

        return loss / len(y_true)

class JointCoverageLoss(ODLoss):
    """
    Joint localization and classification miscoverage.

    A target object is covered if:
      - it has an assigned prediction,
      - the predicted box contains the target box,
      - the true class belongs to the prediction set.
    """
    def compute(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> TensorLike:
        if len(y_true) == 0:
            return ops.array(0.0)

        assignment = check_assignment(
            assignment,
            direction=MatchingDirection.TRUE_TO_PRED,
        )

        loss = ops.array(
            float(len(assignment.unassigned_source_indices))
        )

        prediction_sets = y_pred.prediction_sets

        for true_idx, pred_idx in assignment.matched_pairs:
            loc_covered = y_pred[pred_idx].contains(
                y_true[true_idx]
            )

            cls_covered = ops.any(
                prediction_sets[pred_idx]
                == y_true.labels[true_idx]
            )

            loss += ops.cast(
                ops.logical_not(
                    ops.logical_and(
                        loc_covered,
                        cls_covered,
                    )
                ),
                "float32",
            )

        return loss / len(y_true)
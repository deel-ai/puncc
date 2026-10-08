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

from abc import ABC
from collections.abc import Sequence

from deel.puncc.backend.keras import ops
from deel.puncc.typing import TensorLike
from deel.puncc.od.base import ODPrediction, ODTarget
from deel.puncc.od.matching import (
    AssignmentResult,
    AsymmetricHausdorffDistance,
    DistanceMetric,
    MatchingDirection,
    check_assignment
)

class ODLoss(ABC):
    """
    Base class for object-detection losses.

    An OD loss can operate at two granularities:
    - image-wise:
        one scalar loss is returned for each image;
    - box-wise:
        one scalar loss is returned for each statistical box unit.

    For assignment-based box-wise losses, unmatched source boxes may either
    be ignored or penalized with ``upper_bound``.

    Args:
        boxwise:
            Whether the loss should return one value per box instead of one
            value per image.

        penalize_unmatched_boxes:
            Whether unmatched boxes belonging to the population evaluated by the loss are penalized.
    """
    upper_bound: float = 1.0
    matching_direction: MatchingDirection | None = None

    def __init__(
        self,
        *,
        boxwise: bool = False,# if false : imagewise
        penalize_unmatched_boxes: bool = True, 
    ):
        self.boxwise = boxwise
        self.penalize_unmatched_boxes = penalize_unmatched_boxes

    def __call__(
        self,
        y_pred: Sequence[ODPrediction],
        y_true: Sequence[ODTarget],
        assignments: Sequence[AssignmentResult | None] | None = None,
    ) -> TensorLike:
        assignments = [None] * len(y_pred) if assignments is None else assignments

        if self.boxwise:
            losses: list[TensorLike] = []
            for prediction, target, assignment in zip(y_pred, y_true, assignments, strict=True):
                losses.extend(self.compute_boxwise(prediction, target, assignment))
        else:
            losses = [self.compute_imagewise(prediction, target, assignment)
                for prediction, target, assignment in zip(y_pred, y_true, assignments, strict=True)]
        if not losses:
            return ops.zeros((0,), dtype="float32")
        return ops.stack(losses)

    def compute_imagewise(self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None) -> TensorLike:
        """
        Compute the image-wise loss.

        For box-decomposable losses, the default implementation simply
        averages the elementary box-wise losses.
        """
        losses = self.compute_boxwise(y_pred, y_true, assignment)
        if not losses:
            return ops.array(0.0)
        return ops.mean(ops.stack(losses))

    def compute_boxwise(self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None) -> list[TensorLike]:
        """
        Compute elementary box-wise losses for one image.

        Losses that are intrinsically image-wise should override
        ``compute_imagewise`` only.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support box-wise computation.")
    
    def matched_pairs(self, assignment: AssignmentResult) -> zip[tuple[int, int]]:
        return zip(*assignment.matched_indices(), strict=True)

    def _add_unmatched_penalties(self,
        losses: list[TensorLike],
        unmatched_indices: Sequence[int]) -> list[TensorLike]:
        if not self.penalize_unmatched_boxes:
            return losses

        losses.extend(ops.array(self.upper_bound) for _ in unmatched_indices)
        return losses

class ConfidenceLoss(ODLoss):
    ...

class LocalizationLoss(ODLoss):
    ...

class ClassificationLoss(ODLoss):
    ...

class BoxCountThresholdLoss(ConfidenceLoss):
    def __init__(self):
        super().__init__(boxwise=False, penalize_unmatched_boxes=False)

    def compute_imagewise(self, y_pred, y_true, assignment=None):
        return ops.array(float(len(y_pred) < len(y_true)))

class BoxCountRecallLoss(ConfidenceLoss):
    """
    Count-based approximation of detection recall.
    """
    def __init__(self):
        super().__init__(boxwise=False, penalize_unmatched_boxes=False)

    def compute_imagewise(self, y_pred, y_true, assignment=None)->TensorLike:
        if len(y_true) == 0:
            return ops.array(0.0)
        return ops.array(max(0.0, (len(y_true) - len(y_pred)) / len(y_true)))

class BoxCountTwoSidedLoss(ConfidenceLoss):
    """
    Binary loss based on the difference between the number of predicted
    and target boxes.
    """
    def __init__(self, threshold: int = 3):
        super().__init__(boxwise=False, penalize_unmatched_boxes=False)
        self.threshold = threshold

    def compute_imagewise(self, y_pred: ODPrediction, y_true: ODTarget, assignment: AssignmentResult | None = None) -> TensorLike:
        if len(y_true) == 0:
            return ops.array(0.0)
        return ops.array(float(abs(len(y_true) - len(y_pred)) > self.threshold))

class DetectionRecallLoss(ConfidenceLoss):
    """
    Loss 1 for an unmatched GT object, 0 for a matched GT object.
    """
    def __init__(self, *, boxwise=False):
        super().__init__(boxwise=boxwise, penalize_unmatched_boxes=True)

    def compute_boxwise(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> list[TensorLike]:
        if len(y_true) == 0:
            return []        
        assignment = check_assignment(assignment=assignment)
        unmatched = set(assignment.unmatched_true_indices())
        return [ops.array(self.upper_bound if i in unmatched else 0.0) for i in range(len(y_true))]

class ThresholdedDistanceConfidenceLoss(ConfidenceLoss):
    """
    Fraction of target objects whose closest prediction is farther than ``distance_threshold``.
    """

    def __init__(self,
        distance_threshold: float = 0.5,
        distance_metric: DistanceMetric | None = None,
        *,
        boxwise: bool = False):
        super().__init__(
            boxwise=boxwise,
            penalize_unmatched_boxes=True,
        )
        self.distance_threshold = distance_threshold
        self.distance_metric = AsymmetricHausdorffDistance() if distance_metric is None else distance_metric

    def compute_boxwise(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> list[TensorLike]:
        if not len(y_true):
            return []
        if self.boxwise:
            check_assignment(assignment, direction=MatchingDirection.TRUE_TO_PRED)

        if not len(y_pred):
            return [ops.array(self.upper_bound) for _ in range(len(y_true))]

        distances = self.distance_metric.cost_matrix(y_pred, y_true)
        shortest_distances = ops.min(distances, axis=1)
        return [ops.cast(shortest_distances[i] > self.distance_threshold, "float32") for i in range(len(y_true))]

class ClassificationCoverageLoss(ClassificationLoss):
    """
    Fraction of target objects whose class is not covered by the corresponding prediction set.

    Unassigned target objects count as errors.
    """
    def compute_boxwise(self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None) -> list[TensorLike]:
        assignment = check_assignment(assignment=assignment)
        pairs = list(self.matched_pairs(assignment))

        if pairs and y_pred.class_sets is None:
            raise ValueError("ClassificationCoverageLoss requires prediction class_sets.")

        losses = [ops.cast(ops.logical_not(ops.any(y_pred.class_sets[pred_idx] == y_true.labels[true_idx])), "float32")
            for true_idx, pred_idx in pairs]
        return self._add_unmatched_penalties(losses, assignment.unmatched_true_indices())

class BoxCoverageLoss(LocalizationLoss):
    def compute_boxwise(self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None) -> list[TensorLike]:
        assignment = check_assignment(assignment=assignment)

        true_to_losses: dict[int, list[TensorLike]] = {}

        for true_idx, pred_idx in self.matched_pairs(assignment):
            loss = ops.cast(
                ops.logical_not(
                    y_pred[pred_idx].contains(y_true[true_idx])
                ),
                "float32",
            )
            true_to_losses.setdefault(true_idx, []).append(loss)

        losses = []

        for true_idx in range(len(y_true)):
            if true_idx in true_to_losses:
                losses.append(ops.min(ops.stack(true_to_losses[true_idx])))
            elif self.penalize_unmatched_boxes:
                losses.append(ops.array(self.upper_bound))

        return losses

class PixelCoverageLoss(LocalizationLoss):
    """
    Mean fraction of target area not covered by the assigned
    predicted boxes.

    Unassigned target objects have zero covered area.
    """
    def compute_boxwise(self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None) -> list[TensorLike]:
        assignment = check_assignment(assignment)

        losses = []
        for true_idx, pred_idx in self.matched_pairs(assignment):
            true_box = y_true[true_idx]
            pred_box = y_pred[pred_idx]
            covered_fraction = true_box.intersection(pred_box).area / ops.maximum(true_box.area, 1e-12)
            losses.append(ops.array(1.0) - covered_fraction)
        return self._add_unmatched_penalties(losses, assignment.unmatched_true_indices())

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
        self.base_loss = BoxCoverageLoss() if base_loss is None else base_loss
        super().__init__(boxwise=False, penalize_unmatched_boxes=self.base_loss.penalize_unmatched_boxes)

    def compute_imagewise(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> TensorLike:
        loss = self.base_loss.compute_imagewise(y_pred, y_true, assignment)
        return ops.cast(loss > self.beta, "float32")


class BoxPrecisionLoss(LocalizationLoss):
    """
    Fraction of predicted boxes not entirely contained in their
    assigned target box.

    Unassigned predictions count as errors.
    """
    def compute_boxwise(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> list[TensorLike]:
        assignment = check_assignment(assignment)
        losses = [ops.cast(ops.logical_not(y_true[true_idx].contains(y_pred[pred_idx])), "float32")
            for true_idx, pred_idx in self.matched_pairs(assignment)]
        return self._add_unmatched_penalties(losses, assignment.unmatched_pred_indices())

class IoUThresholdLoss(LocalizationLoss):
    """
    Fraction of target boxes whose assigned prediction has an IoU
    below the given threshold.

    Unassigned targets count as errors.
    """
    def __init__(
        self,
        iou_threshold: float = 0.9,
        *,
        boxwise: bool = False,
        penalize_unmatched_boxes: bool = True,
    ):
        super().__init__(boxwise=boxwise, penalize_unmatched_boxes=penalize_unmatched_boxes)
        self.iou_threshold = iou_threshold

    def compute_boxwise(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> list[TensorLike]:
        assignment = check_assignment(assignment)
        losses = [ops.cast(y_true[true_idx].iou(y_pred[pred_idx]) < self.iou_threshold, "float32")
            for true_idx, pred_idx in self.matched_pairs(assignment)]
        return self._add_unmatched_penalties(losses, assignment.unmatched_true_indices())

class JointCoverageLoss(ODLoss):
    """
    Joint localization and classification miscoverage.

    A target object is covered if:
      - it has an assigned prediction,
      - the predicted box contains the target box,
      - the true class belongs to the prediction set.
    """
    def compute_boxwise(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
        assignment: AssignmentResult | None = None,
    ) -> list[TensorLike]:
        assignment = check_assignment(assignment)
        pairs = list(self.matched_pairs(assignment))

        if pairs and y_pred.class_sets is None:
            raise ValueError("JointCoverageLoss requires prediction class_sets.")

        losses = []
        for true_idx, pred_idx in pairs:
            loc_covered = y_pred[pred_idx].contains(y_true[true_idx])
            cls_covered = ops.any(y_pred.class_sets[pred_idx] == y_true.labels[true_idx])
            losses.append(ops.cast(ops.logical_not(ops.logical_and(loc_covered, cls_covered)), "float32"))
        return self._add_unmatched_penalties(losses, assignment.unmatched_true_indices())
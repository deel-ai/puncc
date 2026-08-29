"""
Matching utilities for object detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.optimize import linear_sum_assignment

from deel.puncc import ops, random
from deel.puncc.typing import TensorLike

from .base import ODPrediction, ODTarget

class MatchingDirection(StrEnum):
    """
    Direction of the assignment
    """
    TRUE_TO_PRED = "true_to_pred"
    PRED_TO_TRUE = "pred_to_true"

@dataclass(slots=True)
class AssignmentResult:
    """
    Results of an assignment between a source set and a target set.
    """
    source_to_target_index: list[int | None]
    unassigned_target_indices: list[int]
    matching_direction:MatchingDirection = MatchingDirection.TRUE_TO_PRED

    def get(self, source_index: int) -> int | None:
        return self.source_to_target_index[source_index]

    @property
    def matched_pairs(self) -> list[tuple[int, int]]:
        return [
            (source_index, target_index)
            for source_index, target_index
            in enumerate(self.source_to_target_index)
            if target_index is not None
        ]

    @property
    def unassigned_source_indices(self) -> list[int]:
        return [
            source_index
            for source_index, target_index
            in enumerate(self.source_to_target_index)
            if target_index is None
        ]


@runtime_checkable
class AssignmentMethod(Protocol):
    """
    General interface for an assignment method between a set of predictions and a set of ground truths.
    """
    def assign(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
    ) -> AssignmentResult:
        ...

@runtime_checkable
class DistanceMetric(Protocol):
    """
    base interface for distance between true and predicted bounding boxes.
    """
    def cost_matrix(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
    ) -> TensorLike:
        ...


class IoUDistance:
    """1 - IoU."""

    def cost_matrix(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
    ) -> TensorLike:
        return 1.0 - y_true.pairwise_iou(y_pred)


class LACDistance:
    """
    Classification matching cost.
    1 - score[j, true_label[i]]
    """

    def cost_matrix(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
    ) -> TensorLike:
        # (n_pred, n_classes) -> (n_classes, n_pred)
        scores_by_class = ops.transpose(
            y_pred.softmaxs,
        )

        # Select, for each GT, the scores corresponding to its true class.
        # Result shape: (n_true, n_pred)
        true_class_scores = ops.take(
            scores_by_class,
            y_true.labels,
            axis=0,
        )

        return 1.0 - true_class_scores

class AsymmetricHausdorffDistance:
    """
    Asymmetric Hausdorff distance between ground-truth and
    predicted bounding boxes.

    For each pair (true_box, pred_box), the cost is:

        max(
            pred_x1 - true_x1,
            pred_y1 - true_y1,
            true_x2 - pred_x2,
            true_y2 - pred_y2,
        )

    The returned matrix has shape (n_true, n_pred).

    A cost <= 0 means that the predicted box contains the ground-truth box.
    """

    def cost_matrix(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
    ) -> TensorLike:
        true_boxes = y_true.boxes
        pred_boxes = y_pred.boxes

        true_x1 = true_boxes[:, 0][:, None]
        true_y1 = true_boxes[:, 1][:, None]
        true_x2 = true_boxes[:, 2][:, None]
        true_y2 = true_boxes[:, 3][:, None]

        pred_x1 = pred_boxes[:, 0][None, :]
        pred_y1 = pred_boxes[:, 1][None, :]
        pred_x2 = pred_boxes[:, 2][None, :]
        pred_y2 = pred_boxes[:, 3][None, :]

        distances = ops.stack(
            (
                pred_x1 - true_x1,
                pred_y1 - true_y1,
                true_x2 - pred_x2,
                true_y2 - pred_y2,
            ),
            axis=-1,
        )

        return ops.max(
            distances,
            axis=-1,
        )

class MixedDistance:
    def __init__(
        self,
        localization_distance: DistanceMetric=IoUDistance(),
        classification_distance: DistanceMetric=LACDistance(),
        class_weight: float = 0.25,
    ):
        self.localization_distance = localization_distance
        self.classification_distance = classification_distance
        self.class_weight = class_weight

    def cost_matrix(
        self,
        y_pred,
        y_true,
    ):
        loc = self.localization_distance.cost_matrix(
            y_pred,
            y_true,
        )

        cls = self.classification_distance.cost_matrix(
            y_pred,
            y_true,
        )

        return (
            (1 - self.class_weight) * loc
            + self.class_weight * cls
        )

@runtime_checkable
class CostMatrixMatcher(Protocol):
    def match(
        self,
        cost_matrix: TensorLike,
        valid_mask: TensorLike | None = None,
    ) -> list[tuple[int, int]]:
        ...

class ArgminMatcher:
    """
    Match each source independently to its lowest-cost valid target.

    Several sources may be assigned to the same target.
    """

    def match(
        self,
        cost_matrix: TensorLike,
        valid_mask: TensorLike | None = None,
    ) -> list[tuple[int, int]]:
        n_source, n_target = ops.shape(cost_matrix)

        if not n_source or not n_target:
            return []

        if valid_mask is None:
            indices = ops.convert_to_numpy(
                ops.argmin(cost_matrix, axis=1)
            )
            return list(enumerate(indices.tolist()))

        indices = ops.convert_to_numpy(
            ops.argmin(
                ops.where(
                    valid_mask,
                    cost_matrix,
                    float("inf"),
                ),
                axis=1,
            )
        )

        valid_sources = ops.convert_to_numpy(
            ops.any(valid_mask, axis=1)
        )

        return [
            (i, int(indices[i]))
            for i in range(n_source)
            if valid_sources[i]
        ]

class HungarianMatcher:
    """One-to-one assignment minimizing the global cost."""

    def match(
        self,
        cost_matrix: TensorLike,
        valid_mask: TensorLike | None = None,
    ) -> list[tuple[int, int]]:
        costs = ops.convert_to_numpy(cost_matrix)
        n_source, n_target = costs.shape

        if not n_source or not n_target:
            return []

        if valid_mask is None:
            sources, targets = linear_sum_assignment(costs)

        else:
            valid = ops.convert_to_numpy(valid_mask).astype(bool)

            if not valid.any():
                return []

            valid_costs = costs[valid]
            cost_span = valid_costs.max() - valid_costs.min()
            unmatched_cost = (
                valid_costs.max()
                + (n_source + 1) * max(cost_span, 1.0)
            )

            augmented = np.full(
                (n_source, n_target + n_source),
                unmatched_cost,
            )
            augmented[:, :n_target] = np.where(
                valid,
                costs,
                np.inf,
            )

            sources, targets = linear_sum_assignment(augmented)

            keep = targets < n_target
            sources = sources[keep]
            targets = targets[keep]

        return list(
            zip(
                sources.tolist(),
                targets.tolist(),
            )
        )

class AssignmentStrategy:
    """
    Generic assignment method based on a pairwise cost matrix.

    The strategy combines:

    - a DistanceMetric to construct a canonical (n_true, n_pred)
      cost matrix,
    - a CostMatrixMatcher to perform the assignment,
    - optional constraints defining which pairs are admissible.

    More exotic assignment algorithms do not need to use this class and
    may implement AssignmentMethod directly.

    Args:
        distance_metric:
            Metric used to build the pairwise cost matrix.

        matcher:
            Cost-matrix matching algorithm.

        iou_threshold:
            If provided, pairs whose IoU is below this threshold are
            considered invalid.

        class_matching:
            If True, a prediction can only be matched to a ground truth
            when its highest-scoring class equals the target label.
    """

    def __init__(
        self,
        distance_metric: DistanceMetric = IoUDistance(),
        matcher: CostMatrixMatcher = ArgminMatcher(),
        *,
        direction: MatchingDirection = MatchingDirection.TRUE_TO_PRED,
        iou_threshold: float | None = None,
        class_matching: bool = False,
    ):
        self.distance_metric = distance_metric
        self.matcher = matcher
        self.direction = MatchingDirection(direction)
        self.iou_threshold = iou_threshold
        self.class_matching = class_matching

    def _compute_valid_mask(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
    ) -> TensorLike | None:
        valid_mask = None

        if self.iou_threshold is not None:
            valid_mask = (
                y_true.pairwise_iou(y_pred)
                >= self.iou_threshold
            )

        if self.class_matching:
            predicted_labels = ops.argmax(
                y_pred.softmaxs,
                axis=1,
            )

            same_class = (
                y_true.labels[:, None]
                == predicted_labels[None, :]
            )

            valid_mask = (
                same_class
                if valid_mask is None
                else ops.logical_and(valid_mask, same_class)
            )

        return valid_mask

    def assign(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
    ) -> AssignmentResult:
        n_true = len(y_true)
        n_pred = len(y_pred)

        reverse = self.direction == MatchingDirection.PRED_TO_TRUE

        n_source, n_target = (
            (n_pred, n_true)
            if reverse
            else (n_true, n_pred)
        )

        if not n_source or not n_target:
            return AssignmentResult(
                source_to_target_index=[None] * n_source,
                unassigned_target_indices=list(range(n_target)),
                matching_direction=self.direction,
            )

        costs = self.distance_metric.cost_matrix(
            y_pred,
            y_true,
        )
        valid_mask = self._compute_valid_mask(
            y_pred,
            y_true,
        )

        if reverse:
            costs = ops.transpose(costs)

            if valid_mask is not None:
                valid_mask = ops.transpose(valid_mask)

        matched_pairs = self.matcher.match(
            costs,
            valid_mask,
        )

        source_to_target = [None] * n_source

        for source, target in matched_pairs:
            source_to_target[source] = target

        assigned_targets = {
            target
            for _, target in matched_pairs
        }

        return AssignmentResult(
            source_to_target_index=source_to_target,
            unassigned_target_indices=[
                target
                for target in range(n_target)
                if target not in assigned_targets
            ],
            matching_direction=self.direction,
        )
    

class RandomMatcher:
    def __init__(
        self,
        injective: bool = False,
        seed: int | None = None,
    ):
        self.injective = injective
        self.seed = seed

    def match(
        self,
        cost_matrix: TensorLike,
        valid_mask: TensorLike | None = None,
    ) -> list[tuple[int, int]]:
        random_costs = random.uniform(
            ops.shape(cost_matrix),
            seed=self.seed,
        )

        if self.injective:
            return HungarianMatcher().match(
                random_costs,
                valid_mask,
            )

        return ArgminMatcher().match(
            random_costs,
            valid_mask,
        )
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
Matching utilities for object detection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.optimize import linear_sum_assignment

from deel.puncc import ops, random
from deel.puncc.typing import TensorLike

from deel.puncc.od.base import ODPrediction, ODTarget


class MatchingDirection(StrEnum):
    """
    Direction of the assignment

    TRUE_TO_PRED
    source = ground truths
    target = predictions

    PRED_TO_TRUE
    source = predictions
    target = ground truths
    """

    TRUE_TO_PRED = "true_to_pred"
    PRED_TO_TRUE = "pred_to_true"


@dataclass(slots=True)
class AssignmentResult:
    """
    Results of an assignment between a source set and a target set.
    pairs are always stored in the order (true_idx, pred_idx) regardless of the matching direction.
    """

    pairs: list[tuple[int, int]]  # (true_idx, pred_idx)
    n_true: int
    n_pred: int
    matching_direction: MatchingDirection = MatchingDirection.TRUE_TO_PRED

    def __post_init__(self) -> None:
        if self.n_true < 0:
            raise ValueError(f"n_true must be non-negative, got {self.n_true}")
        if self.n_pred < 0:
            raise ValueError(f"n_pred must be non-negative, got {self.n_pred}")

        sources = set()
        for t, p in self.pairs:
            if t < 0 or t >= self.n_true:
                raise IndexError(
                    f"True index {t} out of bounds for n_true={self.n_true}"
                )
            if p < 0 or p >= self.n_pred:
                raise IndexError(
                    f"Pred index {p} out of bounds for n_pred={self.n_pred}"
                )
            source_idx = (
                t if self.matching_direction == MatchingDirection.TRUE_TO_PRED else p
            )
            if source_idx in sources:
                raise ValueError(
                    f"Source index {source_idx} is assigned to multiple targets."
                )
            sources.add(source_idx)

    @property
    def assigned_true_indices(self) -> list[int]:
        """
        Returns:
            list[int]: list of indices of groundtruths that were assigned to at least one prediction.
        """
        return sorted({true_idx for true_idx, _ in self.pairs})

    @property
    def assigned_pred_indices(self) -> list[int]:
        """
        Returns:
            list[int]: list of indices of predictions that were assigned to at least one ground-truth object.
        """
        return sorted({pred_idx for _, pred_idx in self.pairs})

    @property
    def unassigned_true_indices(self) -> list[int]:
        """
        Returns:
            list[int]: list of indices of groundtruths that were not assigned to any prediction.
        """
        return [i for i in range(self.n_true) if i not in self.assigned_true_indices]

    @property
    def unassigned_pred_indices(self) -> list[int]:
        """
        Returns:
            list[int]: list of indices of predictions that were not assigned to any ground-truth object.
        """
        return [i for i in range(self.n_pred) if i not in self.assigned_pred_indices]

    @property
    def is_injective(self) -> bool:
        """Whether each candidate is matched to at most one source."""
        if self.matching_direction == MatchingDirection.TRUE_TO_PRED:
            return len(self.assigned_pred_indices) == len(self.pairs)
        return len(self.assigned_true_indices) == len(self.pairs)

    @property
    def is_surjective(self) -> bool:
        """Whether every candidate has at least one assigned source."""
        if self.matching_direction == MatchingDirection.TRUE_TO_PRED:
            return len(self.assigned_pred_indices) == self.n_pred
        return len(self.assigned_true_indices) == self.n_true

    @property
    def is_total(self) -> bool:
        """Whether every source has an assignment."""
        if self.matching_direction == MatchingDirection.TRUE_TO_PRED:
            return len(self.assigned_true_indices) == self.n_true
        return len(self.assigned_pred_indices) == self.n_pred

    @property
    def is_bijective(self) -> bool:
        """Whether the matching defines a bijection between both sets."""
        return self.is_total and self.is_injective and self.is_surjective

    def filter_predictions(self, indices: list[int], inplace: bool = False):
        """
        Args:
            indices (list[int]): indices of predictions to keep in new matching.
        """
        indices = list(set(indices))
        if any(i < 0 or i >= self.n_pred for i in indices):
            raise IndexError("Prediction index out of bounds.")

        remap = {old: new for new, old in enumerate(indices)}

        pairs = [
            (true_idx, remap[pred_idx])
            for true_idx, pred_idx in self.pairs
            if pred_idx in remap
        ]

        if inplace:
            self.pairs = pairs
            self.n_pred = len(indices)
            return self
        return AssignmentResult(
            pairs=pairs,
            n_true=self.n_true,
            n_pred=len(indices),
            matching_direction=self.matching_direction,
        )

    def align_prediction_and_target(
        self,
        prediction: ODPrediction,
        target: ODTarget,
    ) -> tuple[ODPrediction, ODTarget]:
        """
        Align predicted and ground-truth boxes according to the assignment.

        Only matched boxes are retained, and both collections are
        reordered so that corresponding elements refer to the same
        matched pair.

        Args:
            prediction:
                Object detection predictions.
            target:
                Ground-truth annotations.

        Returns:
            A tuple containing the aligned ODPrediction and ODTarget.
        """
        true_indices = [true_idx for true_idx, _ in self.pairs]
        pred_indices = [pred_idx for _, pred_idx in self.pairs]
        return prediction[pred_indices], target[true_indices]


@runtime_checkable
class AssignmentMethod(Protocol):
    """
    General interface for an assignment method between a set of predictions and a set of ground truths.
    """

    def assign(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
    ) -> AssignmentResult: ...


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
        """
        Compute the pairwise cost matrix between predicted and ground-truth boxes.

        Args:
            y_pred (ODPrediction): Predicted bounding boxes.
            y_true (ODTarget): Ground-truth bounding boxes.

        Returns:
            TensorLike: A matrix of shape (n_true, n_pred) containing the pairwise costs.
        """


class IoUDistance:
    """1 - IoU."""

    def cost_matrix(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
    ) -> TensorLike:
        """
        Compute the pairwise cost matrix based on Intersection over Union (IoU) between predicted and ground-truth boxes.

        Args:
            y_pred (ODPrediction): Predicted bounding boxes.
            y_true (ODTarget): Ground-truth bounding boxes.

        Returns:
            TensorLike: A matrix of shape (n_true, n_pred) containing the pairwise costs, where each cost is defined as 1 - IoU.
        """
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
            y_pred.class_scores,
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
        """
        Compute the asymmetric Hausdorff distance between predicted and ground-truth boxes.

        Args:
            y_pred (ODPrediction): Predicted bounding boxes.
            y_true (ODTarget): Ground-truth bounding boxes.

        Returns:
            TensorLike: A matrix of shape (n_true, n_pred) containing the asymmetric Hausdorff distances.
        """
        true_boxes = y_true.coordinates
        pred_boxes = y_pred.coordinates

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
    """
    Combine localization and classification matching costs.

    The pairwise cost matrix is computed as::

        (1 - class_weight) * localization_cost + class_weight * classification_cost

    Args:
        localization_distance:
            Distance metric used for bounding-box localization.
            Defaults to IoUDistance.
        classification_distance:
            Distance metric used for classification.
            Defaults to LACDistance.
        class_weight:
            Weight assigned to the classification cost.
            Defaults to 0.25.

    Notes:
        Both distance metrics must return matrices of shape (n_true, n_pred).
    """

    def __init__(
        self,
        localization_distance: DistanceMetric = IoUDistance(),
        classification_distance: DistanceMetric = LACDistance(),
        class_weight: float = 0.25,
    ):
        self.localization_distance = localization_distance
        self.classification_distance = classification_distance
        self.class_weight = class_weight

    def cost_matrix(
        self,
        y_pred: ODPrediction,
        y_true: ODTarget,
    ) -> TensorLike:
        """
        Compute the mixed cost matrix combining localization and classification costs.

        Args:
            y_pred (ODPrediction): Predicted bounding boxes and class scores.
            y_true (ODTarget): Ground-truth bounding boxes and labels.

        Returns:
            TensorLike: A matrix of shape (n_true, n_pred) containing the mixed costs.
        """
        loc = self.localization_distance.cost_matrix(
            y_pred,
            y_true,
        )

        cls = self.classification_distance.cost_matrix(
            y_pred,
            y_true,
        )

        return (1 - self.class_weight) * loc + self.class_weight * cls


@runtime_checkable
class CostMatrixMatcher(Protocol):
    """
    Protocol for assignment algorithms operating on cost matrices.
    Implementations select pairs of source and target indices from a pairwise cost matrix.
    The assignment may be one-to-one or many-to-one depending on the matching algorithm.
    """

    def match(
        self,
        cost_matrix: TensorLike,
        valid_mask: TensorLike | None = None,
    ) -> list[tuple[int, int]]:
        """
        Compute assignments from a pairwise cost matrix.

        Args:
            cost_matrix:
                Matrix of shape (n_sources, n_targets), where lower
                values indicate preferable assignments.
            valid_mask:
                Optional boolean matrix of the same shape indicating
                which source-target pairs are admissible.

        Returns:
            List of (source_index, target_index) pairs.
        """


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
            indices = ops.convert_to_numpy(ops.argmin(cost_matrix, axis=1))
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

        valid_sources = ops.convert_to_numpy(ops.any(valid_mask, axis=1))

        return [(i, int(indices[i])) for i in range(n_source) if valid_sources[i]]


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
            unmatched_cost = valid_costs.max() + (n_source + 1) * max(cost_span, 1.0)

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
        direction:
            Direction of the assignment. Defaults to TRUE_TO_PRED.
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
            valid_mask = y_true.pairwise_iou(y_pred) >= self.iou_threshold

        if self.class_matching:
            predicted_labels = ops.argmax(
                y_pred.class_scores,
                axis=1,
            )

            same_class = y_true.labels[:, None] == predicted_labels[None, :]

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
        """
        Match predicted boxes with ground-truth objects.

        A pairwise cost matrix is computed using the configured distance metric.
        Optional IoU and class constraints are applied before passing the matrix to the matching algorithm.

        Args:
            y_pred:
                Object detection predictions for one image.
            y_true:
                Ground-truth annotations for the same image.

        Returns:
            An AssignmentResult containing the computed assignments, unmatched indices, and matching direction.
        """

        n_true = len(y_true)
        n_pred = len(y_pred)

        reverse = self.direction == MatchingDirection.PRED_TO_TRUE

        if n_true == 0 or n_pred == 0:
            return AssignmentResult(
                pairs=[],
                n_true=n_true,
                n_pred=n_pred,
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

        source_candidate_pairs = self.matcher.match(
            costs,
            valid_mask,
        )

        pairs = [
            (candidate_idx, source_idx) if reverse else (source_idx, candidate_idx)
            for source_idx, candidate_idx in source_candidate_pairs
        ]

        return AssignmentResult(
            pairs=pairs,
            n_true=n_true,
            n_pred=n_pred,
            matching_direction=self.direction,
        )


class RandomMatcher:
    """
    Randomly assign sources to admissible targets.
    Used mostly for testing and debugging.

    Random costs are generated and passed to either ArgminMatcher or HungarianMatcher
    depending on whether an injective assignment is required.

    Args:
        injective:
            Whether to enforce one-to-one matching.
            Defaults to False.
        seed:
            Optional random seed used to generate matching costs.
            Defaults to None.

    Notes:
        The provided cost matrix determines the number of sources and targets but its values are ignored.
    """

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


def check_assignment(
    assignment: AssignmentResult | None,
    *,
    direction: MatchingDirection | None = None,
) -> AssignmentResult:
    """
    Validate the presence and direction of a matching assignment.

    Args:
        assignment:
            Assignment result to validate.
        direction:
            Optional required matching direction.

    Returns:
        The original AssignmentResult.

    Raises:
        ValueError:
            If no assignment is provided or if its direction differs from the required direction.
    """
    if assignment is None:
        raise ValueError("This loss requires an AssignmentResult.")

    if direction is not None and assignment.matching_direction != direction:
        raise ValueError(
            f"This loss requires a {direction.value} assignment, "
            f"got {assignment.matching_direction.value}."
        )
    return assignment

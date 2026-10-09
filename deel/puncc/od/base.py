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
from dataclasses import dataclass, replace
from enum import StrEnum
from collections.abc import Callable
from typing import ClassVar, Generic, Literal, Self, TypeVar
from deel.puncc.backend.keras import ops
from deel.puncc.od.utils import IndexableUserList, IterableDataclassMixin
from deel.puncc.typing import TensorLike


class BoxExtensionMode(StrEnum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"

    @staticmethod
    def additive_processing(boxes, value) -> TensorLike:
        return boxes + ops.array([-value, -value, value, value])

    @staticmethod
    def multiplicative_processing(boxes, value) -> TensorLike:
        width = boxes[..., 2] - boxes[..., 0]
        height = boxes[..., 3] - boxes[..., 1]
        extension = ops.stack(
            [
                -value * width,
                -value * height,
                value * width,
                value * height,
            ],
            axis=-1,
        )
        return boxes + extension

    @property
    def processor(self) -> Callable[[TensorLike, float], TensorLike]:
        if self == BoxExtensionMode.ADDITIVE:
            return self.additive_processing
        elif self == BoxExtensionMode.MULTIPLICATIVE:
            return self.multiplicative_processing
        else:
            raise ValueError(
                f"Invalid mode: {self}. Must be one of {list(BoxExtensionMode)}"
            )


@dataclass(slots=True)
class Box:
    """
    Bounding box represented as (x_min, y_min, x_max, y_max).
    """

    xyxy: TensorLike  # x1, y1, x2, y2

    def __post_init__(self):
        shape = ops.shape(self.xyxy)
        assert shape == (4,), f"box must be of shape (4,), got {shape}"

    @property
    def xywh(self) -> TensorLike:
        x1, y1, x2, y2 = self.xyxy
        return ops.stack((x1, y1, x2 - x1, y2 - y1))

    @xywh.setter
    def xywh(self, new_box: TensorLike):
        x, y, w, h = new_box
        self.xyxy = ops.stack((x, y, x + w, y + h))

    @property
    def height(self) -> TensorLike:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def width(self) -> TensorLike:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def area(self) -> TensorLike:
        return self.height * self.width

    def contains(self, other_box: Box) -> TensorLike:
        x1, y1, x2, y2 = self.xyxy
        ox1, oy1, ox2, oy2 = other_box.xyxy

        return ops.logical_and(
            ops.logical_and(x1 <= ox1, y1 <= oy1),
            ops.logical_and(x2 >= ox2, y2 >= oy2),
        )

    def intersection(self, other_box: Box) -> Box:
        x1, y1, x2, y2 = self.xyxy
        ox1, oy1, ox2, oy2 = other_box.xyxy

        inter_x1 = ops.maximum(x1, ox1)
        inter_y1 = ops.maximum(y1, oy1)
        inter_x2 = ops.maximum(
            ops.minimum(x2, ox2),
            inter_x1,
        )
        inter_y2 = ops.maximum(
            ops.minimum(y2, oy2),
            inter_y1,
        )

        return Box(ops.stack((inter_x1, inter_y1, inter_x2, inter_y2)))

    @property
    def center(self) -> TensorLike:
        x1, y1, x2, y2 = self.xyxy
        return ops.stack(((x1 + x2) / 2, (y1 + y2) / 2))

    @property
    def top_left(self) -> TensorLike:
        return self.xyxy[:2]

    @property
    def bottom_right(self) -> TensorLike:
        return self.xyxy[2:]

    @property
    def is_empty(self) -> TensorLike:
        return ops.logical_or(
            self.width <= 0,
            self.height <= 0,
        )

    def iou(self, other_box: Box) -> TensorLike:
        intersection_area = self.intersection(other_box).area
        union_area = self.area + other_box.area - intersection_area
        return intersection_area / ops.maximum(union_area, 1e-12)

    @classmethod
    def from_xywh(cls, box: TensorLike) -> Box:
        return cls(cls.xywh_to_xyxy(box))

    @staticmethod
    def xywh_to_xyxy(box: TensorLike) -> TensorLike:
        x, y, w, h = box

        return ops.stack(
            (
                x,
                y,
                x + w,
                y + h,
            )
        )

    def equals(self, other: Box) -> TensorLike:
        return ops.all(self.xyxy == other.xyxy)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int | slice) -> TensorLike:
        return self.xyxy[idx]

    def __iter__(self):
        return iter(self.xyxy)

    def __repr__(self) -> str:
        return f"Box(xyxy={self.xyxy})"

    def __str__(self) -> str:
        return f"Box(xyxy={self.xyxy})"

    def extend(
        self,
        value: float,
        mode: BoxExtensionMode | str = BoxExtensionMode.ADDITIVE,
        *,
        inplace: bool = False,
    ) -> Box:
        mode = BoxExtensionMode(mode)
        xyxy = mode.processor(self.xyxy, value)
        if inplace:
            self.xyxy = xyxy
            return self
        return replace(self, xyxy=xyxy)


TBox = TypeVar("TBox", bound=Box)


@dataclass(slots=True)
class BoxSequence(IterableDataclassMixin[Box], Generic[TBox]):
    item_type: ClassVar[type[Box]] = Box
    boxes: TensorLike  # n, x1, y1, x2, y2

    def __post_init__(self):
        shape = ops.shape(self.boxes)
        assert (
            len(shape) == 2 and shape[-1] == 4
        ), f"boxes must be of shape (n, 4), got {shape}"

    @property
    def x1(self) -> TensorLike:
        return self.boxes[:, 0]

    @property
    def y1(self) -> TensorLike:
        return self.boxes[:, 1]

    @property
    def x2(self) -> TensorLike:
        return self.boxes[:, 2]

    @property
    def y2(self) -> TensorLike:
        return self.boxes[:, 3]

    @property
    def widths(self) -> TensorLike:
        return self.x2 - self.x1

    @property
    def heights(self) -> TensorLike:
        return self.y2 - self.y1

    @property
    def areas(self) -> TensorLike:
        return self.widths * self.heights

    def pairwise_iou(
        self,
        other: BoxSequence,
    ) -> TensorLike:
        boxes1 = ops.expand_dims(self.boxes, axis=1)
        boxes2 = ops.expand_dims(other.boxes, axis=0)

        inter_x1 = ops.maximum(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = ops.maximum(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = ops.minimum(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = ops.minimum(boxes1[..., 3], boxes2[..., 3])

        inter_width = ops.maximum(inter_x2 - inter_x1, 0.0)
        inter_height = ops.maximum(inter_y2 - inter_y1, 0.0)

        intersection = inter_width * inter_height

        area1 = ops.maximum(boxes1[..., 2] - boxes1[..., 0], 0.0) * ops.maximum(
            boxes1[..., 3] - boxes1[..., 1], 0.0
        )
        area2 = ops.maximum(boxes2[..., 2] - boxes2[..., 0], 0.0) * ops.maximum(
            boxes2[..., 3] - boxes2[..., 1], 0.0
        )
        union = area1 + area2 - intersection
        return intersection / ops.maximum(union, 1e-12)

    def pairwise_ioa(
        self,
        other: BoxSequence,
    ) -> TensorLike:
        boxes1 = ops.expand_dims(self.boxes, axis=1)
        boxes2 = ops.expand_dims(other.boxes, axis=0)

        inter_x1 = ops.maximum(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = ops.maximum(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = ops.minimum(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = ops.minimum(boxes1[..., 3], boxes2[..., 3])

        inter_width = ops.maximum(inter_x2 - inter_x1, 0.0)
        inter_height = ops.maximum(inter_y2 - inter_y1, 0.0)

        intersection = inter_width * inter_height

        other_area = ops.maximum(
            boxes2[..., 2] - boxes2[..., 0], 0.0
        ) * ops.maximum(boxes2[..., 3] - boxes2[..., 1], 0.0)

        return intersection / ops.maximum(other_area, 1e-12)

    def extend_boxes(
        self,
        value: float,
        mode: BoxExtensionMode | str = BoxExtensionMode.ADDITIVE,
        *,
        inplace: bool = False,
    ) -> Self:
        mode = BoxExtensionMode(mode)
        boxes = mode.processor(self.boxes, value)
        if inplace:
            self.boxes = boxes
            return self
        return replace(
            self,
            boxes=boxes,
        )


@dataclass(slots=True)
class BoxPrediction(Box):
    class_scores: TensorLike
    confidence: TensorLike
    class_set: TensorLike | None = None

    def __post_init__(self):
        super(BoxPrediction, self).__post_init__()
        if ops.ndim(self.class_scores) != 1:
            raise ValueError("class_scores must be 1D.")

    @property
    def predicted_class(self) -> TensorLike:
        return ops.argmax(self.class_scores)

    @property
    def prediction_set(self) -> TensorLike:
        if self.class_set is not None:
            return self.class_set

        return ops.expand_dims(
            self.predicted_class,
            axis=0,
        )

    def __iter__(self):
        yield self.xyxy
        yield self.class_scores
        yield self.confidence


class PredictionSetSequence(IndexableUserList[TensorLike]): ...


@dataclass(slots=True)
class ODPrediction(BoxSequence[BoxPrediction]):
    item_type: ClassVar[type[Box]] = BoxPrediction

    class_scores: TensorLike  # (n, n_classes)
    confidences: TensorLike  # (n,)
    class_sets: PredictionSetSequence | None = None

    def __post_init__(self):
        super(ODPrediction, self).__post_init__()
        assert (
            ops.shape(self.class_scores)[0] == ops.shape(self.boxes)[0]
        ), "class_scores and boxes must have the same length"
        assert (
            ops.shape(self.confidences)[0] == ops.shape(self.boxes)[0]
        ), "confidence and boxes must have the same length"
        assert self.class_scores.ndim == 2, "class_scores must be 2D"
        assert self.confidences.ndim == 1, "confidence must be 1D"

        if self.class_sets is not None:
            if not isinstance(self.class_sets, PredictionSetSequence):
                self.class_sets = PredictionSetSequence(self.class_sets)

            assert len(self.class_sets) == len(self)

    @property
    def num_classes(self) -> int:
        return self.class_scores.shape[1]

    def __add__(self, other: ODPrediction):
        assert (
            ops.shape(self.class_scores)[1] == ops.shape(other.class_scores)[1]
        ), "Cannot add ODResults with different number of classes"

        class_sets = None
        if self.class_sets is not None and other.class_sets is not None:
            class_sets = PredictionSetSequence(
                self.class_sets.data + other.class_sets.data
            )
        elif self.class_sets is not None or other.class_sets is not None:
            raise ValueError(
                "Cannot add ODResults if only one of them has class_sets defined"
            )

        return ODPrediction(
            boxes=ops.concatenate((self.boxes, other.boxes), axis=0),
            class_scores=ops.concatenate(
                (self.class_scores, other.class_scores), axis=0
            ),
            confidences=ops.concatenate(
                (self.confidences, other.confidences), axis=0
            ),
            class_sets=class_sets,
        )

    def __radd__(self, other: Literal[0] | ODPrediction):
        if other == 0:
            return self
        return self.__add__(other)

    def filter_by_confidence(
        self,
        threshold: float,
        *,
        inplace: bool = False,
    ) -> ODPrediction:
        indices = ops.where_1d(self.confidences >= threshold)
        filtered = self[indices]

        if inplace:
            self.boxes = filtered.boxes
            self.class_scores = filtered.class_scores
            self.confidences = filtered.confidences
            self.class_sets = filtered.class_sets
            return self
        return filtered

    @property
    def predicted_classes(self) -> TensorLike:
        return ops.argmax(
            self.class_scores,
            axis=-1,
        )

    @property
    def prediction_sets(self) -> list[TensorLike]:
        if self.class_sets is not None:
            return list(self.class_sets)

        return [
            ops.expand_dims(
                self.predicted_classes[i],
                axis=0,
            )
            for i in range(len(self))
        ]


@dataclass(slots=True)
class BoxTarget(Box):
    label: TensorLike


@dataclass(slots=True)
class ODTarget(BoxSequence[BoxTarget]):
    item_type: ClassVar[type[Box]] = BoxTarget
    labels: TensorLike  # (n_true,)


T_OD = TypeVar("T_OD", bound=BoxSequence)


class _ODSequence(IndexableUserList[T_OD], Generic[T_OD]):
    @property
    def boxes(self):
        return [item.boxes for item in self.data]

    def box_image_indices(self) -> TensorLike:
        if not self:
            return ops.zeros((0,), dtype="int32")

        return ops.concatenate(
            [
                ops.full(len(item), i, dtype="int32")
                for i, item in enumerate(self.data)
            ],
            axis=0,
        )


class ODTargetSequence(_ODSequence[ODTarget]):
    @property
    def labels(self):
        return [target.labels for target in self.data]

    # TODO : avoid duplication with ODPredictionSequence
    def boxwise(self) -> ODTarget:
        if not self:
            raise ValueError("Cannot flatten an empty ODTargetSequence.")
        boxes = ops.concatenate(self.boxes)
        labels = ops.concatenate(self.labels)
        return ODTarget(boxes, labels)


class ODPredictionSequence(_ODSequence[ODPrediction]):
    def filter_by_confidence(
        self,
        threshold: float,
        *,
        inplace: bool = False,
    ) -> ODPredictionSequence:
        if inplace:
            for pred in self.data:
                pred.filter_by_confidence(
                    threshold,
                    inplace=True,
                )
            return self

        return ODPredictionSequence(
            [
                pred.filter_by_confidence(
                    threshold,
                    inplace=False,
                )
                for pred in self.data
            ]
        )

    @property
    def class_scores(self):
        return [res.class_scores for res in self.data]

    @property
    def confidences(self):
        return [res.confidences for res in self.data]

    def boxwise(self) -> ODPrediction:
        if not self:
            raise ValueError("Cannot flatten an empty ODPredictionSequence.")
        boxes = ops.concatenate(self.boxes)
        class_scores = ops.concatenate(self.class_scores)
        confidences = ops.concatenate(self.confidences)

        class_sets = None
        if any(pred.class_sets is not None for pred in self.data):
            if not all(pred.class_sets is not None for pred in self.data):
                raise ValueError(
                    "Cannot flatten predictions if only some have class_sets defined."
                )
            class_sets = PredictionSetSequence(
                [
                    class_set
                    for pred in self.data
                    for class_set in pred.class_sets
                ]
            )
        return ODPrediction(
            boxes=boxes,
            class_scores=class_scores,
            confidences=confidences,
            class_sets=class_sets,
        )

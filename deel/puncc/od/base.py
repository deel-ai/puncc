from __future__ import annotations
from collections import UserList
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import ClassVar, Literal, Self
from deel.puncc import ops
from deel.puncc.od.utils import IndexableUserList, IterableDataclassMixin
from deel.puncc.typing import TensorLike

class BoxExtensionMode(StrEnum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"

@dataclass(slots=True)
class Box():
    """
        Bounding box represented as (x_min, y_min, x_max, y_max).
    """
    xyxy: TensorLike # x1, y1, x2, y2

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

        return Box(
            ops.stack(
                (inter_x1, inter_y1, inter_x2, inter_y2)
            )
        )

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
        return intersection_area / union_area

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

    def __getitem__(self, idx) -> TensorLike:
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
        mode: BoxExtensionMode = BoxExtensionMode.ADDITIVE,
        *,
        inplace: bool = False,
    ) -> Box:
        if mode == BoxExtensionMode.ADDITIVE:
            xyxy = self.xyxy + ops.array([-value, -value, value, value])
        elif mode == BoxExtensionMode.MULTIPLICATIVE:
            w = self.width
            h = self.height
            margin = value * ops.array([-w, -h, w, h])
            xyxy = self.xyxy + margin
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {list(BoxExtensionMode)}")
        if inplace:
            self.xyxy = xyxy
            return self
        return replace(self, xyxy=xyxy)

@dataclass(slots=True)
class BoxSequence(IterableDataclassMixin[Box]):
    item_type: ClassVar[type[Box]] = Box

    boxes: TensorLike # n, x1, y1, x2, y2
    def __post_init__(self):
        shape = ops.shape(self.boxes)
        assert len(shape) == 2 and shape[-1] == 4, f"boxes must be of shape (n, 4), got {shape}"

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

        inter_x1 = ops.maximum(
            boxes1[..., 0],
            boxes2[..., 0],
        )
        inter_y1 = ops.maximum(
            boxes1[..., 1],
            boxes2[..., 1],
        )
        inter_x2 = ops.minimum(
            boxes1[..., 2],
            boxes2[..., 2],
        )
        inter_y2 = ops.minimum(
            boxes1[..., 3],
            boxes2[..., 3],
        )

        inter_width = ops.maximum(
            inter_x2 - inter_x1,
            0.0,
        )
        inter_height = ops.maximum(
            inter_y2 - inter_y1,
            0.0,
        )

        intersection = inter_width * inter_height

        area1 = (
            ops.maximum(
                boxes1[..., 2] - boxes1[..., 0],
                0.0,
            )
            * ops.maximum(
                boxes1[..., 3] - boxes1[..., 1],
                0.0,
            )
        )

        area2 = (
            ops.maximum(
                boxes2[..., 2] - boxes2[..., 0],
                0.0,
            )
            * ops.maximum(
                boxes2[..., 3] - boxes2[..., 1],
                0.0,
            )
        )

        union = area1 + area2 - intersection
        return intersection / ops.maximum(union, 1e-12)
    
    def extend_boxes(
        self,
        value: float,
        mode: BoxExtensionMode = BoxExtensionMode.ADDITIVE,
        *,
        inplace: bool = False,
    ) -> Self:
        mode = BoxExtensionMode(mode)

        if mode == BoxExtensionMode.ADDITIVE:
            margins = ops.stack(
                (-value, -value, value, value)
            )
            boxes = self.boxes + margins

        elif mode == BoxExtensionMode.MULTIPLICATIVE:
            boxes = self.boxes + ops.stack(
                (
                    -value * self.widths,
                    -value * self.heights,
                    value * self.widths,
                    value * self.heights,
                ),
                axis=-1,
            )
        else:
            raise ValueError(f"Invalid extension mode: {mode}.")
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
    confidence:TensorLike
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

class PredictionSetSequence(
    IndexableUserList[TensorLike]
):
    ...

@dataclass(slots=True)
class ODPrediction(BoxSequence):
    item_type: ClassVar[type[BoxPrediction]] = BoxPrediction

    class_scores:TensorLike # (n, n_classes)
    confidences:TensorLike #(n,)
    class_sets: PredictionSetSequence[TensorLike] | None = None

    def __post_init__(self):
        super(ODPrediction, self).__post_init__()
        assert ops.shape(self.class_scores)[0] == ops.shape(self.boxes)[0], "class_scores and boxes must have the same length"
        assert ops.shape(self.confidences)[0] == ops.shape(self.boxes)[0], "confidence and boxes must have the same length"
        assert self.class_scores.ndim == 2, "class_scores must be 2D"
        assert self.confidences.ndim == 1, "confidence must be 1D"

    @property
    def num_classes(self) -> int:
        return self.class_scores.shape[1]
    
    def __add__(self, other:ODPrediction):
        assert ops.shape(self.class_scores)[1] == ops.shape(other.class_scores)[1], "Cannot add ODResults with different number of classes"
        return ODPrediction(
            boxes=ops.concatenate((self.boxes, other.boxes), axis=0),
            class_scores=ops.concatenate((self.class_scores, other.class_scores), axis=0),
            confidences=ops.concatenate((self.confidences, other.confidences), axis=0),
        )
    
    def __radd__(self, other:Literal[0]|ODPrediction):
        if other == 0:
            return self
        return self.__add__(other)

    def filter_by_confidence(
        self,
        threshold: float,
        *,
        inplace: bool = False,
    ) -> ODPrediction:
        indices = ops.where_1d(
            self.confidences >= threshold
        )
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
    label:TensorLike

@dataclass(slots=True)
class ODTarget(BoxSequence):
    item_type: ClassVar[type[BoxTarget]] = BoxTarget

    labels: TensorLike      # (n_true,)

class ODTargetSequence(IndexableUserList[ODTarget]):
    @property
    def boxes(self):
        return [target.boxes for target in self.data]

    @property
    def labels(self):
        return [target.labels for target in self.data]
    
class ODPredictionSequence(IndexableUserList[ODPrediction]):
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
    def boxes(self):
        return [res.boxes for res in self.data]

    @property
    def class_scores(self):
        return [res.class_scores for res in self.data]
    
    @property
    def confidences(self):
        return [res.confidences for res in self.data]


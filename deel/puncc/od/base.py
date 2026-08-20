from __future__ import annotations
from collections import UserList
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Generator, Literal, Protocol, Sequence, Union, overload, runtime_checkable
from deel.puncc import ops
from deel.puncc.cloning import clone_model
from deel.puncc.typing import TensorLike

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

    def __getitem__(self, idx:int) -> TensorLike:
        return self.xyxy[idx]

    def __iter__(self):
        return iter(self.xyxy)

    def __repr__(self) -> str:
        return f"Box(xyxy={self.xyxy})"

    def __str__(self) -> str:
        return f"Box(xyxy={self.xyxy})"

@dataclass(slots=True)
class BoxSequence():
    boxes: TensorLike # n, x1, y1, x2, y2
    def __post_init__(self):
        shape = ops.shape(self.boxes)
        assert len(shape) == 2 and shape[-1] == 4, f"boxes must be of shape (n, 4), got {shape}"

    def __getitem__(self, idx:int|slice) -> Box:
        if isinstance(idx, slice):
            return BoxSequence(self.boxes[idx])
        return Box(self.boxes[idx])
    
    def __len__(self) -> int:
        return ops.shape(self.boxes)[0]
    
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

        return intersection / ops.maximum(
            union,
            1e-12,
        )

@dataclass(slots=True)
class BoxPrediction(Box):
    class_scores: TensorLike
    confidence:TensorLike

    def __post_init__(self):
        super(BoxPrediction, self).__post_init__()
        assert self.class_scores.ndim == 1
        assert self.confidence >= 0 and self.confidence <= 1

    @property
    def predicted_class(self)->int:
        return int(ops.argmax(self.class_scores))

    def __iter__(self):
        yield self.xyxy
        yield self.class_scores
        yield self.confidence

@dataclass(slots=True)
class ODPrediction(BoxSequence):
    class_scores:TensorLike # (n, n_classes)
    confidences:TensorLike #(n,)

    def __post_init__(self):
        super().__post_init__()
        assert ops.shape(self.class_scores)[0] == ops.shape(self.boxes)[0], "class_scores and boxes must have the same length"
        assert ops.shape(self.confidences)[0] == ops.shape(self.boxes)[0], "confidence and boxes must have the same length"
        assert self.class_scores.ndim == 2, "class_scores must be 2D"
        self.confidence = ops.squeeze(self.confidences)
        assert self.confidence.ndim == 1, "confidence must be 1D"

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ODPrediction(
                boxes=self.boxes[idx],
                class_scores=self.class_scores[idx],
                confidences=self.confidences[idx],
            )

        return BoxPrediction(
            xyxy=self.boxes[idx],
            class_scores=self.class_scores[idx],
            confidence=self.confidences[idx],
        )
    
    def __iter__(self)->Generator[BoxPrediction]:
        for box, smx, conf in zip(self.boxes, self.class_scores, self.confidences):
            yield BoxPrediction(box, smx, conf)

    def __len__(self)->int:
        return self.boxes.shape[0]

    @property
    def num_classes(self) -> int:
        return self.class_scores.shape[1]

    @classmethod
    def empty(cls, num_classes:int):
        return cls(
            boxes=ops.zeros((0, 4)),
            class_scores=ops.zeros((0, num_classes)),
            confidence=ops.zeros((0,))
        )
    
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
        mask = self.confidences >= threshold
        indices = ops.where_1d(mask)

        boxes = ops.take(self.boxes, indices, axis=0)
        softmaxs = ops.take(self.softmaxs, indices, axis=0)
        confidences = ops.take(self.confidences, indices, axis=0)

        if inplace:
            self.boxes = boxes
            self.softmaxs = softmaxs
            self.confidences = confidences
            return self

        return ODPrediction(
            boxes=boxes,
            softmaxs=softmaxs,
            confidences=confidences,
        )

@dataclass(slots=True)
class BoxTarget(Box):
    label:TensorLike

@dataclass(slots=True)
class ODTarget(BoxSequence):
    labels: TensorLike      # (n_true,)


    @overload
    def __getitem__(self, idx: int) -> BoxTarget: ...

    @overload
    def __getitem__(self, idx: slice) -> ODTarget: ...

    def __getitem__(self, idx:int|slice) -> BoxTarget|ODTarget:
        if isinstance(idx, slice):
            return ODTarget(
                boxes=self.boxes[idx],
                labels=self.labels[idx],
            )
        return BoxTarget(
            xyxy=self.boxes[idx],
            label=self.labels[idx],
        )

class ODTargetSequence(UserList[ODTarget]):
    @property
    def boxes(self):
        return [res.boxes for res in self.data]
    
class ODPredictionSequence(UserList[ODPrediction]):
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

@runtime_checkable
class ODPredictor(Protocol):
    def __call__(self, X: Iterable[Any], *args, **kwargs) -> Sequence[tuple[TensorLike, TensorLike, TensorLike]]:
        ...

class ODModel():
    def __init__(self, predictor:ODPredictor):
        self.predictor = predictor

    def __call__(self, X: Iterable[Any], *args, **kwargs) -> ODPrediction:
        predictions = self.predictor(
            X,
            *args,
            **kwargs,
        )

        return ODPredictionSequence(
            [
                ODPrediction(
                    boxes=boxes,
                    class_scores=class_scores,
                    confidences=confidences,
                )
                for boxes, class_scores, confidences
                in predictions
            ]
        )
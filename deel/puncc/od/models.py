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
Base wrapper for Object Detection models and conformal prediction methods.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Self, Sequence, Protocol, runtime_checkable
from deel.puncc.core.risk_control import CRC
from deel.puncc.core.calibration import CalibrationContext
from deel.puncc.prediction_sets import lac_set
from deel.puncc.core.splitters import BaseSplitter, RandomSplitter
from deel.puncc.od.losses import ClassificationLoss, ConfidenceLoss, LocalizationLoss, ODLoss
from deel.puncc.od.matching import AssignmentMethod
from deel.puncc.od.utils import IndexableUserList
from deel.puncc.typing import PredSetFunction, TensorLike
from deel.puncc.od.base import BoxExtensionMode, ODPrediction, ODPredictionSequence, ODTarget, ODTargetSequence


@runtime_checkable
class ODModel(Protocol):
    def __call__(self, X: Iterable[Any], *args:Any, **kwargs:Any) -> Sequence[tuple[TensorLike, TensorLike, TensorLike]]:
        ...

class ODPredictor():
    def __init__(self, model:ODModel):
        self.model = model

    def __call__(self, X: Iterable[Any], *args:Any, **kwargs:Any) -> ODPredictionSequence:
        predictions = self.model(
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

class ODPostProcessing(Protocol):
    def __call__(self, predictions:ODPredictionSequence, lambd:float) -> ODPredictionSequence:
        ...

class ConfidenceThresholdingPostProcessing(ODPostProcessing):
    def __call__(self, predictions:ODPredictionSequence, lambd:float) -> ODPredictionSequence:
        return predictions.filter_by_confidence(1 - lambd)


class BoxExtensionPostProcessing(ODPostProcessing):
    def __init__(self, extension_mode:BoxExtensionMode=BoxExtensionMode.ADDITIVE):
        self.extension_mode = extension_mode

    def __call__(self, predictions:ODPredictionSequence, lambd:float) -> ODPredictionSequence:
        return ODPredictionSequence(
            [
                p.extend_boxes(lambd, mode=self.extension_mode, inplace=False)
                for p in predictions
            ]
        )
    
class ClassificationPredSetPostProcessing(ODPostProcessing):
    def __init__(self, pred_set_function:PredSetFunction=lac_set()):
        self.pred_set_function = pred_set_function

    def __call__(self, predictions:ODPredictionSequence, lambd:float) -> ODPredictionSequence:
        return ODPredictionSequence(
            [
                replace(
                    p,
                    class_sets=IndexableUserList(self.pred_set_function(p.class_scores, lambd))
                )
                for p
                in predictions
            ]
        )

class ODCRC(CRC[ODPredictionSequence, Sequence[ODTarget], ODPredictionSequence]):
    __slots__ = (
        "od_loss",
        "assignment_method",
    )

    def _loss_function(
        self,
        predictions: Sequence[ODPrediction],
        targets: Sequence[ODTarget],
    ) -> TensorLike:
        if self.assignment_method is None:
            return self.od_loss(
                predictions,
                targets,
            )

        assignments = [
            self.assignment_method.assign(
                prediction,
                target,
            )
            for prediction, target
            in zip(predictions, targets)
        ]

        return self.od_loss(
            predictions,
            targets,
            assignments,
        )

    def __init__(
        self,
        model: ODPredictor,
        loss: ODLoss,
        postprocessor: ODPostProcessing,
        *,
        assignment_method:AssignmentMethod|None=None,
        **kwargs:Any,
    ):
        self.od_loss = loss
        self.assignment_method = assignment_method

        super().__init__(
            model=model,
            loss_function=self._loss_function,
            postprocessor=postprocessor,
            loss_function_upper_bound=getattr(
                loss,
                "upper_bound",
                1.0,
            ),
            **kwargs,
        )

class ODConfidenceCRC(ODCRC):
    def __init__(
        self,
        model: ODPredictor,
        loss: ConfidenceLoss,
        *,
        assignment_method: AssignmentMethod | None = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            loss=loss,
            postprocessor=ConfidenceThresholdingPostProcessing(),
            assignment_method=assignment_method,
            **kwargs
        )

class ODLocalizationCRC(ODCRC):
    def __init__(
        self,
        model: ODPredictor,
        loss: LocalizationLoss,
        box_extension_mode: BoxExtensionMode = BoxExtensionMode.ADDITIVE,
        *,
        assignment_method: AssignmentMethod | None = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            loss=loss,
            postprocessor=BoxExtensionPostProcessing(box_extension_mode),
            assignment_method=assignment_method,
            **kwargs
        )

class ODClassificationCRC(ODCRC):
    def __init__(
        self,
        model: ODPredictor,
        loss: ClassificationLoss,
        pred_set_function: PredSetFunction = lac_set(),
        *,
        assignment_method: AssignmentMethod | None = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            loss=loss,
            postprocessor=ClassificationPredSetPostProcessing(pred_set_function),
            assignment_method=assignment_method,
            **kwargs
        )

class TripleCRC:
    def __init__(
        self,
        model: ODPredictor,
        confidence_loss:ConfidenceLoss,
        localization_loss:LocalizationLoss,
        classification_loss:ClassificationLoss,
        splitter: BaseSplitter|None = None,
        box_extension_mode: BoxExtensionMode = BoxExtensionMode.ADDITIVE,
        pred_set_function: PredSetFunction = lac_set(),
        confidence_kwargs: dict | None = None,
        localization_kwargs: dict | None = None,
        classification_kwargs: dict | None = None,
    ):
        self.model = model

        self.confidence_calibrator = ODConfidenceCRC(
            model=model,
            loss=confidence_loss,
            **(confidence_kwargs or {}),
        )
        self.localization_calibrator = ODLocalizationCRC(
            model=model,
            loss=localization_loss,
            box_extension_mode=box_extension_mode,
            **(localization_kwargs or {}),
        )
        self.classification_calibrator = ODClassificationCRC(
            model=model,
            loss=classification_loss,
            pred_set_function=pred_set_function,
            **(classification_kwargs or {}),
        )
        self.splitter = splitter if splitter is not None else RandomSplitter(ratio=0.5)

    def calibrate(
        self,
        X_calib: Iterable[Any],
        y_calib: ODTargetSequence,
    ) -> Self:
        calibration_context = CalibrationContext(
            X_calib=X_calib,
            y_calib=y_calib,
            y_pred=self.model(X_calib),
        )

        (
            confidence_context,
            loc_class_context,
        ) = self.splitter.split_context(
            calibration_context
        )[0]

        self.confidence_calibrator.calibration_context = (
            self.confidence_calibrator.compute_calibration_state(
                confidence_context
            )
        )

        loc_context = loc_class_context.copy()
        class_context = loc_class_context.copy()

        self.localization_calibrator.calibration_context = (
            self.localization_calibrator.compute_calibration_state(
                loc_context
            )
        )

        self.classification_calibrator.calibration_context = (
            self.classification_calibrator.compute_calibration_state(
                class_context
            )
        )

        return self
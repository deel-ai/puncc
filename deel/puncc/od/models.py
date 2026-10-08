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
from typing import Any, ClassVar, Iterable, Self, Sequence, Protocol, runtime_checkable
import warnings


from deel.puncc.core.conformal import ConformalPrediction, ConformalPredictor, NoopConformalPredictor
from deel.puncc.core.predictors import make_predictor
from deel.puncc.core.risk_control import CRC, Postprocessor, RiskLossFunction
from deel.puncc.core.calibration import CalibrationContext
from deel.puncc.prediction_sets import lac_set
from deel.puncc.core.splitters import BaseSplitter, RandomSplitter
from deel.puncc.od.losses import ClassificationLoss, ConfidenceLoss, LocalizationLoss, ODLoss
from deel.puncc.od.matching import AssignmentMethod, MatchingDirection
from deel.puncc.od.utils import IndexableUserList
from deel.puncc.typing import FitFunction, PredSetFunction, Predictor, PredictorLike, TensorLike
from deel.puncc.od.base import BoxExtensionMode, ODPrediction, ODPredictionSequence, ODTarget, ODTargetSequence
from deel.puncc.backend import ops
from deel.puncc.warnings import CalibrationWarning

@runtime_checkable
class ODPredictor(Predictor[Sequence[tuple[TensorLike, TensorLike, TensorLike]]], Protocol):
    ...

@runtime_checkable
class ODPredictorLike(PredictorLike[Sequence[tuple[TensorLike, TensorLike, TensorLike]]], Protocol):
    ...

class PunccODPredictor():
    def __init__(self, model:ODPredictor|ODPredictorLike):
        self.model = make_predictor(model)

    def __call__(self, X: Iterable[Any], *args:Any, **kwargs:Any) -> ODPredictionSequence:
        predictions = self.model(X,*args, **kwargs)
        if isinstance(predictions, ODPredictionSequence):
            return predictions
        return ODPredictionSequence(
            [ODPrediction(boxes=boxes, class_scores=class_scores, confidences=confidences)
                for boxes, class_scores, confidences in predictions])
    
    def fit(self, X:Iterable[Any], y:ODTargetSequence, *args:Any, **kwargs:Any) -> Self:
        fit_method = getattr(self.model, "fit", None)

        if not callable(fit_method):
            raise NotImplementedError(
                "The underlying OD model does not have a fit method."
            )
        fit_method(X, y, *args, **kwargs)
        return self
    
def make_puncc_od_predictor(model:Any) -> PunccODPredictor:
    if isinstance(model, PunccODPredictor):
        return model
    if isinstance(model, (ODPredictor, ODPredictorLike)):
        return PunccODPredictor(model)
    raise ValueError(f"Model of type {type(model)} is not compatible with ODPredictor. Please provide an ODModel or an ODPredictor instance.")

class _ODConformalPredictor(ConformalPredictor):
    prediction_attr_name: ClassVar[str]
    target_attr_name: ClassVar[str]
    prediction_set_attr_name: ClassVar[str]

    def __init__(self, model:ODPredictor|ODPredictorLike,
                 fit_function:FitFunction|None = None,
                 assignment_method:AssignmentMethod|None = None,
                 **kwargs:Any):
        self.assignment_method = assignment_method
        super().__init__(model=model, fit_function=fit_function, **kwargs)
        self.model = make_puncc_od_predictor(self.model)

    def fit(self, *args, **kwargs) -> Self:
        super().fit(*args, **kwargs)
        self.model = make_puncc_od_predictor(self.model)
        return self

    def project_prediction(self,prediction: ODPrediction) -> Any:
        return getattr(prediction, self.prediction_attr_name)

    def project_target(self, target: ODTarget) -> Any:
        return getattr(target, self.target_attr_name)

    def add_assignments_to_context(self, calibration_context:CalibrationContext)->CalibrationContext:
        if (self.assignment_method is not None and "assignments" not in calibration_context):
            calibration_context.update(
                assignments=IndexableUserList([self.assignment_method.assign(prediction, target) 
                                            for prediction, target in zip(calibration_context.y_pred, calibration_context.y_calib, strict=True)])
            )
        return calibration_context
    
    def build_od_prediction_set(self, prediction: ODPredictionSequence, prediction_set: Sequence[Any]) -> ODPredictionSequence:
        return ODPredictionSequence([
            replace(prediction_i, **{self.prediction_set_attr_name: prediction_set_i})
            for prediction_i, prediction_set_i in zip(prediction, prediction_set, strict=True)
        ])
    
class ODClassificationMixin:
    prediction_attr_name: ClassVar[str] = "class_scores"
    target_attr_name: ClassVar[str] = "labels"
    prediction_set_attr_name: ClassVar[str] = "class_sets"

class ODLocalizationMixin:
    prediction_attr_name: ClassVar[str] = "boxes"
    target_attr_name: ClassVar[str] = "boxes"
    #prediction_set_attr_name: ClassVar[str] = "boxes"


    def build_od_prediction_set(
        self,
        prediction: ODPredictionSequence,
        prediction_set: Sequence[Any],
    ) -> tuple[ODPredictionSequence, ODPredictionSequence]:

        inner = ODPredictionSequence([
            replace(prediction_i, boxes=prediction_set_i[..., 0])
            for prediction_i, prediction_set_i in zip(
                prediction, prediction_set, strict=True
            )
        ])

        outer = ODPredictionSequence([
            replace(prediction_i, boxes=prediction_set_i[..., 1])
            for prediction_i, prediction_set_i in zip(
                prediction, prediction_set, strict=True
            )
        ])

        return inner, outer

class BoxWiseODCP(_ODConformalPredictor):
    """
    Apply conformal calibration at the matched-box level.

    Notes:
        Finite-sample conformal coverage requires the box-level
        calibration scores and test score to satisfy the relevant
        exchangeability assumptions.

        Exchangeability of images alone does not imply exchangeability
        of all boxes pooled across images.
    """

    def boxwise_calibration_context(self,calibration_context: CalibrationContext) -> CalibrationContext:
        X_boxwise = []
        predictions = []
        targets = []

        if "assignments" not in calibration_context:
            raise ValueError("Calibration context must contain assignments.")
        
        for X, prediction, target, assignment in zip(
            calibration_context.X_calib,
            calibration_context.y_pred,
            calibration_context.y_calib,
            calibration_context.assignments,
            strict=True,
        ):
            aligned_prediction, aligned_target = assignment.align_prediction_and_target(prediction, target)

            predictions.append(aligned_prediction)
            targets.append(aligned_target)

            X_boxwise.extend([X] * len(aligned_prediction))

        return CalibrationContext(
            X_calib=IndexableUserList(X_boxwise),
            y_pred=self.boxwise_prediction(ODPredictionSequence(predictions)),
            y_calib=self.boxwise_target(ODTargetSequence(targets)),
        )

    def boxwise_prediction(self, prediction: ODPredictionSequence) -> TensorLike:
        return self.project_prediction(prediction.boxwise())

    def boxwise_target(self, target: ODTargetSequence)->TensorLike:
        return self.project_target(target.boxwise())

    def boxwise_X(self, X: Iterable[Any], prediction: ODPredictionSequence) -> IndexableUserList:
        X_boxwise = []
        for x, pred in zip(X, prediction, strict=True):
            X_boxwise.extend([x] * len(pred))
        return IndexableUserList(X_boxwise)

    def compute_calibration_state(self, calibration_context: CalibrationContext) -> CalibrationContext:
        calibration_context = self.add_assignments_to_context(calibration_context)
        calibration_context = self.boxwise_calibration_context(calibration_context)
        return super().compute_calibration_state(calibration_context)

    def conformalize(self,
        prediction: ODPredictionSequence,
        alpha: float | TensorLike,
        calibration_context: CalibrationContext,
        *,
        X: Iterable[Any] | None = None) -> ConformalPrediction[Any, Any]:
        X_boxwise = None

        if X is not None:
            X_boxwise = self.boxwise_X(X, prediction)

        cp = super().conformalize(
            prediction=self.boxwise_prediction(prediction),
            alpha=alpha,
            calibration_context=calibration_context,
            X=X_boxwise,
        )
        prediction_sets = self.unboxwise_prediction_set(cp.prediction_set, prediction)

        return ConformalPrediction(
            prediction=prediction,
            prediction_set=self.build_od_prediction_set(prediction, prediction_sets))

    def unboxwise_prediction_set(self, prediction_set: Any, prediction: ODPredictionSequence) -> IndexableUserList:
        prediction_sets = []
        start = 0
        for prediction_i in prediction:
            stop = start + len(prediction_i)
            prediction_sets.append(prediction_set[start:stop])
            start = stop
        return IndexableUserList(prediction_sets)


### TO DO IMAGE WISE CONFORMAL PREDICTION, USE ODCRC AND ODLOSS FUNCTION


# class ImageWisePostprocessor:
#     def __init__(self, postprocessor):
#         self.postprocessor = postprocessor

#     def __call__(self, predictions: IndexableUserList, lambd: float) -> IndexableUserList:
#         return IndexableUserList([self.postprocessor(prediction, lambd) for prediction in predictions])

# class ImageWiseCP(_ODConformalPredictor, CRC):
#     def __init__(self, *args, 
#                  postprocessor: Postprocessor,
#                  penalize_unmatched_boxes:bool=True, **kwargs):
#         postprocessor = ImageWisePostprocessor(postprocessor)
#         super().__init__(*args, postprocessor=postprocessor, **kwargs)
#         self.penalize_unmatched_boxes = penalize_unmatched_boxes

#     def imagewise_prediction(self, prediction: ODPredictionSequence) -> IndexableUserList:
#         return IndexableUserList([self.project_prediction(prediction_i) for prediction_i in prediction])

#     def conformalize(self, prediction: ODPredictionSequence,alpha: float | TensorLike, calibration_context: CalibrationContext, *, X: Iterable[Any] | None = None) -> ConformalPrediction[Any, Any]:
#         cp = super().conformalize(
#             prediction=self.imagewise_prediction(prediction),
#             alpha=alpha,
#             calibration_context=calibration_context,
#             X=X)
#         return ConformalPrediction(
#             prediction=prediction,
#             prediction_set=self.build_od_prediction_set(prediction, cp.prediction_set)
#         )

#     def imagewise_calibration_context(self, calibration_context: CalibrationContext) -> CalibrationContext:
#         predictions = []
#         targets = []
        
#         if "assignments" not in calibration_context:
#             raise ValueError("Calibration context must contain assignments.")
        
#         for prediction, target, assignment in zip(
#             calibration_context.y_pred,
#             calibration_context.y_calib,
#             calibration_context.assignments,
#             strict=True,
#         ):
#             prediction, target = assignment.align_prediction_and_target(prediction, target)
#             predictions.append(self.project_prediction(prediction))
#             targets.append(self.project_target(target))

#         calibration_context.update(
#             y_pred=IndexableUserList(predictions),
#             y_calib=IndexableUserList(targets),
#         )
#         return calibration_context

#     def compute_calibration_state(
#         self,
#         calibration_context: CalibrationContext,
#     ) -> CalibrationContext:
#         calibration_context = self.add_assignments_to_context(calibration_context)
#         calibration_context = self.imagewise_calibration_context(calibration_context)
#         return super().compute_calibration_state(calibration_context)

#     def compute_loss(
#         self,
#         y_pred: IndexableUserList,
#         y_true: IndexableUserList,
#         calibration_context: CalibrationContext,
#     ) -> TensorLike:
#         image_losses = []

#         for prediction_i, target_i, assignment in zip(
#             y_pred,
#             y_true,
#             calibration_context.assignments,
#             strict=True,
#         ):
#             n_matched = len(prediction_i)
#             n_unmatched = 0
#             if self.penalize_unmatched_boxes:
#                 n_unmatched = len(assignment.unmatched_true_indices())

#             if n_matched == 0:
#                 image_losses.append(ops.convert_to_tensor(self.B if n_unmatched > 0 else 0.0))
#                 continue

#             total_loss = ops.sum(self.loss_function(prediction_i, target_i))

#             if self.penalize_unmatched_boxes:
#                 total_loss += len(assignment.unmatched_true_indices()) * self.B

#             image_losses.append(total_loss / (n_matched + n_unmatched))
#         return ops.stack(image_losses, axis=0)

#     def _r_hat(self, lambd: float, calibration_context: CalibrationContext) -> float:
#         predictions = self.postprocessor(calibration_context.y_pred, lambd)
#         losses = self.compute_loss(predictions, calibration_context.y_calib, calibration_context)
#         return ops.item(ops.mean(losses))

def conf_threshold_postprocessing(predictions:ODPredictionSequence, lambd:float) -> ODPredictionSequence:
        return predictions.filter_by_confidence(1 - lambd)

class ODCRC(_ODConformalPredictor, CRC):
    def compute_calibration_state(self, calibration_context):
        calibration_context = (self.add_assignments_to_context(calibration_context))
        return super().compute_calibration_state(calibration_context)

    def _r_hat(
        self,
        lambd: float,
        calibration_context: CalibrationContext,
    ) -> float:
        predictions = self.postprocessor(calibration_context.y_pred, lambd)
        assignments = calibration_context.assignments if "assignments" in calibration_context else None
        losses = self.loss_function(predictions, calibration_context.y_calib, assignments)
        return ops.item(ops.mean(losses))

    def get_calibration_set_size(self, calibration_context:CalibrationContext) -> int:
        loss:ODLoss = self.loss_function

        if not loss.boxwise:
            return calibration_context.size
        # TODO : get the good number of calculated losses on the calibration set.
        warnings.warn(
            "ODCRC is using box-level losses and the number of box-level losses as the calibration size. This is statistically valid "
            "only if the corresponding box-level losses satisfy the exchangeability assumptions required by CRC. "
            "Exchangeability of images alone does not imply exchangeability of boxes pooled across images.",
            CalibrationWarning,
            stacklevel=2,
        )

        assignments = calibration_context.assignments

        if not loss.penalize_unmatched_boxes:
            return sum(
                len(assignment.matched_pairs)
                for assignment in assignments
            )
        raise NotImplementedError("Cannot infer the calibration size for this boxwise OD loss when unmatched target boxes are penalized.")
        # return sum(
        #     len(assignment.matched_pairs)
        #     + len(assignment.unassigned_source_indices) # TODO : check that
        #     for assignment in assignments
        # )

class ODConfidenceCRC(ODCRC):
    def __init__(
        self,
        model,
        *,
        loss_function:ODLoss,
        loss_function_upper_bound=None,
        assignment_method=None,
        **kwargs,
    ):
        if loss_function.boxwise and not loss_function.penalize_unmatched_boxes:
            raise NotImplementedError(
                "ODConfidenceCRC does not currently support boxwise losses "
                "with penalize_unmatched_boxes=False because the population "
                "of evaluated boxes depends on the confidence threshold."
            )
        super().__init__(
            model=model,
            loss_function=loss_function,
            postprocessor=conf_threshold_postprocessing,
            assignment_method=assignment_method,
            loss_function_upper_bound=loss_function_upper_bound,
            **kwargs,
        )

    def _r_hat(self, lambd, calibration_context):
        threshold = 1 - lambd
        predictions = []
        if "assignments" in calibration_context:
            filtered_assignments = []
            assignments = calibration_context.assignments
        else:
            filtered_assignments = None
            assignments = None

        for i, prediction in enumerate(calibration_context.y_pred):
            kept_indices = ops.where_1d(prediction.confidences >= threshold)
            predictions.append(prediction[kept_indices])
            if filtered_assignments is not None:
                # assignments is not None
                filtered_assignments.append(assignments[i].filter_predictions(ops.tolist(kept_indices)))

        losses = self.loss_function(
            ODPredictionSequence(predictions),
            calibration_context.y_calib,
            assignments=(
                None
                if filtered_assignments is None
                else IndexableUserList(filtered_assignments)
            ),
        )
        return ops.item(ops.mean(losses))

class ODClassificationCRC(ODCRC):
    @staticmethod
    def get_postprocessor(pred_set_function: PredSetFunction):
        def postprocessor(predictions: ODPredictionSequence, lambd: float) -> ODPredictionSequence:
            return ODPredictionSequence([
                replace(prediction, class_sets=pred_set_function(prediction.class_scores, lambd)) for prediction in predictions
            ])
        return postprocessor
    
    def __init__(
        self,
        model: ODPredictor | ODPredictorLike,
        *,
        loss_function: ODLoss,
        pred_set_function: PredSetFunction = lac_set(),
        assignment_method: AssignmentMethod | None = None,
        loss_function_upper_bound: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            loss_function=loss_function,
            postprocessor=self.get_postprocessor(pred_set_function),
            assignment_method=assignment_method,
            loss_function_upper_bound=loss_function_upper_bound,
            **kwargs,
        )

class ODLocalizationCRC(ODCRC):
    @staticmethod
    def get_postprocessor(box_extension_mode: BoxExtensionMode):
        def postprocessor(predictions: ODPredictionSequence, lambd: float,) -> ODPredictionSequence:
            return ODPredictionSequence([
                replace(prediction, boxes=box_extension_mode.processor(prediction.boxes, lambd))
                for prediction in predictions])
        return postprocessor

    def __init__(
        self,
        model: ODPredictor | ODPredictorLike,
        *,
        loss_function: ODLoss,
        box_extension_mode: BoxExtensionMode = BoxExtensionMode.ADDITIVE,
        assignment_method: AssignmentMethod | None = None,
        loss_function_upper_bound: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            loss_function=loss_function,
            postprocessor=self.get_postprocessor(box_extension_mode),
            loss_function_upper_bound=loss_function_upper_bound,
            assignment_method=assignment_method,
            **kwargs)

class NoopODConformalPredictor(_ODConformalPredictor, NoopConformalPredictor):
    pass



class TripleConformalPredictor:
    """Orchestrate confidence, localization and classification conformal prediction for OD."""

    __slots__ = (
        "model",
        "confidence_cp",
        "localization_cp",
        "classification_cp",
        "confidence_threshold",
        "splitter",
        "_confidence_context",
        "_downstream_context",
        "_localization_context",
        "_classification_context",
        "_calibrated",
    )

    def __init__(
        self,
        model: ODPredictor | ODPredictorLike,
        *,
        confidence_cp: _ODConformalPredictor | None = None,
        localization_cp: _ODConformalPredictor | None = None,
        classification_cp: _ODConformalPredictor | None = None,
        confidence_threshold: float = 0.0,
        splitter: BaseSplitter | None = None,
    ):
        self.model = make_puncc_od_predictor(model)

        if confidence_cp is None and localization_cp is None and classification_cp is None:
            raise ValueError("At least one of confidence_cp, localization_cp or classification_cp must be provided.")

        self.confidence_cp = confidence_cp if confidence_cp is not None else NoopODConformalPredictor(model)
        self.localization_cp = localization_cp if localization_cp is not None else NoopODConformalPredictor(model)
        self.classification_cp = classification_cp  if classification_cp is not None else NoopODConformalPredictor(model)

        self.confidence_threshold = confidence_threshold
        self.splitter = splitter if splitter is not None else RandomSplitter(ratio=0.5)
        self.reset_calibration()

    def reset_calibration(self) -> Self:
        self._calibrated = False
        self._confidence_context = CalibrationContext()
        self._downstream_context = CalibrationContext()
        self._localization_context = CalibrationContext()
        self._classification_context = CalibrationContext()
        return self

    @staticmethod
    def _merge_classification(
        prediction: ODPredictionSequence,
        classification: ODPredictionSequence,
    ) -> ODPredictionSequence:
        return ODPredictionSequence([
            replace(pred, class_sets=class_pred.class_sets)
            for pred, class_pred in zip(prediction, classification, strict=True)
        ])

    @classmethod
    def _merge_prediction_sets(cls, localization: Any, classification: ODPredictionSequence) -> Any:
        if isinstance(localization, ODPredictionSequence):
            return cls._merge_classification(localization, classification)

        if (
            isinstance(localization, tuple)
            and len(localization) == 2
            and all(isinstance(prediction, ODPredictionSequence) for prediction in localization)
        ):
            inner, outer = localization
            return (
                cls._merge_classification(inner, classification),
                cls._merge_classification(outer, classification),
            )

        raise TypeError(
            "localization_cp must produce either an ODPredictionSequence "
            "or a pair of ODPredictionSequence objects."
        )

    def calibrate(self, X_calib: Iterable[Any], y_calib: ODTargetSequence) -> Self:
        self.reset_calibration()
        prediction = self.model(X_calib).filter_by_confidence(self.confidence_threshold)
        context = CalibrationContext(X_calib=X_calib, y_calib=y_calib, y_pred=prediction)

        # if no conf calibration phase : all calibration goes to localization and classification
        if isinstance(self.confidence_cp, NoopODConformalPredictor):
            self._localization_context = self.localization_cp.compute_calibration_state(context.copy())
            self._classification_context = self.classification_cp.compute_calibration_state(context.copy())
        # if no localization and classification calibration phase : all calibration goes to confidence
        elif isinstance(self.classification_cp, NoopODConformalPredictor) and isinstance(self.localization_cp, NoopODConformalPredictor):
            self._confidence_context = self.confidence_cp.compute_calibration_state(context.copy())
        else:
        # otherwise : split context in two parts
            confidence_context, self._downstream_context = self.splitter.split_context(context)[0]
            self._confidence_context = self.confidence_cp.compute_calibration_state(confidence_context)
        self._calibrated = True
        return self

    def conformalize(
        self,
        prediction: ODPredictionSequence,
        *,
        alpha_conf: float | TensorLike | None = None,
        alpha_loc: float | TensorLike | None = None,
        alpha_class: float | TensorLike | None = None,
        X: Iterable[Any] | None = None,
    ) -> ConformalPrediction[ODPredictionSequence, Any]:
        if self._calibrated is False:
            raise RuntimeError("The TripleConformalPredictor must be calibrated before conformalization.")
        # check alphas :
        for alpha, cp in zip([alpha_conf, alpha_loc, alpha_class],
                                        [self.confidence_cp, self.localization_cp, self.classification_cp]):
            if alpha is None and not isinstance(cp, NoopODConformalPredictor):
                raise ValueError(f"Alpha value is required for {cp.__class__.__name__}.")

        thresholded_prediction = prediction.filter_by_confidence(self.confidence_threshold)
        confidence_prediction = self.confidence_cp.conformalize(thresholded_prediction, alpha_conf, self._confidence_context, X=X).prediction_set

        if isinstance(self.confidence_cp, NoopODConformalPredictor) or (isinstance(self.localization_cp, NoopODConformalPredictor) and isinstance(self.classification_cp, NoopODConformalPredictor)):
            localization_context = self._localization_context
            classification_context = self._classification_context
        else:
            filtered_prediction = self.confidence_cp.conformalize(self._downstream_context.y_pred, alpha_conf, self._confidence_context, X=self._downstream_context.X_calib).prediction_set
            context = self._downstream_context.copy().update(y_pred=filtered_prediction)
            localization_context = self.localization_cp.compute_calibration_state(context.copy())
            classification_context = self.classification_cp.compute_calibration_state(context.copy())

        localization = self.localization_cp.conformalize(confidence_prediction, alpha_loc, localization_context, X=X).prediction_set
        classification = self.classification_cp.conformalize(confidence_prediction, alpha_class, classification_context, X=X).prediction_set


        prediction_set = self._merge_prediction_sets(localization, classification)

        return ConformalPrediction(prediction=prediction, prediction_set=prediction_set)

    def predict(
        self,
        X: Iterable[Any],
        *,
        alpha_conf: float | TensorLike | None = None,
        alpha_loc: float | TensorLike | None = None,
        alpha_class: float | TensorLike | None = None,
    ) -> ConformalPrediction[ODPredictionSequence, Any]:
        prediction = self.model(X)

        return self.conformalize(
            prediction,
            alpha_conf=alpha_conf,
            alpha_loc=alpha_loc,
            alpha_class=alpha_class,
            X=X,
        )


class TripleCRC(TripleConformalPredictor):
    def __init__(
        self,
        model: ODPredictor | ODPredictorLike,
        confidence_loss: ConfidenceLoss | None = None,
        localization_loss: LocalizationLoss | None = None,
        classification_loss: ClassificationLoss | None = None,
        *,
        assignment_method: AssignmentMethod | None = None,
        confidence_threshold: float = 0.0,
        splitter: BaseSplitter | None = None,
        box_extension_mode: BoxExtensionMode = BoxExtensionMode.ADDITIVE,
        pred_set_function: PredSetFunction = lac_set(),
        confidence_kwargs: dict | None = None,
        localization_kwargs: dict | None = None,
        classification_kwargs: dict | None = None,
    ):
        model = make_puncc_od_predictor(model)

        confidence_cp = None
        localization_cp = None
        classification_cp = None

        if confidence_loss is not None:
            kwargs = dict(confidence_kwargs or {})
            kwargs["assignment_method"] = assignment_method
            confidence_cp = ODConfidenceCRC(
                model=model,
                loss_function=confidence_loss,
                **kwargs,
            )

        if localization_loss is not None:
            kwargs = dict(localization_kwargs or {})
            kwargs["assignment_method"] = assignment_method
            localization_cp = ODLocalizationCRC(
                model=model,
                loss_function=localization_loss,
                box_extension_mode=box_extension_mode,
                **kwargs,
            )

        if classification_loss is not None:
            kwargs = dict(classification_kwargs or {})
            kwargs["assignment_method"] = assignment_method
            classification_cp = ODClassificationCRC(
                model=model,
                loss_function=classification_loss,
                pred_set_function=pred_set_function,
                **kwargs,
            )

        super().__init__(
            model=model,
            confidence_cp=confidence_cp,
            localization_cp=localization_cp,
            classification_cp=classification_cp,
            confidence_threshold=confidence_threshold,
            splitter=splitter,
        )
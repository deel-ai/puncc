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
from typing import (
    Any,
    ClassVar,
    Iterable,
    Self,
    Sequence,
    Protocol,
    runtime_checkable,
)
import warnings


from deel.puncc.core.conformal import (
    ConformalPrediction,
    ConformalPredictor,
    NoopConformalPredictor,
    _check_calibration_samples,
)
from deel.puncc.core.predictors import make_predictor
from deel.puncc.core.CRC import CRC
from deel.puncc.core.calibration import CalibrationContext
from deel.puncc.classification.prediction_sets import lac_set
from deel.puncc.core.splitters import BaseSplitter, RandomSplitter
from deel.puncc.od.losses import (
    ClassificationLoss,
    ConfidenceLoss,
    LocalizationLoss,
    ODLoss,
)
from deel.puncc.od.matching import (
    AssignmentMethod,
    AssignmentStrategy,
    HungarianMatcher,
    MatchingDirection,
)
from deel.puncc.od.utils import IndexableUserList
from deel.puncc.typing import (
    FitFunction,
    PredSetFunction,
    Predictor,
    PredictorLike,
    TensorLike,
)
from deel.puncc.od.base import (
    BoxExtensionMode,
    ODPrediction,
    ODPredictionSequence,
    ODTarget,
    ODTargetSequence,
)
from deel.puncc.backend import ops
from deel.puncc.warnings import CalibrationWarning


@runtime_checkable
class ODPredictor(
    Predictor[Sequence[tuple[TensorLike, TensorLike, TensorLike]]], Protocol
):
    """
    An Object Detection predictor is a predictor that takes as input a sequence of features and returns a
    sequence of predictions (tuples of boxes, classes scores and confidences).
    """


@runtime_checkable
class ODPredictorLike(
    PredictorLike[Sequence[tuple[TensorLike, TensorLike, TensorLike]]], Protocol
):
    """
    An object detection predictor that exposes a "predict" method that takes as input a sequence of
    features and returns a sequence of predictions (tuples of boxes, classes scores and confidences).
    """


class PunccODPredictor:
    """
    Puncc wrapper for object detection predictors.
    It ensures that the output of the underlying predictor is an ODPredictionSequence.
    """

    def __init__(self, model: ODPredictor | ODPredictorLike):
        self.model = make_predictor(model)

    def __call__(
        self, X: Iterable[Any], *args: Any, **kwargs: Any
    ) -> ODPredictionSequence:
        predictions = self.model(X, *args, **kwargs)
        if isinstance(predictions, ODPredictionSequence):
            return predictions
        return ODPredictionSequence(
            [
                ODPrediction(
                    coordinates=boxes,
                    class_scores=class_scores,
                    confidences=confidences,
                )
                for boxes, class_scores, confidences in predictions
            ]
        )

    def fit(self, X: Iterable[Any], y: ODTargetSequence, **kwargs: Any) -> Self:
        """
        Fit the underlying OD model to the given dataset.

        Args:
            X (Iterable[Any]): features
            y (ODTargetSequence): targets

        Raises:
            NotImplementedError: if not fit method has been provided by the underlying OD model or the user.

        Returns:
            Self: fitted PunccODPredictor instance
        """
        fit_method = getattr(self.model, "fit", None)

        if not callable(fit_method):
            raise NotImplementedError(
                "The underlying OD model does not have a fit method."
            )
        fit_method(X, y, **kwargs)
        return self


def make_puncc_od_predictor(model: Any) -> PunccODPredictor:
    """
    Wrapps the object betection model with the appropriate Puncc adapter

    Args:
        model (Any): OD Model

    Raises:
        ValueError:if the model does not exposes prediction method (in __call__ or predict)

    Returns:
        PunccODPredictor: Wrapped model
    """
    if isinstance(model, PunccODPredictor):
        return model
    if isinstance(model, (ODPredictor, ODPredictorLike)):
        return PunccODPredictor(model)
    raise ValueError(
        f"Model of type {type(model)} is not compatible with ODPredictor."
        "Please provide an ODModel oran ODPredictor instance."
    )


class _ODConformalPredictor(ConformalPredictor):
    prediction_attr_name: ClassVar[str]
    target_attr_name: ClassVar[str]
    prediction_set_attr_name: ClassVar[str]

    def __init__(
        self,
        model: ODPredictor | ODPredictorLike | PunccODPredictor,
        fit_function: FitFunction | None = None,
        assignment_method: AssignmentMethod | None = None,
        **kwargs: Any,
    ):
        self.assignment_method = assignment_method
        super().__init__(model=model, fit_function=fit_function, **kwargs)
        self.model = make_puncc_od_predictor(self.model)

    def fit(self, *args, **kwargs) -> Self:
        super().fit(*args, **kwargs)
        self.model = make_puncc_od_predictor(self.model)
        return self

    def project_prediction(self, prediction: ODPrediction) -> Any:
        """
        Project the prediction to the relevant attribute for conformal prediction.

        Args:
            prediction (ODPrediction): object detection prediction

        Returns:
            Any: The relevant attribute of the prediction for conformal
            prediction (e.g., class scores, coordinates, etc.)
        """
        return getattr(prediction, self.prediction_attr_name)

    def project_target(self, target: ODTarget) -> Any:
        """
        Project the target to the relevant attribute for conformal prediction.

        Args:
            target (ODTarget): object detection target

        Returns:
            Any: The relevant attribute of the target for conformal prediction (e.g., labels, coordinates, etc.)
        """
        return getattr(target, self.target_attr_name)

    def add_assignments_to_context(
        self, calibration_context: CalibrationContext
    ) -> CalibrationContext:
        """
        Add assignments to the calibration context if an assignment method is provided and
        assignments are not already present.

        Args:
            calibration_context (CalibrationContext): The calibration context containing predictions and targets.

        Returns:
            CalibrationContext: The updated calibration context with assignments added if applicable.
        """
        if (
            self.assignment_method is not None
            and "assignments" not in calibration_context
        ):
            calibration_context.update(
                assignments=IndexableUserList(
                    [
                        self.assignment_method.assign(prediction, target)
                        for prediction, target in zip(
                            calibration_context.y_pred,
                            calibration_context.y_calib,
                            strict=True,
                        )
                    ]
                )
            )
        return calibration_context

    def build_od_prediction_set(
        self, prediction: ODPredictionSequence, prediction_set: Sequence[Any]
    ) -> ODPredictionSequence:
        """
        Build an ODPredictionSequence with the prediction sets added to the relevant attribute.

        Args:
            prediction (ODPredictionSequence): The original object detection predictions.
            prediction_set (Sequence[Any]): The corresponding prediction sets to be added to the predictions.

        Returns:
            ODPredictionSequence: A new ODPredictionSequence with the prediction sets added
            to the relevant attribute (e.g., class_sets, coordinates, etc.) of each prediction.
        """
        return ODPredictionSequence(
            [
                replace(
                    prediction_i,
                    **{self.prediction_set_attr_name: prediction_set_i},
                )
                for prediction_i, prediction_set_i in zip(
                    prediction, prediction_set, strict=True
                )
            ]
        )


class ODClassificationMixin:
    """
    Mixin for object detection conformal predictors that operate on classification outputs.
    """

    prediction_attr_name: ClassVar[str] = "class_scores"
    target_attr_name: ClassVar[str] = "labels"
    prediction_set_attr_name: ClassVar[str] = "class_sets"


class ODLocalizationMixin:
    """
    Mixin for object detection conformal predictors that operate on localization outputs.
    """

    prediction_attr_name: ClassVar[str] = "coordinates"
    target_attr_name: ClassVar[str] = "coordinates"
    # prediction_set_attr_name: ClassVar[str] = "coordinates"

    def build_od_prediction_set(
        self,
        prediction: ODPredictionSequence,
        prediction_set: Sequence[Any],
    ) -> tuple[ODPredictionSequence, ODPredictionSequence]:
        """
        Build an ODPredictionSequence with the prediction sets added to the relevant attribute.

        Args:
            prediction (ODPredictionSequence): The original object detection predictions.
            prediction_set (Sequence[Any]): The corresponding prediction sets to be added to the predictions.

        Returns:
            tuple[ODPredictionSequence, ODPredictionSequence]: A tuple of two ODPredictionSequence objects,
                one for the inner prediction sets and one for the outer prediction sets,
                with the prediction sets added to the relevant attribute (e.g., coordinates) of each prediction.
        """

        inner = ODPredictionSequence(
            [
                replace(prediction_i, coordinates=prediction_set_i[..., 0])
                for prediction_i, prediction_set_i in zip(
                    prediction, prediction_set, strict=True
                )
            ]
        )

        outer = ODPredictionSequence(
            [
                replace(prediction_i, coordinates=prediction_set_i[..., 1])
                for prediction_i, prediction_set_i in zip(
                    prediction, prediction_set, strict=True
                )
            ]
        )

        return inner, outer


class BoxWiseODCP(_ODConformalPredictor):
    """
    Apply conformal calibration at the matched-box level.

    Notes:
        Finite-sample conformal coverage requires the box-level calibration scores and test score to
        satisfy the relevant exchangeability assumptions.

        Exchangeability of images alone does not imply exchangeability of all boxes pooled across images.
    """

    def __init__(
        self,
        model: ODPredictor | ODPredictorLike | PunccODPredictor,
        fit_function: FitFunction | None = None,
        assignment_method: AssignmentMethod | None = None,
        **kwargs: Any,
    ):
        if assignment_method is None:
            assignment_method = AssignmentStrategy(
                matcher=HungarianMatcher(),
                direction=MatchingDirection.PRED_TO_TRUE,
            )

        super().__init__(
            model=model,
            fit_function=fit_function,
            assignment_method=assignment_method,
            **kwargs,
        )

    def boxwise_calibration_context(
        self, calibration_context: CalibrationContext
    ) -> CalibrationContext:
        """
        Transform the calibration context to a box-wise context by aligning predictions and
        targets according to the assignments.

        Args:
            calibration_context (CalibrationContext): The original calibration context containing image-level
            predictions, targets, and assignments.

        Raises:
            ValueError: If the calibration context does not contain assignments.

        Returns:
            CalibrationContext: A new calibration context where predictions and targets are aligned at the box level,
            and the corresponding features are repeated for each box.
        """
        X_boxwise = []
        predictions = []
        targets = []

        if "assignments" not in calibration_context:
            raise ValueError("Calibration context must contain assignments.")

        for X, prediction, target, assignment in zip(
            calibration_context.X_calib,
            calibration_context.y_pred,
            calibration_context.y_calib,
            getattr(calibration_context, "assignments"),
            strict=True,
        ):
            aligned_prediction, aligned_target = assignment.align_prediction_and_target(
                prediction, target
            )

            predictions.append(aligned_prediction)
            targets.append(aligned_target)

            X_boxwise.extend([X] * len(aligned_prediction))

        return CalibrationContext(
            X_calib=IndexableUserList(X_boxwise),
            y_pred=self.boxwise_prediction(ODPredictionSequence(predictions)),
            y_calib=self.boxwise_target(ODTargetSequence(targets)),
        )

    def boxwise_prediction(self, prediction: ODPredictionSequence) -> TensorLike:
        """
        Project the box-wise predictions to the relevant attribute for conformal prediction.

        Args:
            prediction (ODPredictionSequence): The original object detection predictions.

        Returns:
            TensorLike: The relevant attribute of the box-wise predictions for conformal
            prediction (e.g., class scores, coordinates, etc.)
        """
        return self.project_prediction(prediction.boxwise())

    def boxwise_target(self, target: ODTargetSequence) -> TensorLike:
        """
        Project the box-wise targets to the relevant attribute for conformal prediction.

        Args:
            target (ODTargetSequence): The original object detection targets.

        Returns:
            TensorLike: The relevant attribute of the box-wise targets for
            conformal prediction (e.g., labels, coordinates, etc.)
        """
        return self.project_target(target.boxwise())

    def boxwise_X(
        self, X: Iterable[Any], prediction: ODPredictionSequence
    ) -> IndexableUserList:
        """
        Repeat each image feature for each predicted box in the corresponding prediction.

        Args:
            X (Iterable[Any]): The original image features.
            prediction (ODPredictionSequence): The original object detection predictions.

        Returns:
            IndexableUserList: A list of image features repeated for each predicted
                box, aligned with the box-wise predictions.
        """
        X_boxwise = []
        for x, pred in zip(X, prediction, strict=True):
            X_boxwise.extend([x] * len(pred))
        return IndexableUserList(X_boxwise)

    def compute_calibration_state(
        self, calibration_context: CalibrationContext
    ) -> CalibrationContext:
        calibration_context = self.add_assignments_to_context(calibration_context)
        calibration_context = self.boxwise_calibration_context(calibration_context)
        return super().compute_calibration_state(calibration_context)

    def conformalize(
        self,
        prediction: ODPredictionSequence,
        alpha: float | TensorLike,
        calibration_context: CalibrationContext,
        *,
        X: Iterable[Any] | None = None,
    ) -> ConformalPrediction[Any, Any]:
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
            prediction_set=self.build_od_prediction_set(prediction, prediction_sets),
        )

    def unboxwise_prediction_set(
        self, prediction_set: Any, prediction: ODPredictionSequence
    ) -> IndexableUserList:
        """
        roll back the prediction set from box-wise to image-wise by grouping the box-wise prediction sets

        Args:
            prediction_set (Any): The box-wise prediction sets obtained from the conformalization process.
            prediction (ODPredictionSequence): The original object detection predictions.

        Returns:
            IndexableUserList: A list of image-wise prediction sets, where each element
            corresponds to the prediction set for the respective image, reconstructed from the box-wise prediction sets.
        """
        prediction_sets = []
        start = 0
        for prediction_i in prediction:
            stop = start + len(prediction_i)
            prediction_sets.append(prediction_set[start:stop])
            start = stop
        return IndexableUserList(prediction_sets)


def conf_threshold_postprocessing(
    predictions: ODPredictionSequence, lambd: float
) -> ODPredictionSequence:
    """
    Base postprocessor for OD conformal predictors that filter predictions by confidence threshold.
    """
    return predictions.filter_by_confidence(1 - lambd)


class ODCRC(_ODConformalPredictor, CRC):
    """
    Base class for object detection conformal risk control (CRC) methods.
    """

    def compute_calibration_state(self, calibration_context):
        calibration_context = self.add_assignments_to_context(calibration_context)
        return super().compute_calibration_state(calibration_context)

    def _r_hat(
        self,
        lambd: float,
        calibration_context: CalibrationContext,
    ) -> float:
        predictions = self.postprocessor(calibration_context.y_pred, lambd)
        assignments = (
            getattr(calibration_context, "assignments")
            if "assignments" in calibration_context
            else None
        )
        losses = self.loss_function(
            predictions, calibration_context.y_calib, assignments
        )
        if ops.shape(losses)[0] == 0:
            raise ValueError(
                f"No losses were computed in {self.__class__.__name__}."
                "Check that the calibration datasets contains ground-truth boxes and that"
                "the postprocessordoes not filter out all predictions."
            )
        return ops.item(ops.mean(losses))

    def get_calibration_set_size(self, calibration_context: CalibrationContext) -> int:
        loss: ODLoss = self.loss_function

        if not loss.boxwise:
            return calibration_context.size

        if "y_calib" not in calibration_context:
            if calibration_context.size == 0:
                return 0
            raise ValueError("Ground-truth annotations are required for box-wise CRC.")

        if loss.penalize_unmatched_boxes:
            n = sum(len(target) for target in calibration_context.y_calib)
        else:
            if "assignments" not in calibration_context:
                raise ValueError(
                    "Assignments are required to determine the number "
                    "of evaluated ground truths when unmatched boxes "
                    "are excluded."
                )

            n = 0

            for target, assignment in zip(
                calibration_context.y_calib,
                getattr(calibration_context, "assignments"),
                strict=True,
            ):
                if assignment is None:
                    raise ValueError("Missing assignment for a calibration image.")

                if assignment.n_true != len(target):
                    raise ValueError(
                        "Assignment size does not match the number "
                        "of ground-truth boxes."
                    )

                n += len(assignment.assigned_true_indices)

        if n > 0:
            warnings.warn(
                "Box-wise CRC uses ground-truth boxes as calibration units."
                "Image-level exchangeability alone does not guarantee exchangeability of box-wise losses.",
                CalibrationWarning,
                stacklevel=2,
            )

        return n


class ODConfidenceCRC(ODCRC):
    """
    CRC for confidence step of object detection process
    """

    def __init__(
        self,
        model,
        *,
        loss_function: ODLoss,
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
            assignments = getattr(calibration_context, "assignments")
        else:
            filtered_assignments = None
            assignments = None

        for i, prediction in enumerate(calibration_context.y_pred):
            kept_indices = ops.where_1d(prediction.confidences >= threshold)
            predictions.append(prediction[kept_indices])
            if filtered_assignments is not None:
                # assignments is not None
                filtered_assignments.append(
                    assignments[i].filter_predictions(ops.tolist(kept_indices))  # type: ignore
                )

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
    """
    CRC for classification step of object detection process
    """

    @staticmethod
    def get_postprocessor(pred_set_function: PredSetFunction):
        """
        Create a postprocessor function that applies the given prediction set function to the class scores of each
        prediction.

        Args:
            pred_set_function (PredSetFunction): A function that takes class scores and a threshold (lambda) and
            returns the corresponding prediction sets.
        """

        def postprocessor(
            predictions: ODPredictionSequence, lambd: float
        ) -> ODPredictionSequence:
            return ODPredictionSequence(
                [
                    replace(
                        prediction,
                        class_sets=pred_set_function(prediction.class_scores, lambd),
                    )
                    for prediction in predictions
                ]
            )

        return postprocessor

    def __init__(
        self,
        model: ODPredictor | ODPredictorLike | PunccODPredictor,
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
    """
    CRC for localization step of object detection process
    """

    @staticmethod
    def get_postprocessor(box_extension_mode: BoxExtensionMode):
        """
        Create a postprocessor function that applies the given box extension mode to the coordinates of each

        Args:
            box_extension_mode (BoxExtensionMode): additive or multiplicative
        """

        def postprocessor(
            predictions: ODPredictionSequence,
            lambd: float,
        ) -> ODPredictionSequence:
            return ODPredictionSequence(
                [
                    replace(
                        prediction,
                        coordinates=box_extension_mode.processor(
                            prediction.coordinates, lambd
                        ),
                    )
                    for prediction in predictions
                ]
            )

        return postprocessor

    def __init__(
        self,
        model: ODPredictor | ODPredictorLike | PunccODPredictor,
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
            **kwargs,
        )


class NoopODConformalPredictor(_ODConformalPredictor, NoopConformalPredictor):
    """
    A no-op conformal predictor for object detection that does not perform any conformalization.
    """


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
        model: ODPredictor | ODPredictorLike | PunccODPredictor,
        *,
        confidence_cp: _ODConformalPredictor | None = None,
        localization_cp: _ODConformalPredictor | None = None,
        classification_cp: _ODConformalPredictor | None = None,
        confidence_threshold: float = 0.0,
        splitter: BaseSplitter | None = None,
    ):
        self.model = make_puncc_od_predictor(model)

        if (
            confidence_cp is None
            and localization_cp is None
            and classification_cp is None
        ):
            raise ValueError(
                "At least one of confidence_cp, localization_cp or classification_cp must be provided."
            )

        self.confidence_cp = (
            confidence_cp
            if confidence_cp is not None
            else NoopODConformalPredictor(model)
        )
        self.localization_cp = (
            localization_cp
            if localization_cp is not None
            else NoopODConformalPredictor(model)
        )
        self.classification_cp = (
            classification_cp
            if classification_cp is not None
            else NoopODConformalPredictor(model)
        )

        self.confidence_threshold = confidence_threshold
        self.splitter = splitter if splitter is not None else RandomSplitter(ratio=0.5)
        self.reset_calibration()

    def reset_calibration(self) -> Self:
        """
        Reset the calibration state of the TripleConformalPredictor.

        Returns:
            Self: The TripleConformalPredictor instance with its calibration state reset.
        """
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
        return ODPredictionSequence(
            [
                replace(pred, class_sets=class_pred.class_sets)
                for pred, class_pred in zip(prediction, classification, strict=True)
            ]
        )

    @classmethod
    def _merge_prediction_sets(
        cls, localization: Any, classification: ODPredictionSequence
    ) -> Any:
        if isinstance(localization, ODPredictionSequence):
            return cls._merge_classification(localization, classification)

        if (
            isinstance(localization, tuple)
            and len(localization) == 2
            and all(
                isinstance(prediction, ODPredictionSequence)
                for prediction in localization
            )
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
        """
        Calibrate the TripleConformalPredictor using the provided calibration dataset.

        Args:
            X_calib (Iterable[Any]): The calibration features.
            y_calib (ODTargetSequence): The calibration targets.

        Returns:
            Self: The calibrated TripleConformalPredictor instance.
        """
        _check_calibration_samples(X_calib, y_calib, require_y=True)
        self.reset_calibration()
        prediction = self.model(X_calib).filter_by_confidence(self.confidence_threshold)
        context = CalibrationContext(
            X_calib=X_calib, y_calib=y_calib, y_pred=prediction
        )

        # if no conf calibration phase : all calibration goes to localization and classification
        if isinstance(self.confidence_cp, NoopODConformalPredictor):
            self._localization_context = self.localization_cp.compute_calibration_state(
                context.copy()
            )
            self._classification_context = (
                self.classification_cp.compute_calibration_state(context.copy())
            )
        # if no localization and classification calibration phase : all calibration goes to confidence
        elif isinstance(
            self.classification_cp, NoopODConformalPredictor
        ) and isinstance(self.localization_cp, NoopODConformalPredictor):
            self._confidence_context = self.confidence_cp.compute_calibration_state(
                context.copy()
            )
        else:
            # otherwise : split context in two parts
            confidence_context, self._downstream_context = self.splitter.split_context(
                context
            )[0]
            self._confidence_context = self.confidence_cp.compute_calibration_state(
                confidence_context
            )
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
        """
        Conformalize the given object detection predictions using the calibrated TripleConformalPredictor.

        Args:
            prediction (ODPredictionSequence): The object detection predictions to be conformalized.
            alpha_conf (float | TensorLike | None, optional): The coverage level for the confidence step
                of the conformal prediction. Defaults to None.
            alpha_loc (float | TensorLike | None, optional): The coverage level for the localization step
                of the conformal prediction. Defaults to None.
            alpha_class (float | TensorLike | None, optional):The coverage level for the classification step
                of the conformal prediction. Defaults to None.
            X (Iterable[Any] | None, optional): The input features corresponding to the predictions.
                Defaults to None.

        Raises:
            RuntimeError: _description_
            ValueError: _description_

        Returns:
            ConformalPrediction[ODPredictionSequence, Any]: _description_
        """
        if self._calibrated is False:
            raise RuntimeError(
                "The TripleConformalPredictor must be calibrated before conformalization."
            )
        # check alphas :
        for alpha, cp in zip(
            [alpha_conf, alpha_loc, alpha_class],
            [self.confidence_cp, self.localization_cp, self.classification_cp],
        ):
            if alpha is None and not isinstance(cp, NoopODConformalPredictor):
                raise ValueError(
                    f"Alpha value is required for {cp.__class__.__name__}."
                )

        alpha_conf = -1 if alpha_conf is None else alpha_conf
        alpha_loc = -1 if alpha_loc is None else alpha_loc
        alpha_class = -1 if alpha_class is None else alpha_class

        thresholded_prediction = prediction.filter_by_confidence(
            self.confidence_threshold
        )
        confidence_prediction = self.confidence_cp.conformalize(
            thresholded_prediction, alpha_conf, self._confidence_context, X=X
        ).prediction_set

        if isinstance(self.confidence_cp, NoopODConformalPredictor) or (
            isinstance(self.localization_cp, NoopODConformalPredictor)
            and isinstance(self.classification_cp, NoopODConformalPredictor)
        ):
            localization_context = self._localization_context
            classification_context = self._classification_context
        else:
            filtered_prediction = self.confidence_cp.conformalize(
                self._downstream_context.y_pred,
                alpha_conf,
                self._confidence_context,
                X=self._downstream_context.X_calib,
            ).prediction_set
            context = self._downstream_context.copy().update(y_pred=filtered_prediction)
            localization_context = self.localization_cp.compute_calibration_state(
                context.copy()
            )
            classification_context = self.classification_cp.compute_calibration_state(
                context.copy()
            )

        localization = self.localization_cp.conformalize(
            confidence_prediction, alpha_loc, localization_context, X=X
        ).prediction_set
        classification = self.classification_cp.conformalize(
            confidence_prediction, alpha_class, classification_context, X=X
        ).prediction_set

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
        """
        Make predictions with conformal prediction sets for object detection.

        Args:
            X (Iterable[Any]): The input features for which predictions are to be made.
            alpha_conf (float | TensorLike | None, optional): The coverage level for the confidence
                step of the conformal prediction. Defaults to None.
            alpha_loc (float | TensorLike | None, optional): The coverage level for the localization step
                of the conformal prediction. Defaults to None.
            alpha_class (float | TensorLike | None, optional): The coverage level for the classification
                step of the conformal prediction.

        Returns:
            ConformalPrediction[ODPredictionSequence, Any]: A ConformalPrediction object containing
                the predictions and the corresponding prediction sets for object detection.
        """
        prediction = self.model(X)

        return self.conformalize(
            prediction,
            alpha_conf=alpha_conf,
            alpha_loc=alpha_loc,
            alpha_class=alpha_class,
            X=X,
        )


class TripleCRC(TripleConformalPredictor):
    """
    Triple Conformal Risk Control (CRC) for object detection, orchestrating
    confidence, localization, and classification CRCs.
    """

    def __init__(
        self,
        model: ODPredictor | ODPredictorLike | PunccODPredictor,
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
        _model = make_puncc_od_predictor(model)

        confidence_cp = None
        localization_cp = None
        classification_cp = None

        if confidence_loss is not None:
            kwargs = dict(confidence_kwargs or {})
            kwargs["assignment_method"] = assignment_method
            confidence_cp = ODConfidenceCRC(
                model=_model,
                loss_function=confidence_loss,
                **kwargs,
            )

        if localization_loss is not None:
            kwargs = dict(localization_kwargs or {})
            kwargs["assignment_method"] = assignment_method
            localization_cp = ODLocalizationCRC(
                model=_model,
                loss_function=localization_loss,
                box_extension_mode=box_extension_mode,
                **kwargs,
            )

        if classification_loss is not None:
            kwargs = dict(classification_kwargs or {})
            kwargs["assignment_method"] = assignment_method
            classification_cp = ODClassificationCRC(
                model=_model,
                loss_function=classification_loss,
                pred_set_function=pred_set_function,
                **kwargs,
            )

        super().__init__(
            model=_model,
            confidence_cp=confidence_cp,
            localization_cp=localization_cp,
            classification_cp=classification_cp,
            confidence_threshold=confidence_threshold,
            splitter=splitter,
        )

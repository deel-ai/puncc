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
Base abstractions and common utilities for conformal prediction methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Sequence,
)
from dataclasses import dataclass
from typing import Any, Generic, TypeAlias, TypeVar, Self

from deel.puncc.core.calibration import CalibrationContext
from deel.puncc.core.splitters import FunctionalSplitter
from deel.puncc.core.predictors import make_predictor
from deel.puncc.backend.keras import ops

from deel.puncc.typing import (
    FitFunction,
    Predictor,
    PredictorLike,
    TensorLike,
)

TPrediction = TypeVar("TPrediction")
TSet = TypeVar("TSet")

AlphaCacheKey: TypeAlias = float | tuple[tuple[int, ...], str, bytes]

CacheKey: TypeAlias = tuple[
    CalibrationContext,
    AlphaCacheKey,
]

def alpha_cache_key(
    alpha: float|TensorLike,
) -> AlphaCacheKey:
    """Convert alpha into a stable, hashable, value-based cache key."""
    if isinstance(alpha, (int, float)):
        return float(alpha)

    array = ops.convert_to_numpy(alpha)

    if array.ndim == 0:
        return float(array.item())

    return (
        tuple(array.shape),
        str(array.dtype),
        array.tobytes(),
    )



@dataclass(frozen=True, slots=True)
class ConformalPrediction(Generic[TPrediction, TSet]):
    """
    Result of a conformal prediction.

    The container behaves like a two-element tuple while preserving explicit names for the base prediction and its associated conformal prediction set.

    Attributes:
        prediction (Any):
            The base (non-conformal) prediction produced by the underlying model.

        prediction_set (Any):
            The conformal prediction set associated with the prediction.
            Its representation depends on the conformal method and may for example
            be an interval tensor, a collection of labels, or structured detection outputs.
    """

    prediction: TPrediction
    prediction_set: TSet

    def __iter__(self) -> Iterator[TPrediction | TSet]:
        yield self.prediction
        yield self.prediction_set

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index:int|slice)->TPrediction | TSet | tuple[TPrediction | TSet, ...]:
        return (self.prediction, self.prediction_set)[index]

class ConformalPredictor(ABC):
    """
    Abstract base class for conformal prediction methods.

    Subclasses should implement two method-specific operations:

    - compute_calibration_state derives the state required for conformal
      prediction from a calibration context.
    - conformalize turns already-computed model predictions into conformal
      prediction sets using a calibration context.

    Calibration-dependent state should be stored in the provided CalibrationContext rather than in predictor attributes.
    This makes calibration contexts explicit and allows the same predictor to conformalize predictions against different calibration subsets.

    Args:
        model:
            Predictive model used to generate base predictions.

        fit_function:
            Optional custom function used to fit model.
            If omitted, model.fit is used when available.
    """
    __slots__ = ("model", "fit_function", "calibration_context", "conformalization_cache")

    # Any conformal method should have a model attribute.
    def __init__(self, model:Predictor|PredictorLike,
                 fit_function:FitFunction|None = None):
        self.model = make_predictor(model)
        self.fit_function = fit_function
        self.calibration_context = CalibrationContext()
        self.conformalization_cache:dict[CacheKey, Any] = {}

    def _make_cache_key(
        self,
        alpha: float|TensorLike,
        calibration_context: CalibrationContext,
    ) -> CacheKey:
        return (
            calibration_context,
            alpha_cache_key(alpha),
        )

    def calibrate(self, X_calib:Iterable[Any], y_calib:Iterable[Any])->Self:
        """
        Calibrate the conformal predictor.

        The underlying model is evaluated once on the calibration inputs.
        The resulting raw calibration data are stored in a CalibrationContext before method-specific calibration state is computed by compute_calibration_state.

        Args:
            X_calib:
                Calibration inputs.

            y_calib:
                Calibration targets.

        Returns:
            The calibrated predictor.
        """
        self.conformalization_cache.clear()
        
        self.calibration_context.clear()
        self.calibration_context.update(
            X_calib=X_calib,
            y_calib=y_calib,
            y_pred = self.model(X_calib)
        )
        self.calibration_context = self.compute_calibration_state(self.calibration_context)
        return self

    @abstractmethod
    def compute_calibration_state(self, calibration_context:CalibrationContext)->CalibrationContext:
        """
        Compute method-specific calibration state.

        The provided context contains at least the calibration inputs, targets, and corresponding model predictions.
        Implementations may enrich it with additional sample-aligned quantities such as non-conformity scores.

        Calibration-dependent information should be stored in the context whenever possible so that different contexts can be used independently.

        Args:
            calibration_context:
                Context containing the raw calibration data.

        Returns:
            The context containing the state required by conformalize.
        """

    def predict(self, X_test:Iterable[Any],
                alpha:float|TensorLike,
                *,
                alpha_correction: Callable[[float|TensorLike], float|TensorLike] | None = None,)->ConformalPrediction[Any, Any]:
        """
        Perform a conformal prediction using the calibrated model.


        Args:
            X_test:
                Inputs on which predictions are produced.

            alpha:
                Requested miscoverage level. It may be scalar or tensor-valued.

            alpha_correction:
                Optional transformation applied to alpha before conformalization.

        Returns:
            Base predictions and their associated conformal prediction sets.
        """
        prediction = self.model(X_test)

        if alpha_correction is not None:
            alpha = alpha_correction(alpha)

        return self.conformalize(prediction, alpha, self.calibration_context)

    @abstractmethod
    def conformalize(self, prediction:Any, alpha:float|TensorLike, calibration_context:CalibrationContext)->ConformalPrediction[Any, Any]:
        """
        Conformalize already-computed model predictions.

        This method must not evaluate the predictive model.
        All calibration-dependent information must come from the explicitly provided calibration context.

        Args:
            prediction:
                Base predictions to conformalize.

            alpha:
                Miscoverage level used for conformalization.

            calibration_context:
                Calibration context against which predictions are conformalized.

        Returns:
            Base predictions and their associated conformal prediction sets.
        """

    def fit(self,
            X:Iterable[Any],
            y:Iterable[Any],
            *args:Any, 
            **kwargs:Any
            )->Self:
        """
        Fit the underlying predictive model.

        The custom fit_function is used when provided. Otherwise the predictor's own fit method is called.

        Args:
            X:
                Training inputs.

            y:
                Training targets.

            *args:
                Additional positional arguments forwarded to the fitting function.

            **kwargs:
                Additional keyword arguments forwarded to the fitting function.

        Returns:
            The predictor with its underlying model fitted.
        """
        if self.fit_function is not None:
            self.model = self.fit_function(self.model, X, y, *args, **kwargs)
            return self
        
        fit_method = getattr(
            self.model,
            "fit",
            None,
        )
        if callable(fit_method):
            fit_method(
                X,
                y,
                *args,
                **kwargs,
            )
        raise NotImplementedError("The model does not have a fit method and no fit_function was provided. Please provide a pretrained model or a fit_function.")

class GroupConditionalMixin(ConformalPredictor):
    """
    Mixin adding group-balanced calibration to a conformal predictor.

    Calibration data are partitioned according to group_function and a separate calibration context is computed for each observed group.
    During prediction, test samples are routed to the calibration context associated with their group before delegating conformalization to the next ConformalPredictor implementation in the MRO.

    A global calibration context is also retained and currently acts as a fallback for test groups that were not observed during calibration.

    Args:
        group_function:
            Function assigning a group identifier to each sample.
            It must be computable at prediction time without access to unknown test targets.

        groups:
            Optional sequence defining the groups considered by the splitter.

        *args:
            Positional arguments forwarded to the concrete conformal predictor.

        **kwargs:
            Keyword arguments forwarded to the concrete conformal predictor.
    """
    def __init__(
        self,
        *args:Any,
        group_function: Callable[..., TensorLike],
        groups:Sequence[Any]|None=None,
        **kwargs:Any,
    ):
        super().__init__(*args, **kwargs)
        self.group_splitter = FunctionalSplitter(group_function=group_function, groups=groups)
        self.group_calibration_contexts:dict[Any, CalibrationContext] = {}

    def calibrate(
        self,
        X_calib: Iterable[Any],
        y_calib: Iterable[Any],
    ) -> Self:
        self.conformalization_cache.clear()

        raw_context = CalibrationContext(
            X_calib=X_calib,
            y_calib=y_calib,
            y_pred=self.model(X_calib),
        )

        raw_group_contexts = (self.group_splitter.split_context_by_group(raw_context))


        compute_calib_state = super().compute_calibration_state

        self.group_calibration_contexts = {
            group: compute_calib_state(context)
            for group, context
            in raw_group_contexts.items()
            if context.size > 0
        }

        # Fallback calibration context
        self.calibration_context = (compute_calib_state(raw_context))
        return self

    def predict(
        self,
        X_test:Iterable[Any],
        alpha:float|TensorLike,
        *,
        alpha_correction: Callable[[float|TensorLike], float|TensorLike] | None = None,
    ) -> ConformalPrediction[Any, Any]:
        prediction = self.model(X_test)

        if alpha_correction is not None:
            alpha = alpha_correction(alpha)

        grouped_indices = self.group_splitter.group_indices(X_calib=X_test, y_pred=prediction)

        grouped_prediction_sets: list[Any] = []
        non_empty_indices: list[TensorLike] = []

        for group, indices in grouped_indices.items():
            if len(indices) == 0:
                continue

            # TODO: validate the theoretical guarantees of this fallback.
            # Until then, groups unseen during calibration use the global calibration context.
            group_context = self.group_calibration_contexts.get(group, self.calibration_context)

            group_prediction = ops.take(
                prediction,
                indices,
                axis=0,
            )

            group_result = super().conformalize(
                group_prediction,
                alpha,
                group_context,
            )

            grouped_prediction_sets.append(
                group_result.prediction_set
            )
            non_empty_indices.append(indices)

        first_prediction_set = grouped_prediction_sets[0]

        # Preserve tensor structure
        if isinstance(first_prediction_set, ops.tensor_type,):
            prediction_set = ops.concatenate(grouped_prediction_sets, axis=0,)
            concatenated_indices = ops.concatenate(non_empty_indices, axis=0)
            original_order = ops.argsort(concatenated_indices, axis=0)
            prediction_set = ops.take(prediction_set, original_order, axis=0)
            return ConformalPrediction(
                prediction=prediction,
                prediction_set=prediction_set,
            )

        # Ragged prediction sets cannot generally be represented as a tensor.
        # Reassemble them directly in Python while preserving sample order.
        prediction_set_items: list[Any] = [None] * len(prediction)

        for indices, group_prediction_set in zip(non_empty_indices, grouped_prediction_sets, strict=True):
            original_indices = (ops.convert_to_numpy(indices).tolist())
            for local_idx, original_idx in enumerate(original_indices):
                prediction_set_items[original_idx] = (group_prediction_set[local_idx])

        return ConformalPrediction(
            prediction=prediction,
            prediction_set=prediction_set_items,
        )

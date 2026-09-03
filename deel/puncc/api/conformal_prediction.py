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
Abstract base classes for conformal prediction methods
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, overload
from typing_extensions import Self
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence, Callable
from deel.puncc.api.calibration_context import CalibrationContext
from deel.puncc.api.splitting import FunctionalSplitter
from deel.puncc.typing import FitFunction, Predictor, PredictorLike, TensorLike, make_predictor
from deel.puncc.keras import ops

# To be defined more precisely in the future.
TPrediction = TypeVar("TPrediction")
TSet = TypeVar("TSet")

@dataclass(frozen=True, slots=True)
class ConformalPrediction(Generic[TPrediction, TSet]):
    """
    Container for a conformal prediction result (juste a bit more than tuple (prediction, conformal_prediction)).

    Attributes:
        prediction (TensorLike):
            The base (non-conformal) prediction of the underlying model.

        prediction_set (TSet):
            The conformal prediction set (interval, label set, bounding boxes, etc.).
    """

    prediction: TPrediction
    prediction_set: TSet

    def __iter__(self) -> Iterator[TPrediction | TSet]:
        yield self.prediction
        yield self.prediction_set

    def __len__(self) -> int:
        return 2

    @overload
    def __getitem__(self, index: int) -> TPrediction | TSet: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TPrediction | TSet, ...]: ...

    def __getitem__(self, index:int|slice):
        if isinstance(index, slice):
            return (self.prediction, self.prediction_set)[index]
        if index == 0:
            return self.prediction
        if index == 1:
            return self.prediction_set
        raise IndexError("ConformalPrediction only contains two elements.")

class ConformalPredictor(ABC):
    """
    Abstract base class for conformal prediction methods.
    Any conformal prediction method should inherit from this class and implement the `calibrate` and `predict` methods.
    """
    __slots__ = ("model", "fit_function", "calibration_context")

    # Any conformal method should have a model attribute.
    def __init__(self, model:Predictor|PredictorLike,
                 fit_function:FitFunction|None = None):
        self.model = make_predictor(model)
        self.fit_function = fit_function
        self.calibration_context = CalibrationContext()

    def calibrate(self, X_calib:Iterable[Any], y_calib:Iterable[Any])->Self:
        """
        Calibration step of the conformal method.
        This method may be overloaded by subclasses to implement specific pré-calibration procedures.
       
        Args:
            X_calib (Iterable[Any]): Features of calibration dataset
            y_calib (TensorLike): Labels of calibration dataset
        """
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
        ...

    def predict(self, X_test:Iterable[Any],
                alpha:float|TensorLike,
                *,
                alpha_correction: Callable[[float|TensorLike], float|TensorLike] | None = None,)->ConformalPrediction[Any, Any]:
        """
        Perform a conformal prediction using the calibrated model.

        Args:
            X_test (Iterable[Any]): Features to perform the conformal prediction on
            alpha (float | TensorLike): Miscoverage level(s) for the conformal prediction. Can be a single float or a tensor of floats of the same length as X_test.

        Returns:
            ConformalPrediction: A container for the conformal prediction result, containing the base (non-conformal) prediction and the conformal prediction set.
        """
        prediction = self.model(X_test)

        if alpha_correction is not None:
            alpha = alpha_correction(alpha)

        return self.conformalize(prediction, alpha, self.calibration_context)

    @abstractmethod
    def conformalize(self, prediction:Any, alpha:float, calibration_context:CalibrationContext)->ConformalPrediction[Any, Any]:
        ...

    def fit(self,
            X:Iterable[Any],
            y:Iterable[Any],
            *args:Any, 
            **kwargs:Any
            ):
        if self.fit_function is not None:
            self.model = self.fit_function(self.model, X, y, *args, **kwargs)
        elif callable(getattr(self.model, "fit", None)):
            self.model.fit(X, y, *args, **kwargs) # type: ignore
        else:
            raise NotImplementedError("The model does not have a fit method and no fit_function was provided. Please provide a pretrained model or a fit_function.")
        return self

class GroupBalancedMixin(ConformalPredictor):
    def __init__(
        self,
        *args:Any,
        group_function: Callable[..., TensorLike],
        groups:Sequence[Any]|None=None,
        **kwargs:Any,
    ):
        super().__init__(*args, **kwargs)
        self.group_splitter = FunctionalSplitter(group_function=group_function, groups=groups)
        self.group_calibration_contexts = {}

    def calibrate(
        self,
        X_calib,
        y_calib,
    ) -> Self:
        super().calibrate(
            X_calib,
            y_calib,
        )

        self.group_calibration_contexts = (
            self.group_splitter
            .split_context_by_group(
                self.calibration_context
            )
        )

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

        grouped_indices = (
            self.group_splitter.group_indices(
                X_calib=X_test,
                y_pred=prediction,
            )
        )
        prediction_sets = [
            None
            for _ in range(
                int(ops.shape(prediction)[0])
            )
        ]
        for group, indices in grouped_indices.items():
            group_context = (
                self.group_calibration_contexts[group]
            )

            group_result = super().conformalize(
                prediction[indices],
                alpha,
                group_context,
            )

            original_indices = (
                ops.convert_to_numpy(indices)
                .tolist()
            )

            for local_idx, original_idx in enumerate(
                original_indices
            ):
                prediction_sets[original_idx] = (
                    group_result.prediction_set[
                        local_idx
                    ]
                )

        return ConformalPrediction(
            prediction=prediction,
            prediction_set=prediction_sets,
        )
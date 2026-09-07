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
This module proposes implementation of Conformal Risk Control method as described in [paper]
"""
from __future__ import annotations
from typing import Any, Callable, Generic, TypeAlias, TypeVar
from collections.abc import Iterable

from deel.puncc.core.calibration import CalibrationContext
from deel.puncc.core.conformal import ConformalPrediction, ConformalPredictor
from deel.puncc.typing import FitFunction, Predictor, PredictorLike, TensorLike
from deel.puncc.optimization import ScalarOptimizer, BinarySearchOptimizer
from deel.puncc.backend.keras import ops

TPrediction = TypeVar("TPrediction")
TTarget = TypeVar("TTarget")
TConformalPrediction = TypeVar(
    "TConformalPrediction"
)

Postprocessor: TypeAlias = Callable[
    [TPrediction, float],
    TConformalPrediction,
]

RiskLossFunction: TypeAlias = Callable[
    [
        TConformalPrediction,
        TTarget,
    ],
    TensorLike,
]

class CRC(ConformalPredictor, Generic[TPrediction, TTarget, TConformalPrediction]):
    __slots__ = ("loss_function", "postprocessor", "B", "optimizer", "lambda_bounds", "search_tol", "max_iter")
    def __init__(self, model:Predictor|PredictorLike,
                postprocessor: Callable[[TPrediction, float], TConformalPrediction],
                loss_function: Callable[[TConformalPrediction, TTarget], TensorLike],
                *,
                 loss_function_upper_bound:float|None = None,
                 fit_function:FitFunction|None = None,
                 optimizer:ScalarOptimizer|None = None,
                 lambda_bounds:tuple[float, float]=(0.0, 1.0),
                 search_tol:float=1e-4,
                 max_iter:int=25):
        super().__init__(model, fit_function=fit_function)
        self.postprocessor = postprocessor
        self.loss_function = loss_function
        self.B = loss_function_upper_bound if loss_function_upper_bound is not None else getattr(loss_function, "upper_bound", 1.0)

        self.optimizer = optimizer if optimizer is not None else BinarySearchOptimizer()
        self.lambda_bounds = lambda_bounds
        self.search_tol = search_tol
        self.max_iter = max_iter

    @property
    def len_calib(self)->int:
        return self.calibration_context.size

    def _r_hat(
        self,
        lambd: float,
        calibration_context: CalibrationContext,
    ) -> float:
        losses = self.loss_function(
            self.postprocessor(
                calibration_context.y_pred,
                lambd,
            ),
            calibration_context.y_calib,
        )

        return float(
            ops.convert_to_numpy(
                ops.mean(losses)
            )
        )


    def compute_calibration_state(
        self,
        calibration_context: CalibrationContext,
    ) -> CalibrationContext:
        return calibration_context

    def _get_lambda_from_alpha(
        self,
        alpha: float|TensorLike,
        calibration_context: CalibrationContext,
    ) -> float:
        if alpha >= self.B:
            raise ValueError(
                f"alpha must be smaller than the loss upper bound B={self.B}."
            )

        key = self._make_cache_key(alpha, calibration_context)
        
        if key not in self.conformalization_cache:
            n = calibration_context.size

            def _lambda_loss(
                lambd: float,
            ) -> float:
                return (
                    n / (n + 1)
                    * self._r_hat(
                        lambd,
                        calibration_context,
                    )
                    + self.B / (n + 1)
                    - alpha
                )

            try:
                lambda_hat = self.optimizer(
                    _lambda_loss,
                    *self.lambda_bounds,
                    xtol=self.search_tol,
                    maxiter=self.max_iter,
                )

            except ValueError as e:
                raise ValueError("Could not find a valid lambda for the given alpha. "
                                "This may be due to the loss function upper bound being too low "
                                "or the calibration set not being representative enough.") from e

            self.conformalization_cache[key] = lambda_hat

        return self.conformalization_cache[key]

    def conformalize(
        self,
        prediction: Any,
        alpha: float,
        calibration_context: CalibrationContext,
    ) -> ConformalPrediction[Any, Any]:
        lambda_hat = self._get_lambda_from_alpha(
            alpha,
            calibration_context,
        )

        conformal_prediction = self.postprocessor(
            prediction,
            lambda_hat,
        )

        return ConformalPrediction(
            prediction,
            conformal_prediction,
        )
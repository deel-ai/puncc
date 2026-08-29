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
from typing_extensions import Self
from typing import Any, Callable
from collections.abc import Iterable
from deel.puncc.api.calibration_context import CalibrationContext
from deel.puncc.api.conformalization import ConformalMethod
from deel.puncc.typing import TensorLike, Predictor, PredictorLike
from deel.puncc.optimization import ScalarOptimizer, BinarySearchOptimizer


class CRC(ConformalMethod):
    __slots__ = ("loss_function", "postprocessor", "B", "optimizer", "lambda_bounds", "search_tol", "max_iter", "_lambda_cache")
    def __init__(self, model:Predictor|PredictorLike,
                 postprocessor:Callable[[Iterable, float], Iterable],
                 loss_function:Callable[[Iterable, Iterable], Iterable[float]],
                 loss_function_upper_bound:float=1,
                 optimizer:ScalarOptimizer=BinarySearchOptimizer(),
                 lambda_bounds:tuple[float, float]=(0.0, 1.0),
                 search_tol:float=1e-4,
                 max_iter:int=25):
        super().__init__(model)
        self.postprocessor = postprocessor
        self.loss_function = loss_function
        self.B = loss_function_upper_bound or getattr(loss_function, "upper_bound", 1.0)
        self.optimizer = optimizer
        self.lambda_bounds = lambda_bounds
        self.search_tol = search_tol
        self.max_iter = max_iter

        self._lambda_cache = {}

    @property
    def len_calib(self):
        return len(self.calibration_context)
    

    def _r_hat(self, lambd):
        if len(self.calibration_context) == 0:
            raise ValueError("The model must be calibrated before computing r_hat.")
        return sum(self.loss_function(self.postprocessor(self.calibration_context.y_pred, lambd), self.calibration_context.y_calib)) / self.len_calib

    def compute_calibration_state(
        self,
        calibration_context: CalibrationContext,
    ) -> CalibrationContext:
        self._lambda_cache.clear()
        return calibration_context

    def _get_lambda_from_alpha(self, alpha:float)->float:
        if alpha >= self.B:
            raise ValueError(
                f"alpha must be smaller than the loss upper bound B={self.B}."
            )
        if alpha not in self._lambda_cache:
            n = self.len_calib
            def _lambda_loss(lambda_:float)->float:
                return n/(n+1) * self._r_hat(lambda_) + self.B / (n + 1) - alpha
            try:
                lambda_hat = self.optimizer(_lambda_loss, *self.lambda_bounds, xtol=self.search_tol, maxiter=self.max_iter) 
            except ValueError as e:
                raise ValueError("Could not find a valid lambda for the given alpha. "
                                "This may be due to the loss function upper bound being too low "
                                "or the calibration set not being representative enough.") from e
            self._lambda_cache[alpha] = lambda_hat
        return self._lambda_cache[alpha]

    def predict(self, X_test:Iterable[Any], alpha:float|TensorLike):
        if not isinstance(alpha, float):
            raise NotImplementedError("Vectorized alpha is not implemented yet.")
        if len(self) == 0:
            raise ValueError("The model must be calibrated before prediction.")
        lambda_hat = self._get_lambda_from_alpha(alpha)
        c_lambda_pred = self.model(X_test, lambda_hat)
        return c_lambda_pred

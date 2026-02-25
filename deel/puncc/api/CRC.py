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
from functools import lru_cache
from typing import Any, Callable
from collections.abc import Iterable
from deel.puncc.api.conformalization import ConformalMethod
from deel.puncc.typing import TensorLike, LambdaPredictor
from scipy.optimize import brentq


class CRC(ConformalMethod):
    def __init__(self, model:LambdaPredictor,
                 loss_function:Callable[[Iterable, Iterable], Iterable[float]],
                 loss_function_upper_bound:int=1,
                 root_finder:Callable=brentq):
        super().__init__(model)
        self.loss_function = loss_function
        self.B = loss_function_upper_bound
        self.root_finder = root_finder

        self._x_calib = []
        self._y_calib = []

    @property
    def len_calib(self):
        return len(self._y_calib)
    
    def is_calibrated(self)->bool:
        return self.len_calib > 0

    def _r_hat(self, lambd):
        if not self.is_calibrated():
            raise ValueError("The model must be calibrated before computing r_hat.")
        return sum(self.loss_function(self.model(self._x_calib, lambd), self._y_calib)) / self.len_calib

    def calibrate(self, X_calib:Iterable[Any], y_calib:TensorLike)->Self:
        self._x_calib = X_calib
        self._y_calib = y_calib
        return self

    # TODO : use a better cache that functools cache to avoid storing self references
    @lru_cache(maxsize=8)
    def __get_lambda_from_alpha(self, alpha:float)->float:
        n = self.len_calib
        def _lambda_loss(lambda_:float)->float:
            return n/(n+1) * self._r_hat(lambda_) + self.B / (n + 1) - alpha
        try:
            # TODO : define lambda search space
            # TODO : allow user to give its hyper parameters for the RF algorithm
            lambda_hat = self.root_finder(_lambda_loss, 0.0, 1.0, xtol=1e-4, maxiter=20) 
        except ValueError as e:
            raise ValueError("Could not find a valid lambda for the given alpha. "
                             "This may be due to the loss function upper bound being too low "
                             "or the calibration set not being representative enough.") from e
        return lambda_hat

    def predict(self, X_test:Iterable[Any], alpha:float|TensorLike):
        if not isinstance(alpha, float):
            raise NotImplementedError("Vectorized alpha is not implemented yet.")
        if not self.is_calibrated():
            raise ValueError("The model must be calibrated before prediction.")
        lambda_hat = self.__get_lambda_from_alpha(alpha)
        c_lambda_pred = self.model(X_test, lambda_hat)
        return c_lambda_pred

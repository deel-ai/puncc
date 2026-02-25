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
This module gives basic tools for Cross Conformal Prediction
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Any, Callable
from collections.abc import Iterable
from typing_extensions import Self
from collections.abc import Sequence
from deel.puncc.api.conformal_predictor import ConformalPredictor
from deel.puncc.api.conformalization import ConformalPrediction
from deel.puncc.typing import Predictor, PredictorLike, TensorLike
from deel.puncc.api.splitting import KFoldSplitter, BaseSplitter
from deel.puncc.cloning import clone_model
from deel.puncc.regression import SplitCP
from deel.puncc import ops

class CrossConformalPredictor(ConformalPredictor):
    def __init__(self,
                 model:Predictor|PredictorLike,
                 conformal_predictor_class:Callable[..., ConformalPredictor],
                 splitter:BaseSplitter,
                 random_state:int|None=None,
                 weight_function:Callable[[Iterable[Any]], Iterable[float]]|None = None,
                 fit_function:Callable[[Predictor, Iterable[Any], TensorLike], Predictor]|None = None):
        super().__init__(model, None, None, weight_function, fit_function)
        self.splitter = splitter
        self.random_state = random_state
        self._conformal_predictors = []
        self.conformal_predictor_class = conformal_predictor_class

    @property
    def len_calibr(self)->int:
        return sum(cp.len_calibr for cp in self._conformal_predictors)
    
    @property
    def nc_scores(self) -> Sequence[float]:
        all_scores = []
        for n, cp in enumerate(self._conformal_predictors):
            try:
                all_scores.extend(cp.nc_scores)
            except RuntimeError as e:
                raise RuntimeError(f"The conformal predictor number {n} has not been calibrated yet. Please use `my_predictor.fit(X, y)` before performing a prediction or accessing the non conformity scores.") from e
        return all_scores

    def calibrate(self, X_calib:Iterable[Any], y_calib:TensorLike):
        raise RuntimeError("Cross-conformal predictors do not require a separate calibration step. Please use the `fit` method to train and calibrate the model.")

    def fit(self, X:Iterable[Any], y:TensorLike)->Self:
        for X_fit, y_fit, X_calib, y_calib in self.splitter(X, y):
            self._conformal_predictors.append(self.conformal_predictor_class(clone_model(self.model), weight_function=self.weight_function, fit_function=self.fit_function))
            self._conformal_predictors[-1].fit(X_fit, y_fit)
            self._conformal_predictors[-1].calibrate(X_calib, y_calib)
        return self
    
    @abstractmethod
    def predict(self,
                X_test:Iterable[Any],
                alpha:TensorLike|float,
                correction:Callable|None = None)->ConformalPrediction:
        pass

class CVPlusRegressor(CrossConformalPredictor):
    def __init__(self,
                 model:Predictor|PredictorLike,
                 K:int=5,
                 random_state:int|None=None,
                 weight_function:Callable[[Iterable[Any]], Iterable[float]]|None = None,
                 fit_function:Callable[[Predictor, Iterable[Any], TensorLike], Predictor]|None = None):
        super().__init__(model,
                         splitter = KFoldSplitter(K=K, shuffle=True),
                         conformal_predictor_class=SplitCP,
                         random_state = random_state,
                         weight_function = weight_function,
                         fit_function = fit_function
                         )

    # TODO : see what can be moved to the parent class Here
    def predict(self, X_test:Iterable[Any], alpha:TensorLike|float, correction:Callable|None = None)->ConformalPrediction:
        n = self.len_calibr
        r_l = []
        r_u = []
        if correction is not None:
            alpha = correction(alpha)
        # TODO : avoid double loop ? vectorize ? force nc_scores to be more than a simple iterable ?
        for cp in self._conformal_predictors:
            for ricv in cp.nc_scores:
                r_l.append(cp.model(X_test) - ricv)
                r_u.append(cp.model(X_test) + ricv)
        l_stack = ops.stack(r_l, axis=0) # dim (n, b, 1)
        u_stack = ops.stack(r_u, axis=0) # dim (n, b, 1)
        l_stack = ops.sort(l_stack, axis=0)
        u_stack = ops.sort(u_stack, axis=0)
        l_alpha = l_stack[ops.cast(ops.ceil(alpha * (n+1)), int)]
        u_alpha = u_stack[ops.cast(ops.ceil((1 - alpha) * (n+1)), int)]
        # TODO : replace None with some aggregation of point predictions of multiple predictors ?
        return ConformalPrediction(None, ops.stack([l_alpha, u_alpha], axis=-1))

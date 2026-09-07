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
from deel.puncc.core.calibration import CalibrationContext
from deel.puncc.core.split import SplitConformalPredictor
from deel.puncc.core.conformal import ConformalPredictor, ConformalPrediction
from deel.puncc.corrections import AlphaCorrection


from deel.puncc.typing import Predictor, PredictorLike, TensorLike
from deel.puncc.core.splitters import KFoldSplitter, BaseSplitter
from deel.puncc.cloning import clone_model
from deel.puncc.regression.split import SplitConformalRegression
from deel.puncc import ops

class CrossConformalPredictor(ConformalPredictor):
    def __init__(self,
                 model:Predictor|PredictorLike,
                 conformal_predictor_class:Callable[..., SplitConformalPredictor],
                 splitter:BaseSplitter,
                 weight_function:Callable[[Iterable[Any]], Iterable[float]]|None = None,
                 fit_function:Callable[[Predictor, Iterable[Any], TensorLike], Predictor]|None = None):
        # TODO : implement WCV+
        if weight_function is not None:
            raise NotImplementedError(
                "Weighted CV+ is not supported yet. "
            )
        super().__init__(
            model=model,
            fit_function=fit_function,
        )
        self.splitter = splitter
        # Reserved for future WCV+ support.
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

    def compute_calibration_state(
        self,
        calibration_context:CalibrationContext,
    )->CalibrationContext:
        raise NotImplementedError(
            "Cross-conformal predictors do not use "
            "the standard calibration workflow."
        )


    def fit(self, X:Iterable[Any], y:TensorLike)->Self:
        self._conformal_predictors = []
        for ((X_fit, y_fit),(X_calib, y_calib)) in self.splitter(X=X, y=y):
            cp = self.conformal_predictor_class(clone_model(self.model), fit_function=self.fit_function)
            cp.fit(X_fit, y_fit)
            cp.calibrate(X_calib, y_calib)
            self._conformal_predictors.append(cp)
        return self
    
    @abstractmethod
    def predict(self,
                X_test:Iterable[Any],
                alpha:float|TensorLike,
                *,
                alpha_correction: Callable[[float|TensorLike], float|TensorLike] | None = None,)->ConformalPrediction[Any, Any]:
        pass

    def conformalize(
        self,
        prediction:Any,
        alpha:float,
        calibration_context:CalibrationContext,
    )->ConformalPrediction[Any, Any]:
        raise NotImplementedError(
            "Cross-conformal predictors implement "
            "their own prediction procedure."
        )

class CVPlusRegressor(CrossConformalPredictor):
    def __init__(self,
                 model:Predictor|PredictorLike,
                 K:int=5,
                 random_state:int|None=None,
                 weight_function:Callable[[Iterable[Any]], Iterable[float]]|None = None,
                 fit_function:Callable[[Predictor, Iterable[Any], TensorLike], Predictor]|None = None):
        super().__init__(model,
                         splitter = KFoldSplitter(K=K, shuffle=True, random_state=random_state),
                         conformal_predictor_class=SplitConformalRegression,
                         weight_function = weight_function,
                         fit_function = fit_function
                         )

    # TODO : see what can be moved to the parent class Here
    def predict(self, X_test:Iterable[Any], alpha:float|TensorLike, correction:AlphaCorrection|None = None)->ConformalPrediction:
        n = self.len_calibr
        r_l = []
        r_u = []
        if correction is not None:
            alpha = correction(alpha)
        # TODO : avoid double loop ? vectorize ? force nc_scores to be more than a simple iterable ?

        predictions = []
        for cp in self._conformal_predictors:
            prediction = cp.model(X_test)
            predictions.append(prediction)
            for ricv in cp.nc_scores:
                r_l.append(prediction - ricv)
                r_u.append(prediction + ricv)
        l_stack = ops.stack(r_l, axis=0) # dim (n, b, 1)
        u_stack = ops.stack(r_u, axis=0) # dim (n, b, 1)
        l_stack = ops.sort(l_stack, axis=0)
        u_stack = ops.sort(u_stack, axis=0)
        # TODO : revoir les formules des indices ici
        l_alpha = l_stack[ops.cast(ops.ceil(alpha * (n+1)), int)]
        u_alpha = u_stack[ops.cast(ops.ceil((1 - alpha) * (n+1)), int)]

        # TODO : See if mean is the best aggregation here
        point_prediction = ops.mean(
            ops.stack(predictions, axis=0),
            axis=0,
        )
        return ConformalPrediction(point_prediction, ops.stack([l_alpha, u_alpha], axis=-1))

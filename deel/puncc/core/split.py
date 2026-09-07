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
Basic components for split conformal prediction
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Self

from deel.puncc import ops
from deel.puncc.core.calibration import CalibrationContext
from deel.puncc.core.conformal import ConformalPredictor, ConformalPrediction
from deel.puncc.core.predictors import make_predictor
from deel.puncc.typing import (
    FitFunction,
    NCScoreFunction,
    Predictor,
    PredictorLike,
    PredSetFunction,
    TensorLike,
    WeightFunction,
)

class SplitConformalPredictor(ConformalPredictor):
    """
    Base class for split conformal prediction methods

    Args:
        model (Predictor | PredictorLike): underlying model
        nc_score_function (NCScoreFunction): function to used to compute non conformity scores from the model predictions and the true labels.
        pred_set_function (PredSetFunction): function to build a prediction set from the model prediction and a non conformity score threshold.
        fit_function (Callable[[Predictor, Iterable[Any], TensorLike], Predictor], optional): Optional function that trains the model. Defaults to None.
    """
    __slots__ = (
        "nc_score_function",
        "pred_set_function",
    )

    def __init__(self,
                 model:Predictor|PredictorLike,
                 nc_score_function:NCScoreFunction,
                 pred_set_function: PredSetFunction,
                 *,
                 fit_function:FitFunction|None = None)->None:
        super().__init__(model=model, fit_function=fit_function)

        self.nc_score_function = nc_score_function
        self.pred_set_function = pred_set_function

    @property
    def len_calibr(self) -> int:
        return len(self.nc_scores)

    @property
    def nc_scores(self) -> TensorLike:
        if not hasattr(
            self.calibration_context,
            "nc_scores",
        ):
            raise RuntimeError(
                "The conformal predictor has not been calibrated yet."
            )
        return self.calibration_context.nc_scores

    def compute_calibration_state(self, calibration_context:CalibrationContext)->CalibrationContext:
        calibration_context.nc_scores = (
            self.nc_score_function(
                calibration_context.y_pred,
                calibration_context.y_calib)
        )
        return calibration_context

    def conformalize(self,
                    prediction:Any,
                    alpha:float,
                    calibration_context:CalibrationContext)->ConformalPrediction[Any, Any]:
        quantile = self._get_quantile(
            alpha,
            calibration_context,
        )

        prediction_sets = self.pred_set_function(prediction, quantile)
        return ConformalPrediction(prediction, prediction_sets)

    def _get_quantile(
        self,
        alpha: float|TensorLike,
        calibration_context: CalibrationContext,
    )->TensorLike:
        key = self._make_cache_key(alpha, calibration_context)

        if key not in self.conformalization_cache:
            scores = calibration_context.nc_scores
            n = len(scores)
            self.conformalization_cache[key] = self._compute_quantile(
                scores,
                (1 - alpha) * (n + 1) / n,
                calibration_context,
            )
        return self.conformalization_cache[key]

    def _compute_quantile(
        self,
        scores: TensorLike,
        level: float|TensorLike,
        calibration_context: CalibrationContext,
    ) -> TensorLike:
        return ops.weighted_quantile(
                        scores,
                        level,
                        axis=0,
                        weights=None,
                    )

    def __getstate__(self):
        state = {}
        if getattr(self, "__dict__", None):
            state = self.__dict__.copy()

        for cls in type(self).mro():
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name in ("__dict__", "__weakref__", "model", "conformalization_cache"):
                    continue
                if hasattr(self, name):
                    state[name] = getattr(self, name)
        # Remove the model from the state to avoid serialization issues
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)

    def save(self, path:Path|str)->None:
        with open(path, "wb") as f:
            pickle.dump(self.__getstate__(), f)

    @classmethod
    def load(cls, path:Path|str, model:Predictor |PredictorLike)->Self:
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__setstate__(state)
        obj.conformalization_cache = {}
        obj.model = make_predictor(model)
        return obj
    
class WeightedQuantileMixin(SplitConformalPredictor):
    def __init__(self, *args:Any, weight_function:WeightFunction, **kwargs:Any)->None:
        super().__init__(*args, **kwargs)
        self.weight_function = weight_function

    def _compute_quantile(
        self,
        scores: TensorLike,
        level: float|TensorLike,
        calibration_context: CalibrationContext,
    ) -> TensorLike:
        weights = self.weight_function(
            calibration_context.X_calib
        )

        return ops.weighted_quantile(
            scores,
            level,
            axis=0,
            weights=weights,
        )

class PresetSplitConformalPredictor(SplitConformalPredictor):
    nc_score_function:NCScoreFunction
    pred_set_function:PredSetFunction
    def __init__(self, model:Predictor|PredictorLike, fit_function:FitFunction|None=None):
        super().__init__(
            model=model,
            nc_score_function=type(self).nc_score_function,
            pred_set_function=type(self).pred_set_function,
            fit_function=fit_function,
        )


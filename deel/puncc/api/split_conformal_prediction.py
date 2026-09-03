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
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Callable, Self

from deel.puncc import ops
from deel.puncc.api.calibration_context import CalibrationContext
from deel.puncc.api.conformal_prediction import ConformalPredictor, ConformalPrediction
from deel.puncc.api.splitting import ClasswiseSplitter
from deel.puncc.typing import (
    FitFunction,
    NCScoreFunction,
    Predictor,
    PredictorLike,
    PredSetFunction,
    TensorLike,
    make_predictor,
)

class SplitConformalPredictor(ConformalPredictor):
    """
    Base class for split conformal prediction methods

    Args:
        model (Predictor | PredictorLike): underlying model
        nc_score_function (NCScoreFunction): function to used to compute non conformity scores from the model predictions and the true labels.
        pred_set_function (PredSetFunction): function to build a prediction set from the model prediction and a non conformity score threshold.
        weight_function (Callable[[Iterable[Any]], Iterable[float]], optional): Optional function to allocate different weights to the calibration samples when computing the quantile of the non conformity scores. Defaults to None, which corresponds to the standard unweighted conformal prediction method.
        fit_function (Callable[[Predictor, Iterable[Any], TensorLike], Predictor], optional): Optional function that trains the model. Defaults to None.
    """
    __slots__ = (
        "nc_score_function",
        "pred_set_function",
        "weight_function",
        "_quantile_cache",
    )

    def __init__(self,
                 model:Predictor|PredictorLike,
                 nc_score_function:NCScoreFunction,
                 pred_set_function: PredSetFunction,
                 *,
                 weight_function:Callable[[Iterable[Any]], Iterable[float]]|None = None,
                 fit_function:FitFunction|None = None):
        # Definition of conformal predictor components :
        super().__init__(model=model, fit_function=fit_function)
        self.nc_score_function = nc_score_function
        self.pred_set_function = pred_set_function
        self.weight_function = weight_function
        self._quantile_cache: dict[
            tuple[CalibrationContext, float],
            Any,
        ] = {}

    @property
    def len_calibr(self) -> int:
        return len(self.nc_scores)

    @property
    def nc_scores(self) -> Sequence[float]:
        if not hasattr(
            self.calibration_context,
            "nc_scores",
        ):
            raise RuntimeError(
                "The conformal predictor has not been calibrated yet."
            )
        return self.calibration_context.nc_scores

    def compute_calibration_state(self, calibration_context:CalibrationContext)->CalibrationContext:
        self._quantile_cache.clear()
        calibration_context.nc_scores = (
            self.nc_score_function(
                calibration_context.y_pred,
                calibration_context.y_calib)
        )
        ###TODO : le contexte pourrait être allégé :
        #del calibration_context.y_pred
        #del calibration_context.y_calib
        return calibration_context

    def conformalize(self,
                    prediction:Any,
                    alpha:float,
                    calibration_context:CalibrationContext|None = None)->ConformalPrediction[Any, Any]:

        if calibration_context is None:
            calibration_context = self.calibration_context

        quantile = self._compute_quantile(
            alpha,
            calibration_context,
        )

        prediction_sets = self.pred_set_function(prediction, quantile)
        return ConformalPrediction(prediction, prediction_sets)

    def _compute_quantile(
        self,
        alpha: float|TensorLike,
        calibration_context: CalibrationContext,
    )->float|TensorLike:
        cache_key = (
            calibration_context,
            alpha,
        )
        if cache_key not in self._quantile_cache:
            scores = calibration_context.nc_scores
            n = len(scores)

            weights = None
            if self.weight_function is not None:
                weights = self.weight_function(
                    calibration_context.X_calib
                )

            self._quantile_cache[cache_key] = ops.weighted_quantile(
                scores,
                (1 - alpha) * (n + 1) / n,
                axis=0,
                weights=weights,
            )
        
        return self._quantile_cache[cache_key]

    def __getstate__(self):
        state = {}
        if getattr(self, "__dict__", None):
            state = self.__dict__.copy()

        for cls in type(self).mro():
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name in ("__dict__", "__weakref__", "model", "_quantile_cache"):
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
        obj._quantile_cache = {}
        obj.model = make_predictor(model)
        return obj

class StaticSplitConformalPredictor(SplitConformalPredictor):
    nc_score_function:NCScoreFunction
    pred_set_function:PredSetFunction
    def __init__(self, model, weight_function=None, fit_function=None):
        super().__init__(
            model=model,
            nc_score_function=type(self).nc_score_function,
            pred_set_function=type(self).pred_set_function,
            weight_function=weight_function,
            fit_function=fit_function,
        )

class ClassificationSplitConformalPredictor(StaticSplitConformalPredictor):
    @classmethod
    def pred_set_function(cls, y_pred:TensorLike, quantile:float|TensorLike):
        n, K = y_pred.shape[0], y_pred.shape[1]
        y_pred_tiled = ops.repeat(y_pred, repeats=K, axis=0)
        y_true_flat  = ops.tile(ops.arange(K), (n,))
        scores_flat = cls.nc_score_function(y_pred_tiled, y_true_flat)
        scores = ops.reshape(scores_flat, (n, K))
        mask = scores <= quantile
        return [ops.where_1d(mask[i]) for i in range(n)]


class ClasswiseConformalPredictorMixin(SplitConformalPredictor):#(ClassificationConformalPredictor):
    __slots__ = ("classwise_calibration_contexts",)
    splitter = ClasswiseSplitter()

    def calibrate(self, X_calib:Iterable[Any], y_calib:Iterable[Any])->Self:
        super().calibrate(X_calib, y_calib)
        self.classwise_calibration_contexts = self.splitter.split_context_by_group(self.calibration_context)
        return self

    def conformalize(self,
                    prediction:Any,
                    alpha:float,
                    calibration_context:CalibrationContext|None = None)->ConformalPrediction[Any, Any]:
        if calibration_context is None:
            calibration_context = self.calibration_context

        if calibration_context is self.calibration_context:
            classwise_contexts = self.classwise_calibration_contexts
        else:
            classwise_contexts = (
                self.splitter.split_context_by_group(
                    calibration_context
                )
            )

        n_classes = int(ops.shape(prediction)[-1])
        global_quantile = None
        quantiles = []

        for k in range(n_classes):
            if k in classwise_contexts:
                calib_context_k = classwise_contexts[k]
                quantile_k = self._compute_quantile(alpha, calib_context_k)
                quantiles.append(quantile_k)
            else:
                if global_quantile is None:
                    # TODO:
                    # Define the fallback strategy for classes absent from the calibration set.
                    # Using the global quantile is pragmatic but does not provide the
                    # class-conditional guarantee for the missing class.
                    global_quantile = self._compute_quantile(alpha, calibration_context)
                quantiles.append(global_quantile)
        y_set = self.pred_set_function(prediction, ops.stack(quantiles, axis=0))
        return ConformalPrediction(prediction, y_set)

class ScoreCalibrator:
    def __init__(self,
                 nc_score_function:Callable[[Iterable[Any]], Sequence[float]],
                 weight_function:Callable[[Iterable[Any]], Iterable[float]]|None = None):
        # Definition of conformal predictor components :
        self.nc_score_function = nc_score_function
        self.weight_function = weight_function

        # Utilities for the calibration procedure :
        self._z_calib = None
        self._nc_scores = None

    @property
    def len_calibr(self):
        if self._nc_scores is None:
            return 0
        return len(self._nc_scores)

    @property
    def nc_scores(self) -> Sequence[float]:
        if self._nc_scores is None:
            raise RuntimeError("The conformal predictor has not been calibrated yet. Please use the `calibrate` method before performing a prediction or accessing the non conformity scores.")
        return self._nc_scores

    def calibrate(self, z_calib:Iterable[Any]):
        self._z_calib = z_calib
        self._nc_scores = self.nc_score_function(z_calib)
        return self

    def is_conformal(self, z:Iterable[Any], alpha:float)->TensorLike:
        n = self.len_calibr
        weights = None
        if self.weight_function is not None:
            weights = self.weight_function(self._z_calib)
        quantile = ops.weighted_quantile(self.nc_scores, (1 - alpha) * (n + 1) / n, axis=0, weights=weights)
        test_nonconf_scores = self.nc_score_function(z)
        return test_nonconf_scores <= quantile

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
from pathlib import Path
from collections.abc import Sequence, Iterable
from typing import Any, Callable, Self
import pickle
from deel.puncc.typing import Predictor, PredictorLike, TensorLike, NCScoreFunction, PredSetFunction
from deel.puncc import ops
from deel.puncc.api.conformalization import ConformalMethod, ConformalPrediction

class NoModel(Predictor):
    """
    Empty model class used as a placeholder when loading a ConformalPredictor without a model.
    """
    def __call__(self, *args, **kwargs):
        raise RuntimeError("When loading a ConformalPredictor, the model must be set manually after loading. The model was not saved to avoid issues with model serialization. Please set the model attribute of the loaded ConformalPredictor instance to a valid model before using it.")

class ConformalPredictor(ConformalMethod):
    """
    Base class for split conformal prediction methods

    Args:
        model (Predictor | PredictorLike): underlying model
        nc_score_function (NCScoreFunction): function to used to compute non conformity scores from the model predictions and the true labels.
        pred_set_function (PredSetFunction): function to build a prediction set from the model prediction and a non conformity score threshold.
        weight_function (Callable[[Iterable[Any]], Iterable[float]], optional): Optional function to allocate different weights to the calibration samples when computing the quantile of the non conformity scores. Defaults to None, which corresponds to the standard unweighted conformal prediction method.
        fit_function (Callable[[Predictor, Iterable[Any], TensorLike], Predictor], optional): Optional function that trains the model. Defaults to None.
    """
    def __init__(self,
                 model:Predictor|PredictorLike,
                 nc_score_function:NCScoreFunction,
                 pred_set_function: PredSetFunction,
                 *,
                 weight_function:Callable[[Iterable[Any]], Iterable[float]]|None = None,
                 fit_function:Callable[[Predictor, Iterable[Any], TensorLike], Predictor]|None = None):
        # Definition of conformal predictor components :
        super().__init__(model=model, fit_function = fit_function)
        self.nc_score_function = nc_score_function
        self.pred_set_function = pred_set_function

        self.weight_function = weight_function

        # Utilities for the calibration procedure :
        self._x_calib = None
        self._nc_scores = None

    @property
    def len_calibr(self):
        """
        Size of the calibration set
        """
        if self._nc_scores is None:
            return 0
        return len(self._nc_scores)
    
    @property
    def nc_scores(self) -> Sequence[float]:
        if self._nc_scores is None:
            raise RuntimeError("The conformal predictor has not been calibrated yet. Please use `my_predictor.calibrate(X, y)` before performing a prediction or accessing the non conformity scores.")
        return self._nc_scores

    def calibrate(self, X_calib:Iterable[Any],
                  y_calib:TensorLike):
        predictions = self.model(X_calib)
        self._x_calib = X_calib
        self._nc_scores = self.nc_score_function(predictions, y_calib)
        return self

    def predict(self,
                X_test:Iterable[Any],
                alpha:TensorLike|float,
                correction:Callable|None = None)->ConformalPrediction:
        prediction = self.model(X_test)
        n = self.len_calibr
        weights = None
        if correction is not None:
            alpha = correction(alpha) # TODO : add kwargs and other stuff that may be impacted by the correction?
        if self.weight_function is not None:
            weights = self.weight_function(self._x_calib)
        quantile = ops.weighted_quantile(self.nc_scores, (1 - alpha) * (n + 1) / n, axis=0, weights=weights)
        prediction_sets = self.pred_set_function(prediction, quantile)
        return ConformalPrediction(prediction, prediction_sets)

    def __getstate__(self):
        state = {}
        if getattr(self, "__dict__", None):
            state = self.__dict__.copy()
        for cls in type(self).mro():
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name == "__dict__":
                    continue
                if hasattr(self, name):
                    state[name] = getattr(self, name)
        # Remove the model from the state to avoid serialization issues
        state["model"] = NoModel()
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)

    def save(self, path:Path|str)->None:
        with open(path, "wb") as f:
            pickle.dump(self.__getstate__(), f)

    @classmethod
    def load(cls, path:Path|str)->ConformalPredictor:
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__setstate__(state)
        return obj
    
class StaticConformalPredictor(ConformalPredictor):
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

class ClassificationConformalPredictor(StaticConformalPredictor):
    @classmethod
    def pred_set_function(cls, y_pred:TensorLike, quantile:float|TensorLike):
        n, K = y_pred.shape[0], y_pred.shape[1]
        y_pred_tiled = ops.repeat(y_pred, repeats=K, axis=0)
        y_true_flat  = ops.tile(ops.arange(K), (n,))
        scores_flat = cls.nc_score_function(y_pred_tiled, y_true_flat)
        scores = ops.reshape(scores_flat, (n, K))
        mask = scores <= quantile
        return [ops.where_1d(mask[i]) for i in range(n)]
    
class ClasswiseConformalPredictorMixin(ClassificationConformalPredictor):
    def calibrate(self, X_calib:Iterable[Any], y_calib:TensorLike)->Self:
        super().calibrate(X_calib, y_calib)
        self._y_calib = y_calib
        return self

    def predict(self,
                X_test:Iterable[Any],
                alpha:TensorLike|float,
                correction:Callable|None = None)->ConformalPrediction:
        prediction = self.model(X_test)

        if correction is not None:
            alpha = correction(alpha)
        
        weights = None
        if self.weight_function is not None:
            weights = self.weight_function(self._x_calib)

        scores = self.nc_scores
        y_calib = self._y_calib

        nb_classes = int(ops.shape(prediction)[-1])
        n = self.len_calibr
        q_global = None#ops.weighted_quantile(scores, (1 - alpha) * (n + 1) / n, axis=0, weights=weights)
        qs = []
        for k in range(nb_classes):
            mask = ops.equal(y_calib, k)
            s_k = scores[mask]
            n_k = len(s_k)
            # TODO : check that
            if n_k == 0:
                if q_global is None:
                    q_global = ops.weighted_quantile(scores, (1 - alpha) * (n + 1) / n, axis=0, weights=weights)
                qs.append(q_global)
            else:
                qs.append(ops.weighted_quantile(s_k, (1 - alpha) * (n_k + 1) / n, axis=0, weights=weights))
        q = ops.stack(qs, axis=0)
        y_set = self.pred_set_function(prediction, q)
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


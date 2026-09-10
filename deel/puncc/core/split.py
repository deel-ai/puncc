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
Core components for split conformal prediction.

This module defines the base split conformal predictor, support for weighted quantile computation,
and a convenience base class for predictors with predefined nonconformity score and prediction-set functions.
"""

from __future__ import annotations


from typing import Any

from deel.puncc import ops
from deel.puncc.core.calibration import CalibrationContext
from deel.puncc.core.conformal import ConformalPredictor, ConformalPrediction
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
    The predictor computes nonconformity scores from a calibration dataset and uses their empirical quantile to construct conformal prediction sets.

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
        """
        Size of the calibration set used to compute nonconformity scores.
        """
        return len(self.nc_scores)

    @property
    def nc_scores(self) -> TensorLike:
        """
        Nonconformity scores computed on the calibration dataset.

        Returns:
            Calibration nonconformity scores.

        Raises:
            RuntimeError: If the predictor has not been calibrated.
        """
        if not hasattr(
            self.calibration_context,
            "nc_scores",
        ):
            raise RuntimeError(
                "The conformal predictor has not been calibrated yet."
            )
        return self.calibration_context.nc_scores

    def compute_calibration_state(self, calibration_context:CalibrationContext)->CalibrationContext:
        """
        Compute the nonconformity scores associated with a calibration context.
        The scores are computed from the stored model predictions and calibration targets and added to the provided context.

        Args:
            calibration_context: Context containing calibration predictions and targets.

        Returns:
            The updated calibration context containing nonconformity scores.
        """
        calibration_context.nc_scores = (
            self.nc_score_function(
                calibration_context.y_pred,
                calibration_context.y_calib)
        )
        return calibration_context

    def conformalize(self,
                    prediction:Any,
                    alpha:float|TensorLike,
                    calibration_context:CalibrationContext)->ConformalPrediction[Any, Any]:
        """
        Conformalize model predictions.

        The conformal quantile is computed from the provided calibration context and used to construct prediction sets.

        Args:
            prediction: Base predictions to conformalize.
            alpha: Target miscoverage level.
            calibration_context: Calibration state used for conformalization.

        Returns:
            The base predictions and their associated conformal prediction sets.
        """
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
        """
        Return the conformal quantile associated with a miscoverage level.
        Previously computed quantiles may be retrieved from the conformalization cache.

        Args:
            alpha: Target miscoverage level.
            calibration_context: Calibration context providing nonconformity scores.

        Returns:
            Conformal nonconformity threshold.
        """
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
        """
        Compute an empirical quantile of nonconformity scores.

        This default implementation uses uniform weights.
        Subclasses may override this method to implement alternative quantile computations.

        Args:
            scores: Calibration nonconformity scores.
            level: Quantile level.
            calibration_context: Calibration context associated with the scores.

        Returns:
            Empirical quantile of the nonconformity scores.
        """
        return ops.weighted_quantile(
            scores,
            level,
            axis=0,
            weights=None,
        )


    
class WeightedQuantileMixin(SplitConformalPredictor):
    """
    Add sample-weighted quantile computation to split conformal prediction.

    The provided weight function is evaluated on the calibration inputs and its output is used when computing conformal quantiles.

    Args:
        weight_function: Function assigning a weight to each calibration sample.
    """
    def __init__(self, *args:Any, weight_function:WeightFunction, **kwargs:Any)->None:
        super().__init__(*args, **kwargs)
        self.weight_function = weight_function

    def _compute_quantile(
        self,
        scores: TensorLike,
        level: float|TensorLike,
        calibration_context: CalibrationContext,
    ) -> TensorLike:
        """
        Compute a weighted empirical quantile of nonconformity scores.

        Args:
            scores: Calibration nonconformity scores.
            level: Quantile level.
            calibration_context: Calibration context containing the inputs used to compute sample weights.

        Returns:
            Weighted empirical quantile of the nonconformity scores.
        """
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
    """
    Base class for split conformal methods with predefined score functions.

    Subclasses define ``nc_score_function`` and ``pred_set_function`` as class attributes.
    Instances therefore only require the predictive model and, optionally, a custom fitting function.

    Args:
        model: Underlying predictive model.
        fit_function: Optional custom function used to fit the predictive model.
    """
    __slots__ = ()
    nc_score_function:NCScoreFunction
    pred_set_function:PredSetFunction
    def __init__(self, model:Predictor|PredictorLike, fit_function:FitFunction|None=None):
        super().__init__(
            model=model,
            nc_score_function=type(self).nc_score_function,
            pred_set_function=type(self).pred_set_function,
            fit_function=fit_function,
        )

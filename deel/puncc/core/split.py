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


from typing import Any, Iterable, Self, TypeAlias
from collections.abc import Callable

import warnings

from deel.puncc import ops
from deel.puncc.core.calibration import CalibrationContext
from deel.puncc.core.conformal import ConformalPredictor, ConformalPrediction
from deel.puncc.core.predictors import make_predictor
from deel.puncc.nonconformity_scores import absolute_difference
from deel.puncc.typing import (
    FitFunction,
    NCScoreFunction,
    Predictor,
    PredictorLike,
    PredSetFunction,
    TensorLike,
    WeightFunction,
)
from deel.puncc.warnings import ExperimentalWarning


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

    def __init__(
        self,
        model: Predictor | PredictorLike,
        *,
        nc_score_function: NCScoreFunction,
        pred_set_function: PredSetFunction,
        fit_function: FitFunction | None = None,
    ) -> None:
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
            raise RuntimeError("The conformal predictor has not been calibrated yet.")
        return self.calibration_context.nc_scores

    def compute_calibration_state(
        self, calibration_context: CalibrationContext
    ) -> CalibrationContext:
        """
        Compute the nonconformity scores associated with a calibration context.
        The scores are computed from the stored model predictions and calibration targets and added to the provided context.

        Args:
            calibration_context: Context containing calibration predictions and targets.

        Returns:
            The updated calibration context containing nonconformity scores.
        """
        calibration_context.nc_scores = self.nc_score_function(
            calibration_context.y_pred, calibration_context.y_calib
        )
        return calibration_context

    def conformalize(
        self,
        prediction: Any,
        alpha: float | TensorLike,
        calibration_context: CalibrationContext,
        *,
        X: Any | None = None,
    ) -> ConformalPrediction[Any, Any]:
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
        quantile = self._get_quantile(alpha, calibration_context, X=X)

        prediction_sets = self.pred_set_function(prediction, quantile)
        return ConformalPrediction(prediction, prediction_sets)

    @staticmethod
    def extend_scores_with_inf(scores: TensorLike) -> TensorLike:
        """
        Append an infinite nonconformity score along the sample axis.

        An infinite score is added to account for the finite-sample correction when computing conformal quantiles.

        Integer scores are converted to float32 so that infinity can be represented.
        Floating-point scores retain their dtype.

        Args:
            scores:
                Nonconformity scores with the sample dimension on axis 0.

        Returns:
            A tensor containing the original scores followed by one additional infinite score along axis 0.

        Notes:
            For multidimensional scores, the appended sample contains infinity in every component.
            The original scores are not modified.
        """
        if "float" not in ops.dtype(scores):
            scores = ops.cast(scores, "float32")
        return ops.concatenate(
            [scores, ops.full_like(scores[:1], float("inf"))], axis=0
        )

    def _get_quantile(
        self,
        alpha: float | TensorLike,
        calibration_context: CalibrationContext,
        *,
        X: Any | None = None,
    ) -> TensorLike:
        """
        Return the conformal quantile associated with a miscoverage level.

        Args:
            alpha: Target miscoverage level.
            calibration_context: Calibration context providing nonconformity scores.

        Returns:
            Conformal nonconformity threshold.
        """
        scores = self.extend_scores_with_inf(calibration_context.nc_scores)
        return ops.weighted_quantile(
            scores,
            1 - alpha,
            axis=0,
            weights=None,
        )


class WeightedQuantileMixin(SplitConformalPredictor):
    """
    Mixin providing weighted conformal quantiles for split conformal prediction.

    Nonconformity scores are weighted using a user-provided weight function.
    The weights are evaluated on the calibration inputs and separately on each test input.

    For each test sample, its weight is appended to the calibration weights and used to compute a test-dependent conformal quantile.

    Args:
        weight_function:
            Function mapping input samples to non-negative weights.
            It must return one weight per sample.

    Notes:
        The weight function is evaluated during calibration and prediction.

        Calibration weights are stored in the calibration context, while test weights are computed when conformalizing predictions.

        The validity of weighted conformal coverage guarantees depends on the assumptions underlying the chosen weighting scheme.
    """

    def __init__(
        self, *args: Any, weight_function: WeightFunction, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.weight_function = weight_function

    def compute_calibration_state(self, calibration_context):
        """
        Compute nonconformity scores and calibration sample weights.

        The weight function is evaluated on the calibration inputs, and the resulting weights are stored in the calibration context.

        Args:
            calibration_context:
                Context containing calibration inputs, targets and predictions.

        Returns:
            The updated calibration context containing nonconformity scores and calibration weights.
        """
        calibration_context = super().compute_calibration_state(calibration_context)
        calibration_context.calibration_weights = self.weight_function(
            calibration_context.X_calib
        )
        return calibration_context

    def _get_quantile(
        self,
        alpha: float | TensorLike,
        calibration_context: CalibrationContext,
        *,
        X: Any | None = None,
    ) -> TensorLike:
        """
        Compute a weighted conformal quantile for each test sample.

        Each quantile is computed from the calibration nonconformity scores, using the stored
        calibration weights and the weight associated with the corresponding test sample.

        Args:
            alpha:
                Target miscoverage level.
            calibration_context:
                Context containing nonconformity scores and calibration weights.
            X:
                Test inputs used to compute sample-specific weights.

        Returns:
            Tensor containing one conformal quantile per test sample, possibly with additional dimensions for multivariate scores.
        """

        scores = self.extend_scores_with_inf(calibration_context.nc_scores)
        calibration_weights = calibration_context.calibration_weights
        test_weights = self.weight_function(X)

        quantiles = []

        for test_weight in test_weights:
            weights = ops.concatenate(
                [calibration_weights, ops.reshape(test_weight, (1,))], axis=0
            )

            quantiles.append(
                ops.weighted_quantile(
                    scores,
                    1 - alpha,
                    axis=0,
                    weights=weights,
                )
            )
        return ops.stack(quantiles, axis=0)


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
    nc_score_function: NCScoreFunction
    pred_set_function: PredSetFunction

    def __init__(
        self,
        model: Predictor | PredictorLike,
        fit_function: FitFunction | None = None,
    ):
        super().__init__(
            model=model,
            nc_score_function=type(self).nc_score_function,
            pred_set_function=type(self).pred_set_function,
            fit_function=fit_function,
        )


LocalScaleFunction: TypeAlias = Callable[[Any], TensorLike]


class LocallyScaledMixin(SplitConformalPredictor):
    """
    Mixin providing locally scaled nonconformity scores.

    Calibration scores are divided by a sample-dependent scale estimated from the calibration inputs.
    During prediction, the conformal quantile is multiplied by the scale estimated for each test input.

    This allows prediction set sizes to adapt to local characteristics of the input data.

    Args:
        scale_function:
            Function mapping input samples to strictly positive scale
            estimates with shape (n_samples,) or (n_samples, 1).

    Notes:
        The scale function must be available during both calibration and prediction.
        A small epsilon is added to the scale to improve numerical stability. Its default value is 1e-12.
        The statistical validity of the resulting prediction sets depends on the assumptions of the underlying conformal method.
    """

    eps: float = 1e-12

    def __init__(self, *args, scale_function: LocalScaleFunction, **kwargs):
        self.scale_function = scale_function
        super().__init__(*args, **kwargs)

    def _scale(self, X: Any) -> TensorLike:
        """
        Compute and validate sample-dependent scale estimates.

        Args:
            X:
                Input samples used to evaluate the scale function.

        Returns:
            A one-dimensional tensor of strictly positive scale estimates with shape (n_samples,).

        Raises:
            ValueError:
                If the scale function returns a tensor with an invalid shape or containing non-positive values.

        Notes:
            Scale estimates with shape (n_samples, 1) are squeezed to shape (n_samples,).
        """
        scale = self.scale_function(X)
        if scale.ndim == 2 and scale.shape[-1] == 1:
            scale = ops.squeeze(scale, axis=-1)
        if scale.ndim != 1:
            raise ValueError("scale_function must return shape (n,) or (n, 1).")
        if ops.any(scale <= 0):
            raise ValueError(
                "Locally scaled conformal prediction requires strictly positive scale estimates."
            )
        return scale

    def compute_calibration_state(
        self, calibration_context: CalibrationContext
    ) -> CalibrationContext:
        """
        Compute locally normalized calibration scores.

        First compute the nonconformity scores using the parent method, then divide them by the sample-dependent scale estimates.

        Args:
            calibration_context:
                Context containing calibration inputs, targets and
                model predictions.

        Returns:
            The updated calibration context containing normalized
            nonconformity scores.

        Notes:
            Scale estimates are broadcast over any additional score
            dimensions, allowing multivariate nonconformity scores.
        """
        calibration_context = super().compute_calibration_state(calibration_context)
        scores = calibration_context.nc_scores
        scale = self._scale(calibration_context.X_calib)
        # Add singleton dimensions after the sample dimension to match multivariate nonconformity scores.
        if scores.ndim > 1:
            scale = ops.reshape(scale, (-1,) + (1,) * (scores.ndim - 1))

        scale = ops.cast(scale, scores.dtype)
        calibration_context.nc_scores = scores / (scale + self.eps)
        return calibration_context

    def conformalize(self, prediction, alpha, calibration_context=None, *, X=None):
        """
        Construct prediction sets using locally rescaled conformal quantiles.

        Compute the conformal quantile from normalized calibration scores,
        then multiply it by the scale estimated for each test sample.

        Args:
            prediction:
                Base model predictions.
            alpha:
                Target miscoverage level.
            calibration_context:
                Context containing normalized calibration scores.
            X:
                Test inputs used to compute local scale estimates.
                Required for conformalization.

        Returns:
            A ConformalPrediction containing the original predictions and their locally scaled prediction sets.

        Raises:
            RuntimeError:
                If the calibration context does not contain nonconformity scores.
            ValueError:
                If X is not provided or the estimated scales are invalid.

        Notes:
            Scale estimates are broadcast over additional dimensions when working with multivariate scores.
        """
        if not hasattr(calibration_context, "nc_scores"):
            raise RuntimeError("The conformal predictor has not been calibrated yet.")

        if X is None:
            raise ValueError("X is required for locally scaled conformal prediction.")

        quantile = self._get_quantile(alpha, calibration_context, X=X)
        scale = self._scale(X)

        score_ndim = calibration_context.nc_scores.ndim - 1
        if score_ndim > 0:
            scale = ops.reshape(scale, (-1,) + (1,) * score_ndim)

        scale = ops.cast(scale, quantile.dtype)
        quantile = quantile * (scale + self.eps)

        return ConformalPrediction(
            prediction, self.pred_set_function(prediction, quantile)
        )


class LocallyAdaptiveMixin(LocallyScaledMixin):
    def __init__(
        self,
        *args,
        dispersion_estimator: Predictor | PredictorLike,
        dispersion_estimation_function: Callable[
            [TensorLike, TensorLike], TensorLike
        ] = absolute_difference(),
        **kwargs,
    ):
        self.dispersion_estimator = make_predictor(dispersion_estimator)
        self.dispersion_estimation_function = dispersion_estimation_function
        super().__init__(*args, scale_function=self.dispersion_estimator, **kwargs)

    def fit(
        self,
        X: Iterable[Any],
        y: Iterable[Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        super().fit(X, y, *args, **kwargs)
        mu_pred = self.model(X)
        self.dispersion_estimator.fit(
            X, self.dispersion_estimation_function(mu_pred, y)
        )
        return self


def inverse_root_leverage_weight(h: TensorLike) -> TensorLike:
    return 1 / ops.sqrt(1 + h)


class LeverageWeightedMixin(LocallyScaledMixin):
    def __init__(
        self,
        *args,
        leverage_weight_function: Callable[
            [TensorLike], TensorLike
        ] = inverse_root_leverage_weight,
        **kwargs,
    ):
        self.leverage_weight_function = leverage_weight_function
        self._feature_mean = None
        self._feature_scale = None
        self._gram_pinv = None

        super().__init__(
            *args,
            scale_function=self._leverage_scale,
            **kwargs,
        )

    def fit(
        self, X: Iterable[Any], y: Iterable[Any] | None = None, *args, **kwargs
    ) -> Self:
        super().fit(X, y, *args, **kwargs)
        return self.fit_leverage(X)

    def fit_leverage(self, X) -> Self:
        X = ops.array(X)

        if ops.shape(X)[0] <= ops.shape(X)[1]:
            raise ValueError(
                "LeverageWeightedCP requires more training samples than features."
            )

        self._feature_mean = ops.mean(X, axis=0)
        self._feature_scale = ops.std(X, axis=0)

        if ops.item(ops.any(self._feature_scale == 0)):
            raise ValueError("LeverageWeightedCP requires non-constant features.")

        X = (X - self._feature_mean) / self._feature_scale
        self._gram_pinv = ops.pinv(ops.transpose(X) @ X)
        return self

    def _leverage(self, X):
        if self._gram_pinv is None:
            raise RuntimeError("Call fit() or fit_leverage() before calibration.")

        X = (ops.array(X) - self._feature_mean) / self._feature_scale
        return ops.sum((X @ self._gram_pinv) * X, axis=-1)

    def _leverage_scale(self, X):
        return 1 / self.leverage_weight_function(self._leverage(X))

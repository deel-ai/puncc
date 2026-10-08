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
Sequential conformal regression using Ensemble Batch Prediction Intervals.

EnbPI initializes signed residuals from out-of-bag ensemble predictions and
updates their empirical distribution as new outcomes become available.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Integral
from typing import Any, Never, Self

from deel.puncc import ops
from deel.puncc.core.calibration import CalibrationContext
from deel.puncc.core.conformal import CacheKey, ConformalPrediction, alpha_cache_key
from deel.puncc.core.ensemble import BootstrapEnsemble
from deel.puncc.core.samplers import BootstrapSampler
from deel.puncc.exceptions import NotCalibratedError
from deel.puncc.typing import AlphaCorrection, FitFunction, Predictor, PredictorLike, TensorLike


class EnbPIRegressor:
    """
    Ensemble Batch Prediction Intervals for scalar sequential regression.

    Fitting trains a bootstrap ensemble and initializes signed residuals as
    ``y - out_of_bag_prediction``, in the original training order. Prediction
    uses the ensemble's nested aggregation and two residual quantiles. A grid
    search over beta in [0, alpha] selects the narrowest candidate interval.

    The ``predict`` method returns a ConformalPrediction containing
    point predictions of shape (n_test,) and intervals of shape (n_test, 2).
    It does not advance the residual window. Call ``update`` with observed
    outcomes and the predictions actually issued to advance that window.

    Args:
        model: Base regression model used as the bootstrap ensemble template.
        B (int): Number of bootstrap models.
        sampler: Bootstrap index sampler. Defaults to the ensemble's IID
            sampler. Supply a seeded sampler for reproducible resampling.
        aggregation: "mean", "median", or an aggregation callable accepting
            ``axis=0``. Used at both levels of the bootstrap ensemble.
        fit_function: Optional function training each bootstrap model and
            returning the fitted predictor.
        beta_grid_size (int): Number of equally spaced beta candidates,
            including both endpoints. Must be at least 2. Defaults to 101.

    Attributes:
        model: The underlying BootstrapEnsemble.
        window_size_: Residual window length, set to the training sample count
            by fit. None before fitting.

    Note:
        Quantiles use the inverse empirical CDF without interpolation, with
        the sample minimum and maximum at levels 0 and 1. No split-conformal
        finite-sample correction is applied. The beta search is approximate.
        EnbPI's coverage results require assumptions on the regression error
        and its temporal dependence; exact finite-sample coverage is not
        guaranteed for arbitrary time series.

    References:
        Xu and Xie, "Conformal prediction for time series", Algorithm 1:
        https://arxiv.org/abs/2010.09107
    """

    def __init__(
        self,
        model:Predictor|PredictorLike,
        *,
        B:int=50,
        sampler:BootstrapSampler|None=None,
        aggregation:str|Callable[..., TensorLike]="mean",
        fit_function:FitFunction|None=None,
        beta_grid_size:int=101,
    ) -> None:
        if (
            isinstance(beta_grid_size, bool)
            or not isinstance(beta_grid_size, Integral)
            or beta_grid_size < 2
        ):
            raise ValueError("beta_grid_size must be an integer >= 2.")

        self.model = BootstrapEnsemble(
            model,
            B=B,
            sampler=sampler,
            aggregation=aggregation,
            fit_function=fit_function,
        )
        self.calibration_context = CalibrationContext()
        self.conformalization_cache: dict[CacheKey, TensorLike] = {}
        self.beta_grid_size = int(beta_grid_size)
        self.window_size_: int|None = None

    @property
    def residuals(self) -> TensorLike:
        """Current signed residuals, ordered from oldest to newest."""
        if "residuals" not in self.calibration_context:
            raise NotCalibratedError("EnbPIRegressor must be fitted before use.")
        return self.calibration_context.residuals

    @property
    def len_calibr(self) -> int:
        """Number of residuals in the current calibration window."""
        return len(self.residuals)

    def calibrate(self, X_calib:Iterable[Any], y_calib:TensorLike) -> Never:
        """EnbPI initializes calibration in fit and accepts feedback via update."""
        raise RuntimeError(
            "EnbPIRegressor does not use a separate calibration set. "
            "Call fit to initialize it, and update to incorporate observed outcomes."
        )

    def compute_calibration_state(
        self,
        calibration_context:CalibrationContext,
    ) -> CalibrationContext:
        """
        Store aligned targets, predictions, and their signed residuals.

        During initialization, y_pred must contain out-of-bag predictions.
        For feedback, it must contain predictions issued before observing y.
        """
        y = ops.reshape(
            ops.convert_to_tensor(calibration_context.y_calib),
            (-1,),
        )
        prediction = ops.reshape(
            ops.convert_to_tensor(calibration_context.y_pred),
            (-1,),
        )
        residuals = ops.subtract(y, prediction)
        calibration_context.update(
            y_calib=y,
            y_pred=prediction,
            residuals=residuals,
        )
        self.conformalization_cache.clear()
        return calibration_context

    def fit(
        self,
        X:Iterable[Any],
        y:TensorLike,
        *args:Any,
        **kwargs:Any,
    ) -> Self:
        """
        Fit the ensemble and initialize the residual window internally.

        Args:
            X: Training inputs in chronological order.
            y: Scalar training targets aligned with X.
            *args: Additional arguments forwarded to each bootstrap fit.
            **kwargs: Additional keyword arguments forwarded to each fit.

        Returns:
            The fitted regressor. A successful refit resets the entire window.
        """

        # Prepare a new ensemble so a failed fit or OOB prediction cannot pair
        # new models with residuals left over from a previous successful fit.
        ensemble = BootstrapEnsemble(
            self.model.model,
            B=self.model.B,
            sampler=self.model.sampler,
            aggregation=self.model.aggregation,
            fit_function=self.model.fit_function,
        )
        ensemble.fit(X, y, *args, **kwargs)
        context = CalibrationContext(
            y_calib=y,
            y_pred=ensemble.predict_oob_training(X),
        )
        context = self.compute_calibration_state(context)

        self.model = ensemble
        self.calibration_context = context
        self.window_size_ = len(y)
        return self

    def predict(
        self,
        X_test:Iterable[Any],
        alpha:float|TensorLike,
        *,
        alpha_correction:AlphaCorrection|None=None,
    ) -> ConformalPrediction[Any, Any]:
        """
        Produce ensemble point predictions and EnbPI prediction intervals.

        Args:
            X_test: Inputs on which predictions are produced.
            alpha: Requested scalar miscoverage level.
            alpha_correction: Optional transformation applied to alpha before
                constructing the intervals.

        Returns:
            Point predictions of shape (n_test,) and intervals of shape
            (n_test, 2), stored in a ConformalPrediction.

        Note:
            Prediction leaves the residual window unchanged. Call update
            with the corresponding observed outcomes to advance the window.
        
        Example:
            For a fitted regressor and chronological test arrays, process
            one feedback batch at a time::

                batch_size = 1
                for start in range(0, len(X_test), batch_size):
                    stop = start + batch_size
                    result = regressor.predict(
                        X_test[start:stop], alpha=0.1
                    )
                    # Update once this batch's outcomes are observed.
                    regressor.update(
                        y_test[start:stop],
                        prediction=result.prediction,
                    )

            Increasing batch_size delays feedback until the entire batch
            has been predicted.
        """
        prediction = self.model(X_test)
        if alpha_correction is not None:
            alpha = alpha_correction(alpha)
        return self.conformalize(
            prediction, alpha, self.calibration_context, X=X_test
        )

    def _get_offsets(
        self,
        alpha:float,
        calibration_context:CalibrationContext,
    ) -> TensorLike:
        """Find and cache the narrowest pair of residual quantiles on the grid."""
        if "residuals" not in calibration_context:
            raise NotCalibratedError("The calibration context has no EnbPI residuals.")
        key = (calibration_context, alpha_cache_key(alpha))
        if key not in self.conformalization_cache:
            residuals = calibration_context.residuals
            sorted_residuals = ops.sort(residuals, axis=0)
            n = len(residuals)
            beta = ops.linspace(0.0, alpha, self.beta_grid_size)
            levels = ops.stack([beta, 1 - alpha + beta], axis=0)

            # Inverse empirical CDF: rank ceil(n * q), expressed as a zero-based
            # index. Clipping handles q=0 and numerical rounding near q=1.
            indices = ops.cast(ops.clip(ops.ceil(n * levels) - 1, 0, n - 1), "int32")
            bounds = ops.take(sorted_residuals, indices, axis=0)
            best = ops.argmin(bounds[1] - bounds[0], axis=0)
            # argmin selects the first (smallest-beta) candidate on width ties.
            self.conformalization_cache[key] = ops.take(bounds, best, axis=1)
        return self.conformalization_cache[key]

    def conformalize(
        self,
        prediction:TensorLike,
        alpha:float|TensorLike,
        calibration_context:CalibrationContext,
        *,
        X:Any|None=None,
    ) -> ConformalPrediction[Any, Any]:
        """
        Add residual-quantile offsets to already-computed predictions.

        All calibration information comes from the supplied context. Neither
        models nor observations are updated. X is unused and retained for
        compatibility with other PUNCC conformalization methods.
        """
        prediction = ops.reshape(ops.convert_to_tensor(prediction), (-1,))
        offsets = self._get_offsets(alpha, calibration_context)
        intervals = ops.expand_dims(prediction, axis=-1) + offsets
        return ConformalPrediction(prediction, intervals)

    def update(self, y:TensorLike, *, prediction:TensorLike) -> Self:
        """
        Incorporate observed outcomes without retraining the ensemble.

        Args:
            y: Newly observed targets, in chronological order. Accepts a
                scalar, a vector, or a single-column batch.
            prediction: Corresponding point predictions issued before the
                targets were observed, for example result.prediction.

        Returns:
            The regressor with its updated calibration window.

        Note:
            Each observation must be supplied exactly once and in time order.
            Callers control feedback batch size by choosing when to update.
            The most recent window_size_ observations are retained, even when
            an incoming batch is larger than the window. Predictions are not
            recomputed and bootstrap models are not fitted again.
        """
        if self.window_size_ is None:
            raise NotCalibratedError("EnbPIRegressor must be fitted before update.")
        y = ops.reshape(ops.convert_to_tensor(y), (-1,))
        prediction = ops.reshape(ops.convert_to_tensor(prediction),(-1,))
        if len(y) != len(prediction):
            raise ValueError("Targets and predictions must have the same length.")

        previous = self.calibration_context
        targets = ops.concatenate([previous.y_calib, y], axis=0)[-self.window_size_:]
        predictions = ops.concatenate([previous.y_pred, prediction], axis=0)[-self.window_size_:]
        context = CalibrationContext(y_calib=targets, y_pred=predictions)
        self.calibration_context = self.compute_calibration_state(context)
        return self

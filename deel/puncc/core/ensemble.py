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
Bootstrap ensembles for scalar regression.

The ensemble fits independent predictors on bootstrap samples and keeps track
of which models excluded each training observation. This supports the nested
out-of-bag aggregation used by EnbPI, independently of residual calibration.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Integral
from typing import Any, Self

from deel.puncc import ops
from deel.puncc.cloning import clone_model
from deel.puncc.core.predictors import make_predictor
from deel.puncc.core.samplers import BootstrapSampler, IIDBootstrapSampler
from deel.puncc.core.splitters import tensor_indexing
from deel.puncc.typing import FitFunction, Predictor, PredictorLike, TensorLike


class BootstrapEnsemble:
    """
    Bootstrap ensemble with observation-specific out-of-bag aggregation.

    For training observation i, the leave-one-out predictor aggregates only
    models whose bootstrap sample excluded i. The final predictor aggregates
    these leave-one-out predictions over all training observations.

    Args:
        model: Regression model used as the template for independent clones.
        B (int): Number of bootstrap models. Must be positive.
        sampler: Bootstrap index sampler. Defaults to IIDBootstrapSampler.
        aggregation: "mean", "median", or a callable accepting ``axis=0`` and
            removing that axis. The same aggregation is used at both levels.
        fit_function: Optional function fitting each clone and returning the
            fitted predictor. Otherwise, the clone's ``fit`` method is used.

    Attributes:
        models_: Fitted predictors, in bootstrap sampling order.
        oob_mask_: Boolean tensor of shape (n_train, B). Entry (i, b) is true
            when model b was trained without observation i. None before fit.
        n_train_: Number of original training observations. None before fit.

    Note:
        Inputs must support row selection through the shared tensor_indexing
        helper: arrays, compatible tensors, or lists/tuples of samples.
        Select the PUNCC backend before
        fitting. Predictions must be finite scalars, shaped (m,) or (m, 1).
        A sampler's random seed controls resampling; model randomness is
        configured on the template or in ``fit_function``.
    """

    def __init__(
        self,
        model:Predictor|PredictorLike,
        *,
        B:int=50,
        sampler:BootstrapSampler|None=None,
        aggregation:str|Callable[..., TensorLike]="mean",
        fit_function:FitFunction|None=None,
    ) -> None:
        if isinstance(B, bool) or not isinstance(B, Integral) or B < 1:
            raise ValueError(f"B must be a positive integer. Provided value: {B}.")
        if isinstance(aggregation, str):
            if aggregation not in ("mean", "median"):
                raise ValueError("aggregation must be 'mean', 'median', or a callable.")
        elif not callable(aggregation):
            raise TypeError("aggregation must be 'mean', 'median', or a callable.")

        self.model = make_predictor(model)
        self.B = int(B)
        self.sampler = sampler if sampler is not None else IIDBootstrapSampler()
        self.aggregation = aggregation
        self.fit_function = fit_function

        self.models_: list[Predictor] = []
        self.oob_mask_: TensorLike|None = None
        self.n_train_: int|None = None

    @staticmethod
    def _prediction_vector(
        prediction:TensorLike,
        n_samples:int,
    ) -> TensorLike:
        prediction = ops.convert_to_tensor(prediction)
        shape = tuple(ops.shape(prediction))
        if shape == (n_samples, 1):
            prediction = ops.squeeze(prediction, axis=-1)
        elif shape != (n_samples,):
            raise ValueError(
                f"Expected scalar predictions with shape ({n_samples},) or "
                f"({n_samples}, 1). Received shape: {shape}."
            )
        # Preserve double precision; promote integer and half-precision outputs.
        if not bool(ops.item(ops.all(ops.isfinite(prediction)))):
            raise ValueError("BootstrapEnsemble requires finite predictions.")
        return prediction

    def _check_fitted(self) -> None:
        if not self.models_ or self.oob_mask_ is None:
            raise RuntimeError("BootstrapEnsemble must be fitted before prediction.")

    def _aggregate(self, predictions:TensorLike) -> TensorLike:
        """Reduce the first axis, keeping every remaining prediction axis."""
        if self.aggregation == "mean":
            result = ops.mean(predictions, axis=0)
        elif self.aggregation == "median":
            result = ops.median(predictions, axis=0)
        else:
            result = self.aggregation(predictions, axis=0)
        result = ops.convert_to_tensor(result)
        if tuple(ops.shape(result)) != tuple(ops.shape(predictions))[1:]:
            raise ValueError("aggregation must reduce axis 0 without keeping it.")
        return result

    def _mean_weights(self) -> TensorLike:
        """Give equal weight to the eligible models in each training row."""
        return self.oob_mask_ / ops.sum(self.oob_mask_, axis=1, keepdims=True)

    def fit(
        self,
        X:Iterable[Any],
        y:TensorLike,
        *args:Any,
        **kwargs:Any,
    ) -> Self:
        """
        Fit independent models on bootstrap samples.

        The complete sampling plan is checked for out-of-bag coverage before
        any model is trained. Fitted state is replaced only once all fits
        succeed; a failed refit leaves the previous ensemble available.

        Args:
            X: Training inputs, in original observation order.
            y: Scalar targets with the same number of observations as X.
            *args: Additional arguments forwarded to each fitting function.
            **kwargs: Additional keyword arguments forwarded to each fit.

        Returns:
            The fitted ensemble.

        Raises:
            ValueError: If training data are incompatible, the sampler returns
                the wrong number of samples, or an observation has no OOB model.
            NotImplementedError: If a clone has no fit method and no custom
                fitting function is provided.
        """
        n_samples = len(X)
        if n_samples < 2:
            raise ValueError("BootstrapEnsemble requires at least 2 observations.")
        if len(y) != n_samples:
            raise ValueError("X and y must contain the same number of observations.")
        target_shape = getattr(y, "shape", None)
        if target_shape is None:
            target_shape = ops.shape(ops.convert_to_tensor(y))
        if tuple(target_shape) not in ((n_samples,), (n_samples, 1)):
            raise ValueError("BootstrapEnsemble requires scalar regression targets.")

        samples = list(self.sampler(n_samples=n_samples, n_resamples=self.B))
        if len(samples) != self.B:
            raise ValueError(f"The sampler must return exactly {self.B} samples.")

        # Each column describes the observations excluded from one model.
        oob_mask = ops.stack([
            ops.bincount(sample.oob_indices, minlength=n_samples) > 0
            for sample in samples
        ], axis=1)
        missing = ops.where_1d(ops.sum(ops.cast(oob_mask, "int32"), axis=1) == 0)
        if len(missing):
            raise ValueError(
                f"{len(missing)} training observations have no out-of-bag model. "
                "Increase B or change the sampler configuration."
            )

        models: list[Predictor] = []
        for sample in samples:
            model = make_predictor(clone_model(self.model, clone_weights=False))
            X_fit = tensor_indexing(X, sample.train_indices)
            y_fit = tensor_indexing(y, sample.train_indices)

            if self.fit_function is not None:
                fitted_model = self.fit_function(model, X_fit, y_fit, *args, **kwargs)
                if fitted_model is None:
                    raise TypeError("fit_function must return the fitted predictor.")
                model = make_predictor(fitted_model)
            else:
                fit_method = getattr(model, "fit", None)
                if not callable(fit_method):
                    raise NotImplementedError(
                        "The model has no fit method. Provide a fit_function."
                    )
                fit_method(X_fit, y_fit, *args, **kwargs)
            models.append(model)

        self.models_ = models
        self.oob_mask_ = oob_mask
        self.n_train_ = n_samples
        return self

    def predict_members(self, X:Iterable[Any]) -> TensorLike:
        """
        Evaluate every fitted model once on the supplied inputs.

        Args:
            X: Inputs on which predictions are produced.

        Returns:
            Tensor of scalar predictions with shape (B, n_test).
        """
        self._check_fitted()
        n_samples = len(X)
        return ops.stack([
            self._prediction_vector(model(X), n_samples)
            for model in self.models_
        ], axis=0)

    def predict_oob_training(self, X:Iterable[Any]) -> TensorLike:
        """
        Predict each training observation using only models that excluded it.

        Args:
            X: Original training inputs, in exactly the order supplied to fit.
                Only the length is checked; observation identity is not stored.

        Returns:
            Out-of-bag prediction vector with shape (n_train,).
        """
        self._check_fitted()
        if len(X) != self.n_train_:
            raise ValueError("X must contain all original training observations.")
        predictions = self.predict_members(X)

        if self.aggregation == "mean":
            weights = self._mean_weights(predictions.dtype)
            return ops.sum(weights * ops.transpose(predictions), axis=1)

        # Take the training column first, avoiding an n_train by n_train array.
        result = []
        for i in range(self.n_train_):
            model_indices = ops.where_1d(self.oob_mask_[i])
            eligible = ops.take(predictions[:, i], model_indices, axis=0)
            result.append(self._aggregate(eligible))
        return self._prediction_vector(ops.stack(result, axis=0), self.n_train_)

    def _leave_one_out_predictions(self, predictions:TensorLike) -> TensorLike:
        """Form every observation-specific predictor from cached model outputs."""
        if self.aggregation == "mean":
            return self._mean_weights(predictions.dtype) @ predictions

        result = []
        for i in range(self.n_train_):
            model_indices = ops.where_1d(self.oob_mask_[i])
            eligible = ops.take(predictions, model_indices, axis=0)
            result.append(self._aggregate(eligible))
        return ops.stack(result, axis=0)

    def predict_leave_one_out(self, X:Iterable[Any]) -> TensorLike:
        """
        Evaluate the predictor associated with each excluded training point.

        Row i aggregates the bootstrap models that excluded training point i,
        evaluated on every supplied test input. No additional models are fitted.

        Args:
            X: Inputs on which predictions are produced.

        Returns:
            Tensor with shape (n_train, n_test).

        Note:
            This method materializes the full matrix. Use batches of test
            inputs to limit memory consumption.
        """
        return self._leave_one_out_predictions(self.predict_members(X))

    def predict(self, X:Iterable[Any]) -> TensorLike:
        """
        Aggregate the observation-specific out-of-bag predictors.

        This is the nested aggregation used as the EnbPI point prediction.
        It generally differs from aggregating all bootstrap models uniformly.

        Args:
            X: Inputs on which predictions are produced.

        Returns:
            Prediction vector with shape (n_test,).
        """
        predictions = self.predict_members(X)
        if self.aggregation == "mean":
            # mean(W @ P, axis=0) == mean(W, axis=0) @ P.
            # Avoid allocating the (n_train, n_test) intermediate matrix.
            member_weights = ops.mean(self._mean_weights(predictions.dtype), axis=0)
            result = member_weights @ predictions
        else:
            result = self._aggregate(self._leave_one_out_predictions(predictions))
        return self._prediction_vector(result, len(X))

    def __call__(self, X:Iterable[Any]) -> TensorLike:
        return self.predict(X)

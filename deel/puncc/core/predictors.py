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
Predictor utilities and composite predictor implementations.

This module provides adapters and lightweight predictor structures used by
PUNCC, including predictor normalization, predictor stacking, identity and
lookup predictors, and interoperability with scikit-learn estimators.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Self

from deel.puncc.typing import Predictor, PredictorLike, TensorLike
from deel.puncc.backend import ops
from deel.puncc.cloning import clone_model


class _PredictorAdapter:
    """
    Adapt an object exposing ``predict`` to the PUNCC predictor interface.

    The wrapped object is made callable while preserving access to its attributes and methods.
    """
    def __init__(self, model: PredictorLike) -> None:
        self._model = model

    def __call__(self, X: Iterable[Any], *args: Any, **kwargs: Any) -> Any:
        return self._model.predict(X, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __setattr__(self, name:str, value:Any):
        if name == "_model":
            super().__setattr__(name, value)
        else:
            setattr(self._model, name, value)

    def clone(self, clone_weights: bool = True) -> _PredictorAdapter:
        """
        Clone the wrapped predictor.

        Args:
            clone_weights: Whether learned model parameters should be copied.

        Returns:
            A new adapter containing the cloned model.
        """
        return _PredictorAdapter(clone_model(self._model, clone_weights=clone_weights))

def make_predictor(model: Predictor|PredictorLike) -> Predictor:
    """
    Convert a supported prediction model to the PUNCC predictor interface.

    Callable objects are returned unchanged.
    Objects exposing a ``predict`` method are wrapped in a callable adapter.

    Args:
        model: Prediction model to adapt.

    Returns:
        A callable predictor.

    Raises:
        TypeError: If the model exposes neither ``__call__`` nor ``predict``.
    """
    if callable(model):
        return model
    if hasattr(model, "predict") and callable(model.predict):
        predictor = _PredictorAdapter(model)
        return predictor
    #TODO : maybe check for other types of predictors ? like predict_proba models ?
    raise TypeError("The provided model neither have __call__ nor predict method.")

class MultiPredictorStack:
    """
    Combine several predictors by stacking their outputs.

    Each predictor receives the same input samples.
    Their predictions are stacked along a new last axis.

    Can be used for multi-output regression using several models, quantil regression, mean/variance estimation...
    """

    def __init__(self, *models:Predictor|PredictorLike):
        self.models = [make_predictor(m) for m in models]

    def clone(self, clone_weights:bool=True)->MultiPredictorStack:
        """
        Clone all predictors in the stack.

        Args:
            clone_weights: Whether learned model parameters should be copied.

        Returns:
            A new predictor stack containing the cloned models.
        """
        return self.__class__(*[clone_model(model, clone_weights=clone_weights) for model in self.models])

    def __call__(self, X:Iterable[Any])->TensorLike:
        predictions = [model(X) for model in self.models]
        return ops.stack(predictions, axis=-1)
    
    def fit(self,
            X_train:Iterable[Any],
            y_train:TensorLike):
        """
        Fit every predictor in the stack on the same dataset.

        Args:
            X_train: Training input samples.
            y_train: Training targets.

        Returns:
            The fitted predictor stack.

        Raises:
            NotImplementedError: If one of the predictors does not expose a ``fit`` method.
        """
        # TODO : how to deal with complex models that does not exposes fit function as a method ?
        for model in self.models:
            fit_method = getattr(model, "fit", None)
            if callable(fit_method):
                fit_method(X_train, y_train)
            else:
                raise NotImplementedError("One of the models does not have a fit method. Please provide pretrained models or expose a fit method.")
        return self

def stack_predictors(*models:Predictor|PredictorLike)->MultiPredictorStack:
    """
    Create a MultiPredictorStack with several models.

    Args:
        *models: Predictors to combine.

    Returns:
        A predictor stacking model outputs along the last axis.
    """
    return MultiPredictorStack(*models)

class MeanVarPredictor(MultiPredictorStack):
    """
    Combine a mean predictor and a dispersion predictor.

    The mean model is fitted directly on the training targets.
    The dispersion model is then fitted on dispersion targets computed from the mean-model predictions and the observed targets.
    By default, the dispersion target is the absolute residual between the mean prediction and the observed target.
    """
    def __init__(self, mean_model:Predictor|PredictorLike,
                 dispersion_model:Predictor|PredictorLike):
        super().__init__(mean_model, dispersion_model)

    def dispersion_estimation(self, mu:TensorLike, y:TensorLike)->TensorLike:
        """
        Compute dispersion targets from predictions and observations.
        Can be overloaded to implement other dispersion targets (e.g., squared residuals, quantile loss, etc.)

        Args:
            mu: Predictions produced by the mean model.
            y: Observed targets.

        Returns:
            Absolute residuals between the predictions and targets.
        """
        return ops.abs(mu - y)

    def fit(self,
            X_train:Iterable[Any],
            y_train:TensorLike):
        """
        Fit the mean model and then the dispersion model.

        The mean model is first fitted on ``(X_train, y_train)``.
        Its predictions on the training samples are then used to compute the targets required to train the dispersion model.

        Args:
            X_train: Training input samples.
            y_train: Training targets.

        Returns:
            The fitted predictor.

        Raises:
            NotImplementedError: If either model does not expose a ``fit`` method.
        """
        for model in self.models:
            if not callable(getattr(model, "fit", None)):
                raise NotImplementedError("One of the models does not have a fit method. Please provide pretrained models or expose a fit method.")
        fit_method_0 = getattr(self.models[0], "fit")
        fit_method_1 = getattr(self.models[1], "fit")
        
        fit_method_0(X_train, y_train)
        mu_pred = self.models[0](X_train)
        fit_method_1(X_train, self.dispersion_estimation(mu_pred, y_train) )
        return self

class IDPredictor:
    """
    Identity predictor returning its input unchanged.
    """
    def fit(self,
            X_train:Iterable[Any],
            y_train:TensorLike):
        """
        No-op fitting step.

        Args:
            X_train: Unused training inputs.
            y_train: Unused training targets.

        Returns:
            The predictor itself.
        """
        return self
    
    def predict(self, X:Iterable[Any])->TensorLike:
        """
        Return the input unchanged.
        """
        return X

    def __call__(self, X:Iterable[Any])->TensorLike:
        """
        Return the input unchanged.
        """
        return X

class LookupTablePredictor:
    """
    Predict targets by exact lookup of previously fitted samples.

    The predictor stores training samples and their associated targets.
    Prediction succeeds only when each requested sample appears exactly once in the stored lookup table.
    """
    def __init__(self)->None:
        self.X = None
        self.y = None

    def fit(self,
            X_train:Iterable[Any],
            y_train:TensorLike):
        """
        Store samples and their associated targets.

        Args:
            X_train: Samples used as lookup keys.
            y_train: Targets associated with the samples.

        Returns:
            The fitted predictor.
        """
        self.X = ops.asarray(X_train)
        self.y = ops.asarray(y_train)
        return self
    
    def predict(
        self,
        X: Iterable[Any],
    ) -> TensorLike:
        """
        Retrieve targets associated with the requested samples.
        Samples are matched using exact element-wise equality.

        Args:
            X: Samples whose targets should be retrieved.

        Returns:
            Targets corresponding to the requested samples.

        Raises:
            RuntimeError: If the predictor has not been fitted.
            ValueError: If a requested sample is absent from the lookup table or appears more than once.
        """
        if self.X is None or self.y is None:
            raise RuntimeError(
                "LookupTablePredictor must be fitted before prediction."
            )

        X = ops.asarray(X)

        predictions = []

        for x in X:
            matches = ops.all(
                ops.equal(self.X, x),
                axis=-1,
            )

            indices = ops.where_1d(matches)

            if len(indices) == 0:
                raise ValueError(
                    "At least one requested sample was not found "
                    "in the lookup table."
                )

            if len(indices) > 1:
                raise ValueError(
                    "A requested sample appears multiple times "
                    "in the lookup table."
                )

            predictions.append(
                ops.take(self.y, indices[0], axis=0)
            )

        return ops.stack(predictions, axis=0)

    __call__ = predict

class SklearnWrapper:
    """
    Adapt a scikit-learn estimator to the active tensor backend.

    Inputs are converted to NumPy arrays before calls to the wrapped estimator.
    Predictions are converted back to tensors using the active PUNCC backend.

    Args:
        model: Scikit-learn compatible estimator exposing ``fit`` and ``predict`` methods.
    """
    def __init__(self, model) -> None:
        self.model = model

    def fit(self, X: TensorLike, y: TensorLike) -> Self:
        """
        Fit the wrapped estimator.

        Input tensors are converted to NumPy arrays before being passed to scikit-learn model.

        Args:
            X: Training input samples.
            y: Training targets.

        Returns:
            The fitted wrapper.
        """
        X_np = ops.convert_to_numpy(X)
        y_np = ops.convert_to_numpy(y)
        self.model.fit(X_np, y_np)
        return self

    def predict(self, X: TensorLike) -> TensorLike:
        """
        Run prediction with the wrapped estimator.

        Input tensors are converted to NumPy arrays and the resulting predictions are converted back to the active tensor backend.

        Args:
            X: Input samples.

        Returns:
            Predictions represented using the active tensor backend.
        """
        X_np = ops.convert_to_numpy(X)
        y_pred = self.model.predict(X_np)
        return ops.convert_to_tensor(y_pred)

    def clone(self, clone_weights: bool = True) -> Self:
        """
        Clone the wrapped estimator.

        Args:
            clone_weights: Whether learned estimator state should be copied.

        Returns:
            A new wrapper containing the cloned estimator.
        """
        return type(self)(
            clone_model(self.model, clone_weights=clone_weights)
        )

    def __call__(self, X: TensorLike) -> TensorLike:
        return self.predict(X)

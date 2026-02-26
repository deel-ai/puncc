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
Basic definitions of type aliases and protocols used by conformal prediction methods
"""
from typing import TYPE_CHECKING, Any, Union, TypeAlias, Protocol, runtime_checkable, Callable
from collections.abc import Iterable, Sequence
from deel.puncc.cloning import clone_model

# if TYPE_CHECKING:
#     import numpy as _np
#     import torch as _torch
#     import tensorflow as _tf
#     from jax import Array as _JaxArray
#     TensorLike: TypeAlias = Union[_np.ndarray, _torch.Tensor, _tf.Tensor, _JaxArray]
# else:
#     TensorLike = Any

TensorLike:TypeAlias = Any

@runtime_checkable
class Predictor(Protocol):
    def __call__(self, X: Iterable[Any], *args, **kwargs) -> TensorLike:
        ...

@runtime_checkable
class LambdaPredictor(Predictor, Protocol):
    def __call__(self, X:Iterable[Any], lambd:float, *args, **kwargs) -> Any:
        ...

@runtime_checkable
class Fitable(Protocol):
    def fit(self, X: Iterable[Any], y: TensorLike, *args, **kwargs) -> Any:
        ...

@runtime_checkable
class PredictorLike(Protocol):
    def predict(self, X: Iterable[Any], *args, **kwargs) -> TensorLike:
        ...
    
class _PredictorAdapter:
    """Wraps a .predict(...) provider into a callable."""
    def __init__(self, model: PredictorLike) -> None:
        self._model = model

    def __call__(self, X: Iterable[Any], *args: Any, **kwargs: Any) -> Any:
        return self._model.predict(X, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)
    
    def __setattr__(self, name, value):
        if name == "_model":
            super().__setattr__(name, value)
        else:
            setattr(self._model, name, value)

    def clone(self):
        return _PredictorAdapter(clone_model(self._model))

def make_predictor(model: Union[Predictor, PredictorLike]) -> Predictor:
    if callable(model):
        return model
    if hasattr(model, "predict") and callable(model.predict):
        predictor = _PredictorAdapter(model)
        return predictor
    raise TypeError("The provided model neither have __call__ nor predict method.")

# A nonconformity score function takes as input the true labels and the model's predictions, and outputs a sequence of nonconformity scores.
NCScoreFunction:TypeAlias = Callable[[TensorLike, TensorLike], Sequence[float]]

# A prediction set function takes as input the model's predictions and a threshold (float or tensor), and outputs a sequence of prediction sets (e.g., list of sets of labels for classification, list of intervals for regression).
PredSetFunction:TypeAlias = Callable[[TensorLike, float|TensorLike], Sequence[Any]]

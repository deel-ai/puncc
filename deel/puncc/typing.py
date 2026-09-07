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
from __future__ import annotations
from typing import Any, TypeVar, Union, TypeAlias, Protocol, runtime_checkable
from collections.abc import Iterable, Callable
from deel.puncc.cloning import clone_model

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     import numpy as _np
#     import torch as _torch
#     import tensorflow as _tf
#     from jax import Array as _JaxArray
#     TensorLike: TypeAlias = Union[_np.ndarray, _torch.Tensor, _tf.Tensor, _JaxArray]
# else:
#     TensorLike = Any

TensorLike:TypeAlias = Any
TPrediction_co = TypeVar(
    "TPrediction_co",
    covariant=True,
)


@runtime_checkable
class Predictor(Protocol[TPrediction_co]):
    def __call__(self, X: Iterable[Any], *args:Any, **kwargs:Any) -> TPrediction_co:
        ...

@runtime_checkable
class Fittable(Protocol):
    def fit(self, X: Iterable[Any], y: TensorLike, *args:Any, **kwargs:Any) -> Any:
        ...

@runtime_checkable
class PredictorLike(Protocol):
    def predict(self, X: Iterable[Any], *args:Any, **kwargs:Any) -> TensorLike:
        ...

# A nonconformity score function takes as input the true labels and the model's predictions, and outputs a sequence of nonconformity scores.
NCScoreFunction: TypeAlias = Callable[
    [TensorLike, TensorLike],
    TensorLike,
]
# A prediction set function takes as input the model's predictions and a threshold (float or tensor), and outputs a sequence of prediction sets (e.g., list of sets of labels for classification, list of intervals for regression).
PredSetFunction: TypeAlias = Callable[
    [TensorLike, float | TensorLike],
    Any,
]

# A weight function takes as input a sequence of features and outputs a sequence of weights (float or tensor) for each instance in the input sequence.
WeightFunction = Callable[
    [Iterable[Any]],
    TensorLike,
]

# A fit function takes as input a model, features and targets and returns a predictor fitted to given dataset
FitFunction: TypeAlias = Callable[[Predictor, Iterable[Any], Iterable[Any]], Predictor]

# A function that takes a float or tensor and returns a float or tensor, used for alpha correction in conformal prediction methods.
AlphaCorrection:TypeAlias = Callable[[float|TensorLike], float|TensorLike]
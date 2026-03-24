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
Definitions of some specific perdictor structures
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Iterator, TypeVar, overload
from collections.abc import Iterable
import warnings
from deel.puncc.typing import Predictor, PredictorLike, TensorLike, make_predictor
from deel.puncc import ops
from deel.puncc.cloning import clone_model

class MultiPredictorStack(ABC):
    def __init__(self, models:Iterable[Predictor|PredictorLike],
                 *,
                 expand_1d:bool=True):
        self.models = [make_predictor(m) for m in models]
        self.expand_1d = expand_1d

    def clone(self, clone_weights:bool=True)->MultiPredictorStack:
        return MultiPredictorStack(models=[clone_model(model, clone_weights=clone_weights) for model in self.models])

    def __call__(self, X:Iterable[Any])->TensorLike:
        predictions = [model(X) for model in self.models]

        if self.expand_1d:
            predictions = [pred if len(ops.shape(pred)) != 1 else ops.expand_dims(pred, axis=-1) for pred in predictions]

        return ops.stack([model(X) for model in self.models], axis=-1)
    
    def fit(self,
            X_train:Iterable[Any],
            y_train:TensorLike):
        for model in self.models:
            if callable(getattr(model, "fit", None)):
                model.fit(X_train, y_train)
                continue
            warnings.warn("One of the models does not have a fit method. Please provide pretrained models or expose a fit method.")
        return self

def stack_predictors(*models:Predictor|PredictorLike)->MultiPredictorStack:
    return MultiPredictorStack(models=models)

class MeanVarPredictor(MultiPredictorStack):
    def __init__(self, mean_model:Predictor|PredictorLike,
                 dispersion_model:Predictor|PredictorLike):
        super().__init__(models=[mean_model, dispersion_model])

    #@abstractmethod
    def dispertion_estimation(self, mu:TensorLike, y:TensorLike)->TensorLike:
        #...
        return ops.abs(mu - y)


    def fit(self,
            X_train:Iterable[Any],
            y_train:TensorLike):
        for model in self.models:
            if not callable(getattr(model, "fit", None)):
                raise NotImplementedError("One of the models does not have a fit method. Please provide pretrained models or expose a fit method.")
        self.models[0].fit(X_train, y_train)
        mu_pred = self.models[0](X_train)
        self.models[1].fit(X_train, self.dispertion_estimation(mu_pred, y_train) )
        return self

# class MeanScalePredictor(MeanDispertionPredictor):
#     def dispertion_estimation(self, mu:TensorLike, y:TensorLike)->TensorLike:
#         return ops.abs(mu - y)

# class MeanVarPredictor(MeanDispertionPredictor):
#     def dispertion_estimation(self, mu:TensorLike, y:TensorLike)->TensorLike:
#         return ops.square(mu - y)

class IDPredictor():
    def fit(self,
            X_train:Iterable[Any],
            y_train:TensorLike):
        return self
    
    def predict(self, X:Iterable[Any])->TensorLike:
        return X

    def __call__(self, X:Iterable[Any])->TensorLike:
        return X

class LookupTablePredictor():
    def __init__(self, *args, **kwargs):
        self.X = None
        self.y = None

    def fit(self,
            X_train:Iterable[Any],
            y_train:TensorLike):
        self.X = ops.asarray(X_train)
        self.y = ops.asarray(y_train)
        return self
    
    def predict(self, X:Iterable[Any])->TensorLike:
        X = ops.asarray(X)
        indices = ops.argwhere(ops.isin(self.X, X)).squeeze()
        return self.y[indices]
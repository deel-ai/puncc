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
from abc import ABC
from collections.abc import Iterable
from typing import Any
from deel.puncc.typing import Predictor, PredictorLike, TensorLike, make_predictor
from deel.puncc import ops
from deel.puncc.cloning import clone_model

class MultiPredictorStack(ABC):
    def __init__(self, *models:Predictor|PredictorLike,
                 expand_1d:bool=True):
        self.models = [make_predictor(m) for m in models]
        self.expand_1d = expand_1d

    def clone(self, clone_weights:bool=True)->MultiPredictorStack:
        return self.__class__(*[clone_model(model, clone_weights=clone_weights) for model in self.models], expand_1d=self.expand_1d)

    def __call__(self, X:Iterable[Any])->TensorLike:
        predictions = [model(X) for model in self.models]

        if self.expand_1d:
            predictions = [pred if len(ops.shape(pred)) != 1 else ops.expand_dims(pred, axis=-1) for pred in predictions]

        return ops.stack(predictions, axis=-1)
    
    def fit(self,
            X_train:Iterable[Any],
            y_train:TensorLike):
        for model in self.models:
            if callable(getattr(model, "fit", None)):
                model.fit(X_train, y_train)
            else:
                raise NotImplementedError("One of the models does not have a fit method. Please provide pretrained models or expose a fit method.")
        return self

def stack_predictors(*models:Predictor|PredictorLike)->MultiPredictorStack:
    return MultiPredictorStack(*models)

class MeanVarPredictor(MultiPredictorStack):
    def __init__(self, mean_model:Predictor|PredictorLike,
                 dispersion_model:Predictor|PredictorLike,
                 expand_1d:bool=True):
        super().__init__(mean_model, dispersion_model, expand_1d=expand_1d)

    #@abstractmethod
    def dispersion_estimation(self, mu:TensorLike, y:TensorLike)->TensorLike:
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
        self.models[1].fit(X_train, self.dispersion_estimation(mu_pred, y_train) )
        return self

# class MeanScalePredictor(MeanDispersionPredictor):
#     def dispersion_estimation(self, mu:TensorLike, y:TensorLike)->TensorLike:
#         return ops.abs(mu - y)

# class MeanVarPredictor(MeanDispersionPredictor):
#     def dispersion_estimation(self, mu:TensorLike, y:TensorLike)->TensorLike:
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
    
    def predict(
        self,
        X: Iterable[Any],
    ) -> TensorLike:
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
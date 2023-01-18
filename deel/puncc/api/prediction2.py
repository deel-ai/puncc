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
This module provides standard wrappings for DL/ML models.
"""
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from typing import Any
from typing import Callable
from typing import Iterable

import numpy as np

from deel.puncc.api.calibration2 import NonConformityScores
from deel.puncc.api.utils import supported_types_check


class BasePredictor:
    """Interface of a base predictor class.

    :param Any model: prediction model
    :param bool is_trained: boolean flag that informs if the model is pre-trained
    :param dict compile_kwargs: configuration in case the model needs to be
                                compiled (for example Keras models).
    """

    def __init__(self, model: Any, is_trained: bool = False, **compile_kwargs):
        self.model = model
        self.is_trained = is_trained
        self.compile_kwargs = compile_kwargs
        if self.compile_kwargs:
            self.model.compile(**self.compile_kwargs)

    def fit(self, X: Iterable, y: Iterable, **kwargs) -> None:
        """Fit model to the training data.

        :param ndarray|DataFrame|Tensor X: train features
        :param ndarray|DataFrame|Tensor y: train labels
        :param dict kwargs: fit configuration to be passed to the model's fit method
        """
        self.model.fit(X, y, **kwargs)
        self.is_trained = True

    def predict(self, X: Iterable, **kwargs) -> Iterable:
        """Compute predictions on new examples.

        :param ndarray|DataFrame|Tensor X: new examples' features.
        :param dict kwargs: predict configuration to be passed to the model's predict method.
        :returns: y_pred.
        :rtype: ndarray
        """
        # Remove axis of length one to avoid broadcast in computation
        # of non-conformity scores
        return np.squeeze(self.model.predict(X, **kwargs))

    def copy(self):
        """Returns a copy of the predictor. The underlying model is either
        cloned (Keras model) or deepcopied (sklearn and similar models).

        :returns: copy_predictor.
        :rtype: BasePredictor
        """
        try:
            model = deepcopy(self.model)
        except Exception as e:
            # print(e) ## For debug
            try:
                import tensorflow as tf

                model = tf.keras.models.clone_model(self.model)
            except Exception as e:
                raise RuntimeError(e)

        copy_predictor = BasePredictor(model, **self.compile_kwargs)
        return copy_predictor


class LocallyAdaptivePredictor:
    def __init__(
        self,
        mu_model: Any,
        var_model: Any,
        is_trained: bool = False,
        compile_args1=dict(),
        compile_args2=dict(),
    ):
        self.mu_model = mu_model
        self.var_model = var_model
        self.is_trained = is_trained
        self.compile_args1 = compile_args1
        self.compile_args2 = compile_args2
        if len(self.compile_args1.keys()) != 0:
            self.mu_model.compile(**self.compile_args1)
        if len(self.compile_args2.keys()) != 0:
            self.var_model.compile(**self.compile_args2)

    def fit(self, X: Iterable, y: Iterable, dictargs1=dict(), dictargs2=dict()) -> None:
        """Fit model to the training data.

        :param ndarray|DataFrame|Tensor X: train features
        :param ndarray|DataFrame|Tensor y: train labels
        :param dict dictargs1: fit configuration to be passed to the mu_model's fit method
        :param dict dictargs2: fit configuration to be passed to the var_model's fit method
        """
        self.mu_model.fit(X, y, **dictargs1)
        mu_pred = self.mu_model.predict(X)
        mads = NonConformityScores.MAD(mu_pred, y)
        self.var_model.fit(X, mads, **dictargs2)
        self.is_trained = True

    def predict(self, X, dictargs1=dict(), dictargs2=dict()):
        mu_pred = self.mu_model.predict(X, **dictargs1)
        var_pred = self.var_model.predict(X, **dictargs2)
        supported_types_check(mu_pred, var_pred)
        if isinstance(mu_pred, np.ndarray):
            Y_pred = np.column_stack((mu_pred, var_pred))
        else:  # In case the models return something other than an np.ndarray
            raise NotImplementedError("Predicted values must be of type numpy.ndarray.")
        return np.squeeze(Y_pred)

    def copy(self):
        """Returns a copy of the predictor. The underlying models are either
        cloned (Keras model) or deepcopied (sklearn and similar models).

        :returns: copy_predictor.
        :rtype: LocallyAdaptivePredictor
        """
        try:
            mu_model = deepcopy(self.mu_model)
        except Exception as e:
            try:
                import tensorflow as tf

                mu_model = tf.keras.models.clone_model(self.mu_model)
            except Exception as e:
                raise RuntimeError(e)

        try:
            var_model = deepcopy(self.var_model)
        except Exception as e:
            try:
                import tensorflow as tf

                var_model = tf.keras.models.clone_model(self.var_model)
            except Exception as e:
                raise RuntimeError(e)

        copy_predictor = LocallyAdaptivePredictor(
            mu_model,
            var_model,
            compile_args1=self.compile_args1,
            compile_args2=self.compile_args2,
        )
        return copy_predictor


class QuantilePredictor:
    def __init__(
        self,
        q_lo_model: Any,
        q_hi_model: Any,
        is_trained: bool = False,
        compile_args1=dict(),
        compile_args2=dict(),
    ):
        self.q_lo_model = q_lo_model
        self.q_hi_model = q_hi_model
        self.is_trained = is_trained
        self.compile_args1 = compile_args1
        self.compile_args2 = compile_args2
        if len(self.compile_args1.keys()) != 0:
            self.q_lo_model.compile(**self.compile_args1)
        if len(self.compile_args2.keys()) != 0:
            self.q_hi_model.compile(**self.compile_args2)

    def fit(self, X: Iterable, y: Iterable, dictargs1=dict(), dictargs2=dict()) -> None:
        """Fit model to the training data.

        :param ndarray|DataFrame|Tensor X: train features
        :param ndarray|DataFrame|Tensor y: train labels
        :param dict dictargs1: fit configuration to be passed to the q_lo_model's fit method
        :param dict dictargs2: fit configuration to be passed to the q_hi_model's fit method
        """
        self.q_lo_model.fit(X, y, **dictargs1)
        self.q_hi_model.fit(X, y, **dictargs2)
        self.is_trained = True

    def predict(self, X, dictargs1=dict(), dictargs2=dict()):
        q_lo_pred = self.q_lo_model.predict(X, **dictargs1)
        q_hi_pred = self.q_hi_model.predict(X, **dictargs2)
        supported_types_check(q_lo_pred, q_hi_pred)
        if isinstance(q_lo_pred, np.ndarray):
            Y_pred = np.column_stack((q_lo_pred, q_hi_pred))
        else:  # In case the models return something other than an np.ndarray
            raise NotImplementedError("Predicted values must be of type numpy.ndarray.")
        return np.squeeze(Y_pred)

    def copy(self):
        """Returns a copy of the predictor. The underlying models are either
        cloned (Keras model) or deepcopied (sklearn and similar models).

        :returns: copy_predictor.
        :rtype: LocallyAdaptivePredictor
        """
        try:
            q_lo_model = deepcopy(self.q_lo_model)
        except Exception as e:
            try:
                import tensorflow as tf

                q_lo_model = tf.keras.models.clone_model(self.q_lo_model)
            except Exception as e:
                raise RuntimeError(e)

        try:
            q_hi_model = deepcopy(self.q_hi_model)
        except Exception as e:
            try:
                import tensorflow as tf

                q_hi_model = tf.keras.models.clone_model(self.q_hi_model)
            except Exception as e:
                raise RuntimeError(e)

        copy_predictor = QuantilePredictor(
            q_lo_model,
            q_hi_model,
            compile_args1=self.compile_args1,
            compile_args2=self.compile_args2,
        )
        return copy_predictor

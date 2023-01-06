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

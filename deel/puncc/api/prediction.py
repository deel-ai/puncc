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
from typing import Callable

import numpy as np


class BasePredictor(ABC):
    """Interface of a base predictor class.

    :param bool is_trained: boolean flag that informs if the models are pre-trained
    """

    def __init__(self, is_trained: bool = False):
        self.is_trained = is_trained

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit model to the training data.

        :param ndarray X: train features
        :param ndarray y: train labels
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute predictions on new examples.

        :param ndarray X: new examples' features.
        :returns: y_pred, y_lower, y_upper, var_pred.
        :rtype: tuple[ndarray, ndarray, ndarray, ndarray]
        """
        return None, None, None, None


class MeanPredictor(BasePredictor):
    """Wrapper of conditional mean models.

    :param callable mu_model: conditional mean model.
    :param bool is_trained: boolean flag that informs if the models are pre-trained.
    """

    def __init__(self, mu_model: Callable, is_trained=False):
        self.mu_model = mu_model
        super().__init__(is_trained)

    def fit(self, X, y, **kwargs):
        self.mu_model.fit(X, y, **kwargs)
        self.is_trained = True

    def predict(self, X):
        y_pred = self.mu_model.predict(X)
        return y_pred, None, None, None


class MeanVarPredictor(BasePredictor):
    """Wrapper of joint conditional mean and mean absolute dispertion models.

    :param callable mu_model: conditional mean model.
    :param callable var_model: mean absolute dispertion model.
    :param bool is_trained: boolean flag that informs if the models are pre-trained.
    """

    def __init__(self, mu_model: Callable, var_model: Callable, is_trained=False):
        self.mu_model = mu_model
        self.var_model = var_model
        super().__init__(is_trained)

    def fit(self, X, y, **kwargs):
        self.mu_model.fit(X, y, **kwargs)
        y_pred = self.mu_model.predict(X)
        residual = np.abs(y - y_pred)
        self.var_model.fit(X, residual, **kwargs)
        self.is_trained = True

    def predict(self, X):
        y_pred = self.mu_model.predict(X)
        var_pred = self.var_model.predict(X)
        y_pred_lo, y_pred_hi = None, None
        return y_pred, y_pred_lo, y_pred_hi, var_pred


class QuantilePredictor(BasePredictor):
    """Wrapper of upper and lower quantiles models.

    :param callable q_lo_model: lower quantile model.
    :param callable q_hi_model: upper quantile model.
    :param bool is_trained: boolean flag that informs if the models are pre-trained.
    """

    def __init__(self, q_lo_model: Callable, q_hi_model: Callable, is_trained=False):
        self.q_lo_model = q_lo_model
        self.q_hi_model = q_hi_model
        super().__init__(is_trained)

    def fit(self, X, y, **kwargs):
        self.q_lo_model.fit(X, y, **kwargs)
        self.q_hi_model.fit(X, y, **kwargs)
        self.is_trained = True

    def predict(self, X):
        y_pred_lo = self.q_lo_model.predict(X)
        y_pred_hi = self.q_hi_model.predict(X)
        return None, y_pred_lo, y_pred_hi, None

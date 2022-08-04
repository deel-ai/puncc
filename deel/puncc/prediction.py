"""
This module provides wrappings for ML models.
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePredictor(ABC):
    """Interface of a base predictor class.

    Attributes:
        is_trained: boolean flag that informs if the models are pre-trained
    """

    def __init__(self, is_trained: bool = False):
        self.is_trained = is_trained

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit model to the training data.
        Args:
            X: train features
            y: train labels
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute predictions on new examples.
        Args:
            X: new examples' features
        Returns:
            y_pred, y_lower, y_upper, var_pred
        """
        return None, None, None, None


class MeanPredictor(BasePredictor):
    """Wrapper of conditional mean models.

    Attributes:
        mu_model: conditional mean model
        is_trained: boolean flag that informs if the models are pre-trained
    """

    def __init__(self, mu_model, is_trained=False):
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

    Attributes:
        mu_model: conditional mean model
        var_model: mean absolute dispertion model
        is_trained: boolean flag that informs if the models are pre-trained
    """

    def __init__(self, mu_model, var_model, is_trained=False):
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

    Attributes:
        q_lo_model: lower quantile model
        q_hi_model: upper quantile model
        is_trained: boolean flag that informs if the models are pre-trained
    """

    def __init__(self, q_lo_model, q_hi_model, is_trained=False):
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

"""
This module provides wrappings for ML models.
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePredictor(ABC):
    """Abstract structure of a base predictor class.

    Attributes:
        is_trained: boolean flag that informs if the models are pre-trained
    """

    def __init__(self, is_trained=False):
        self.is_trained = is_trained

    def _format(self, X: np.array, y: np.array):
        """Format data to be consistent with the fit and predict methods.
        Args:
            X: features
            y: labels
        Returns:
            formated_X, formated_y
        """
        return X, y

    @abstractmethod
    def fit(self, X: np.array, y: np.array, **kwargs) -> None:
        """Fit model to the training data.
        Args:
            X: train features
            y: train labels
        """
        pass

    @abstractmethod
    def predict(self, X: np.array, **kwargs):
        """Compute predictions on new examples.
        Args:
            X: new examples' features
        Returns:
            y_pred, y_lower, y_upper, sigma_pred
        """
        pass


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
        sigma_model: mean absolute dispertion model
        is_trained: boolean flag that informs if the models are pre-trained
    """

    def __init__(self, mu_model, sigma_model, is_trained=False):
        self.mu_model = mu_model
        self.sigma_model = sigma_model
        super().__init__(is_trained)

    def fit(self, X, y, **kwargs):
        self.mu_model.fit(X, y, **kwargs)
        y_pred = self.mu_model.predict(X)
        residual = np.abs(y - y_pred)
        self.sigma_model.fit(X, residual, **kwargs)
        self.is_trained = True

    def predict(self, X):
        y_pred = self.mu_model.predict(X)
        sigma_pred = self.sigma_model.predict(X)
        y_pred_lower, y_pred_upper = None, None
        return y_pred, y_pred_lower, y_pred_upper, sigma_pred


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
        y_pred_lower = self.q_lo_model.predict(X)
        y_pred_upper = self.q_hi_model.predict(X)
        return None, y_pred_lower, y_pred_upper, None

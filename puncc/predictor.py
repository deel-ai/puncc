"""
This module provides wrappings for ML models.
"""

from abc import ABC, abstractmethod
import numpy as np

class BasePredictor(ABC):
    """Abstract structure of a base predictor class."""

    def __init__(self):
        self.is_trained = False

    def _format(self, X: np.array, y: np.array):
        """Format data to be consistent with the the fit and predict methods.
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
    def __init__(self, mu_model):
        super().__init__()
        self.name = "MeanPredictor"
        self.mu_model = mu_model

    def _format(self, X, y):
        return X, y

    def fit(self, X, y, **kwargs):
        self.mu_model.fit(X, y)
        self.is_trained = True

    def predict(self, X, **kwargs):
        y_pred = self.mu_model.predict(X)
        return y_pred, None, None, None


class MeanVarPredictor(BasePredictor):
    def __init__(self, mu_model, sigma_model, gaussian=False):
        super().__init__()
        self.name = "MeanVarPredictor"
        self.mu_model = mu_model
        self.sigma_model = sigma_model
        self.gaussian = gaussian

    def fit(self, X, y, **kwargs):
        self.mu_model.fit(X, y)
        y_pred = self.mu_model.predict(X)
        residual = np.abs(y - y_pred)
        self.sigma_model.fit(X, residual)
        self.is_trained = True
        
    def predict(self, X, **kwargs):
        y_pred = self.mu_model.predict(X)
        sigma_pred = self.sigma_model.predict(X)
        y_pred_lower, y_pred_upper = None, None
        return y_pred, y_pred_lower, y_pred_upper, sigma_pred


class QuantilePredictor(BasePredictor):
    def __init__(self, q_lo_model, q_hi_model):
        """
        Args:
            q_lo_model: lower quantile model
            q_hi_model: upper quantile model
        """
        super().__init__()
        self.name = "QuantilePredictor"
        self.q_lo_model = q_lo_model
        self.q_hi_model = q_hi_model

    def fit(self, X, y, **kwargs):
        self.q_lo_model.fit(X, y)
        self.q_hi_model.fit(X, y)
        self.is_trained = True

    def predict(self, X, **kwargs):
        y_pred_lower = self.q_lo_model.predict(X)
        y_pred_upper = self.q_hi_model.predict(X)
        return None, y_pred_lower, y_pred_upper, None

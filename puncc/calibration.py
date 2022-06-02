"""
This module implements different calibration methods.
"""
from abc import ABC, abstractmethod
from tkinter import E
from puncc.utils import EPSILON, w_quantile
import numpy as np


class Calibrator(ABC):
    """Abstract structure of a Calibrator class."""

    def __init__(self, w_estimator):
        self._residuals = None
        self._w_estimator = w_estimator
        self.weights = None

    def compute_weights(self, X):
        """Compute and normalizes weight of the nonconformity distribution
        based on the provided w estimator.
        Args:
            X: features array
        """
        if self._w_estimator is None:
            raise RuntimeError(
                "Weight estimator is None."
                + "Please provide a valid function."
            )
        w = self._w_estimator(X)
        len_calib = len(self._w_calib)
        sum_w_calib = np.sum(self._w_calib)
        w_norm = np.zeros((len(X), len_calib + 1))
        for i in range(len(X)):
            w_norm[i, :len_calib] = self._w_calib / (sum_w_calib + w[i])
            w_norm[i, len_calib] = w[i] / (sum_w_calib + w[i])
        return w_norm

    @abstractmethod
    def estimate(
        self,
        y_true: np.array,
        y_pred: np.array,
        y_pred_lower: np.array,
        y_pred_upper: np.array,
        X: np.array = None,
    ) -> None:
        """Compute residuals on calibration set.
        Args:
            y_true: true values
            y_pred: predicted values
            y_pred_lower: lower bound of the prediction interval
            y_pred_upper: upper bound of the prediction interval
            X: features array
        """
        pass

    @abstractmethod
    def calibrate(
        self,
        y_pred: np.array,
        y_pred_lower: np.array,
        y_pred_upper: np.array,
        alpha: float,
        X: np.array = None,
    ):
        """Compute calibrated prediction intervals for new examples.
        Args:
            y_pred: predicted values
            y_pred_lower: lower bound of the prediction interval
            y_pred_upper: upper bound of the prediction interval
            alpha: maximum miscoverage target
            X: features array
        Returns:
            y_lower, y_upper
        """
        pass


class MeanCalibrator(Calibrator):
    def __init__(self, w_estimator=None):
        super().__init__(w_estimator)

    def estimate(
        self,
        y_true: np.array,
        y_pred: np.array,
        X: np.array = None,
        *args,
        **kwargs
    ) -> None:
        """Compute residuals on calibration set.
        Args:
            y_true: true values
            y_pred: predicted values
        """
        self._residuals = np.abs(y_true - y_pred)
        if self._w_estimator is not None:
            self._w_calib = self._w_estimator(X)

    def calibrate(
        self,
        y_pred: np.array,
        alpha: float,
        X: np.array = None,
        *args,
        **kwargs
    ):
        """Compute calibrated prediction intervals for new examples.
        Args:
            y_pred: predicted values
            alpha: maximum miscoverage target
        Returns:
            y_lower, y_upper
        """
        if self._w_estimator is None:
            residuals_Q = np.quantile(
                self._residuals, (1 - alpha) * (1 + 1 / len(self._residuals))
            )
        else:
            self.weights = self.compute_weights(X)
            residuals_Q = w_quantile(
                np.concatenate((self._residuals, [np.inf])),
                1 - alpha,
                w=self.weights,
            )
            print(self.weights)
        y_pred_lower = y_pred - residuals_Q
        y_pred_upper = y_pred + residuals_Q
        return y_pred_lower, y_pred_upper


class MeanVarCalibrator(Calibrator):
    def __init__(self, w_estimator=None):
        super().__init__(w_estimator)

    def estimate(
        self,
        y_true: np.array,
        y_pred: np.array,
        sigma_pred: np.array,
        X: np.array = None,
        *args,
        **kwargs
    ) -> None:
        """Compute residuals on calibration set.
        Args:
            y_true: true values
            y_pred: predicted values
            sigma_pred: predicted absolute deviations
        """
        self._residuals = np.abs(y_true - y_pred)
        # Epsilon addition improves numerical stability
        self._residuals = self._residuals / (sigma_pred + EPSILON)

    def calibrate(
        self,
        y_pred: np.array,
        sigma_pred: np.array,
        alpha: float,
        X: np.array = None,
        *args,
        **kwargs
    ):
        """Compute calibrated prediction intervals for new examples.
        Args:
            y_pred: predicted values
            sigma_pred: predicted absolute deviations
            alpha: maximum miscoverage target
        Returns:
            y_lower, y_upper
        """
        if self._w_estimator is None:
            residuals_Q = np.quantile(
                self._residuals, (1 - alpha) * (1 + 1 / len(self._residuals))
            )
        else:
            self.weights = self.compute_weights(X)
            residuals_Q = w_quantile(
                np.concatenate((self._residuals, [np.inf])),
                1 - alpha,
                w=self.weights,
            )
        y_pred_upper = y_pred + sigma_pred * residuals_Q
        y_pred_lower = y_pred - sigma_pred * residuals_Q
        return y_pred_lower, y_pred_upper


class QuantileCalibrator(Calibrator):
    def __init__(self, w_estimator=None):
        super().__init__(w_estimator)

    def estimate(
        self,
        y_true: np.array,
        y_pred_lower: np.array,
        y_pred_upper: np.array,
        X: np.array = None,
        *args,
        **kwargs
    ) -> None:
        """Compute residuals on calibration set.
        Args:
            y_pred: predicted values
            y_pred_lower: lower bound of the prediction interval
            y_pred_upper: upper bound of the prediction interval
        """
        self._residuals = np.maximum(
            y_pred_lower - y_true,
            y_true - y_pred_upper,
        )

    def calibrate(
        self,
        y_pred_lower: np.array,
        y_pred_upper: np.array,
        alpha: float,
        X: np.array = None,
        *args,
        **kwargs
    ):
        """Compute calibrated prediction intervals for new examples.
        Args:
            y_pred_lower: lower bound of the prediction interval
            y_pred_upper: upper bound of the prediction interval
            alpha: maximum miscoverage target
        Returns:
            y_lower, y_upper
        """
        if self._w_estimator is None:
            residuals_Q = np.quantile(
                self._residuals, (1 - alpha) * (1 + 1 / len(self._residuals))
            )
        else:
            self.weights = self.compute_weights(X)
            residuals_Q = w_quantile(
                np.concatenate((self._residuals, [np.inf])),
                1 - alpha,
                w=self.weights,
            )
        y_pred_upper = y_pred_upper + residuals_Q
        y_pred_lower = y_pred_lower - residuals_Q
        return y_pred_lower, y_pred_upper

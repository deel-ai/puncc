"""
This module implements different calibration methods.
"""
from abc import ABC, abstractmethod
from puncc.utils import EPSILON, quantile, check_alpha_calib
import numpy as np


class Calibrator(ABC):
    """Abstract structure of a Calibrator class."""

    def __init__(self, w_estimator=None):
        self._residuals = None
        self._w_estimator = w_estimator
        self.calib_size = None
        self.weights = None

    def compute_weights(self, X, calib_size):
        """Compute and normalizes weight of the nonconformity distribution
        based on the provided w estimator.
        Args:
            X: features array
            w_estimator: weight function. By default, equal weights are
                         associated with samples mass density.
        """
        if self._w_estimator is None:  # equal weights
            return np.ones((len(X), calib_size + 1)) / (calib_size + 1)
        # Computation of normalized weights
        w = self._w_estimator(X)
        sum_w_calib = np.sum(self._w_calib)
        w_norm = np.zeros((len(X), calib_size + 1))
        for i in range(len(X)):
            w_norm[i, :calib_size] = self._w_calib / (sum_w_calib + w[i])
            w_norm[i, calib_size] = w[i] / (sum_w_calib + w[i])
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
            X: features array
        """
        self._residuals = np.abs(y_true - y_pred)
        if self._w_estimator is not None:
            self._w_calib = self._w_estimator(X)
        self.calib_size = len(X)

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
        # Check consistency of alpha w.r.t to the size of calibration data
        check_alpha_calib(alpha=alpha, n=self.calib_size)

        residuals_Qs = list()
        self.weights = self.compute_weights(X, self.calib_size)
        for w in self.weights:
            residuals_Q = quantile(
                np.concatenate((self._residuals, [np.inf])),
                1 - alpha,
                w=w,
            )
            residuals_Qs.append(residuals_Q)
        residuals_Qs = np.array(residuals_Qs)
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
            X: features array
        """
        self._residuals = np.abs(y_true - y_pred)
        # Epsilon addition improves numerical stability
        self._residuals = self._residuals / (sigma_pred + EPSILON)
        self.calib_size = len(X)

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
            X: features array
        Returns:
            y_lower, y_upper
        """
        # Check consistency of alpha w.r.t to the size of calibration data
        check_alpha_calib(alpha=alpha, n=self.calib_size)

        if self._w_estimator is not None:
            self.weights = self.compute_weights(X, self.calib_size)
        residuals_Q = quantile(
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
            X: features array
        """
        self._residuals = np.maximum(
            y_pred_lower - y_true,
            y_true - y_pred_upper,
        )
        self.calib_size = len(X)

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
        # Check consistency of alpha w.r.t to the size of calibration data
        check_alpha_calib(alpha=alpha, n=self.calib_size)

        if self._w_estimator is not None:
            self.weights = self.compute_weights(X, self.calib_size)
        residuals_Q = quantile(
            np.concatenate((self._residuals, [np.inf])),
            1 - alpha,
            w=self.weights,
        )
        y_pred_upper = y_pred_upper + residuals_Q
        y_pred_lower = y_pred_lower - residuals_Q
        return y_pred_lower, y_pred_upper

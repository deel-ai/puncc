"""
This module implements different calibration methods.
"""
from abc import ABC, abstractmethod
from puncc.utils import EPSILON
import numpy as np


class Calibrator(ABC):
    """Abstract structure of a Calibrator class."""

    def __init__(self):
        pass

    @abstractmethod
    def estimate(
        self,
        y_true: np.array,
        y_pred: np.array,
        y_pred_lower: np.array,
        y_pred_upper: np.array,
    ) -> None:
        """Compute residuals on calibration set.
        Args:
            y_true: true values
            y_pred: predicted values
            y_pred_lower: lower bound of the prediction interval
            y_pred_upper: upper bound of the prediction interval
        """
        pass

    @abstractmethod
    def calibrate(
        self,
        y_pred: np.array,
        y_pred_lower: np.array,
        y_pred_upper: np.array,
        alpha: float,
    ):
        """Compute calibrated prediction intervals for new examples.
        Args:
            y_pred: predicted values
            y_pred_lower: lower bound of the prediction interval
            y_pred_upper: upper bound of the prediction interval
            alpha: maximum miscoverage target
        Returns:
            y_lower, y_upper
        """
        pass


class MeanCalibrator(Calibrator):
    def __init__(self):
        super().__init__()
        self.name = "MeanCalibrator"
        self._residuals = None

    def estimate(self, y_true: np.array, y_pred: np.array, *args, **kwargs) -> None:
        """Compute residuals on calibration set.
        Args:
            y_true: true values
            y_pred: predicted values
        """
        self._residuals = np.abs(y_true - y_pred)

    def calibrate(self, y_pred: np.array, alpha: float, *args, **kwargs):
        """Compute calibrated prediction intervals for new examples.
        Args:
            y_pred: predicted values
            alpha: maximum miscoverage target
        Returns:
            y_lower, y_upper
        """
        residuals_Q = np.quantile(
            self._residuals, (1 - alpha) * (1 + 1 / len(self._residuals))
        )
        y_pred_lower = y_pred - residuals_Q
        y_pred_upper = y_pred + residuals_Q
        return y_pred_lower, y_pred_upper


class MeanVarCalibrator(Calibrator):
    def __init__(self):
        super().__init__()
        self.name = "MeanVarCalibrator"
        self._residuals = None

    def estimate(
        self, y_true: np.array, y_pred: np.array, sigma_pred: np.array, *args, **kwargs
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
        self, y_pred: np.array, sigma_pred: np.array, alpha: float, *args, **kwargs
    ):
        """Compute calibrated prediction intervals for new examples.
        Args:
            y_pred: predicted values
            sigma_pred: predicted absolute deviations
            alpha: maximum miscoverage target
        Returns:
            y_lower, y_upper
        """
        residuals_Q = np.quantile(
            self._residuals, (1 - alpha) * (1 + 1 / len(self._residuals))
        )
        y_pred_upper = y_pred + sigma_pred * residuals_Q
        y_pred_lower = y_pred - sigma_pred * residuals_Q
        return y_pred_lower, y_pred_upper


class QuantileCalibrator(Calibrator):
    def __init__(self):
        super().__init__()
        self._residuals = None
        self.name = "QuantileCalibrator"

    def estimate(
        self,
        y_true: np.array,
        y_pred_lower: np.array,
        y_pred_upper: np.array,
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
        self.residuals_Q = np.quantile(
            self._residuals, (1 - alpha) * (1 + 1 / len(self._residuals))
        )
        y_pred_upper = y_pred_upper + self.residuals_Q
        y_pred_lower = y_pred_lower - self.residuals_Q
        return y_pred_lower, y_pred_upper

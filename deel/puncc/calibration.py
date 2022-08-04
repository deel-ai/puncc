"""
This module implements the core Calibrator interface and different 
calibration methods.
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional
from deel.puncc.utils import EPSILON, quantile, check_alpha_calib
import numpy as np


def tibshirani_weights(*, X, weight_func, weights_calib):
    """Compute and normalize inference weights of the nonconformity distribution
    based on Tibshirani el al. http://arxiv.org/abs/1904.06019
    Args:
        X: features array
        weight_func: weight function. By default, equal weights are
            associated with samples mass density.
        weights_calib: weights assigned to the calibration samples
    Returns:
        normalized weights
    """
    calib_size = len(weights_calib)
    if weight_func is None:  # equal weights
        return np.ones((len(X), calib_size + 1)) / (calib_size + 1)
    # Computation of normalized weights
    w = weight_func(X)
    sum_w_calib = np.sum(weights_calib)

    w_norm = np.zeros((len(X), calib_size + 1))
    for i in range(len(X)):
        w_norm[i, :calib_size] = weights_calib / (sum_w_calib + w[i])
        w_norm[i, calib_size] = w[i] / (sum_w_calib + w[i])
    return w_norm


def barber_weights(*, X, weight_func, weights_calib):
    """Compute and normalize inference weights of the nonconformity distribution
    based on Barber et al. https://arxiv.org/abs/2202.13415
    Args:
        X: features array
        weight_func: weight function. By default, equal weights are
            associated with samples mass density
        weights_calib: weights assigned to the calibration samples
    Returns:
        normalized weights
    """
    calib_size = len(weights_calib)
    # Computation of normalized weights
    sum_w_calib = np.sum(weights_calib)

    w_norm = np.zeros((len(X), calib_size + 1))
    for i in range(len(X)):
        w_norm[i, :calib_size] = weights_calib / (sum_w_calib + 1)
        w_norm[i, calib_size] = 1 / (sum_w_calib + 1)
    return w_norm


class BaseCalibrator(ABC):
    """Abstract structure of a Calibrator class."""

    def __init__(self, weight_func=None, weight_method: str = "equi"):
        self._residuals = None
        self._len_calib = 0
        self._weights_calib = None
        self._weights_inference = None
        self.weight_func = weight_func
        self.weight_method = weight_method
        if weight_method == "equi":
            self._weight_norm = barber_weights
        elif weight_method == "barber":
            self._weight_norm = barber_weights
        elif weight_method == "tibshirani":
            self._weight_norm = tibshirani_weights
        else:
            error_msg = (
                f"{weight_method} is not implemented. Choose 'barber' or 'tibshirani'."
            )
            raise NotImplemented(error_msg)

    @abstractmethod
    def _nconf_score(
        self,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        var_pred: Optional[np.ndarray] = None,
        y_pred_lo: Optional[np.ndarray] = None,
        y_pred_hi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _calibrate(
        self,
        *,
        nconf_quantiles: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        var_pred: Optional[np.ndarray] = None,
        y_pred_lo: Optional[np.ndarray] = None,
        y_pred_hi: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def get_weights(self):
        return self._weights_inference

    def fit(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None,
        var_pred: Optional[np.ndarray] = None,
        y_pred_lo: Optional[np.ndarray] = None,
        y_pred_hi: Optional[np.ndarray] = None,
    ) -> None:
        """Compute nonconformity scores on the calibration set.
        Args:
            y_true: true values
            y_pred: predicted values
            X: features array
            var_pred: variability predictions
            y_pred_lo: lower bound of the prediction interval
            y_pred_hi: upper bound of the prediction interval
        """
        self._residuals = self._nconf_score(
            y_true=y_true,
            y_pred=y_pred,
            var_pred=var_pred,
            y_pred_lo=y_pred_lo,
            y_pred_hi=y_pred_hi,
        )
        self._len_calib = len(self._residuals)
        if self.weight_method is "equi":  # chosen method is equiprobable weights
            self._weights_calib = np.ones((self._len_calib))
        elif self.weight_func is not None:
            if X is None:
                raise RuntimeError(
                    f"'X' should be provided when weight function is not default"
                )
            else:
                self._weights_calib = self.weight_func(X)

    def calibrate(
        self,
        *,
        alpha: float,
        y_pred: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        var_pred: Optional[np.ndarray] = None,
        y_pred_lo: Optional[np.ndarray] = None,
        y_pred_hi: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute calibrated prediction intervals for new examples.
        Args:
            alpha: significance level
            y_pred: predicted values
            X: features array
            var_pred: variability predictions
            y_pred_lo: lower bound of the prediction interval
            y_pred_hi: upper bound of the prediction interval
            alpha: maximum miscoverage target
        Returns:
            y_lower, y_upper
        """
        if self._residuals is None:
            raise RuntimeError("Run 'fit' method before calling 'calibrate'.")

        # Check consistency of alpha w.r.t the size of calibration data
        check_alpha_calib(alpha=alpha, n=self._len_calib)

        residuals_Qs = list()

        self._weights_inference = self._weight_norm(
            X=X, weight_func=self.weight_func, weights_calib=self._weights_calib
        )
        for w in self._weights_inference:
            residuals_Q = quantile(
                np.concatenate((self._residuals, np.array([np.inf]))),
                1 - alpha,
                w=w,
            )
            residuals_Qs.append(residuals_Q)
        residuals_Qs = np.array(residuals_Qs)

        y_lo, y_hi = self._calibrate(
            nconf_quantiles=residuals_Qs,
            y_pred=y_pred,
            var_pred=var_pred,
            y_pred_lo=y_pred_lo,
            y_pred_hi=y_pred_hi,
        )

        return y_lo, y_hi


class MeanCalibrator(BaseCalibrator):
    def __init__(self, weight_func=None, weight_method: str = "equi"):
        super().__init__(weight_func, weight_method)

    def _nconf_score(
        self,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        var_pred: Optional[np.ndarray] = None,
        y_pred_lo: Optional[np.ndarray] = None,
        y_pred_hi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if y_pred is None:
            raise RuntimeError(
                "Cannot compute nonconformity scores because 'y_pred' is 'None'"
            )
        else:
            return np.abs(y_true - y_pred)  # type: ignore

    def _calibrate(
        self,
        *,
        nconf_quantiles: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        var_pred: Optional[np.ndarray] = None,
        y_pred_lo: Optional[np.ndarray] = None,
        y_pred_hi: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if y_pred is None:
            raise RuntimeError("Cannot calibrate PIs because 'y_pred' is 'None'")
        else:
            y_lo = y_pred - nconf_quantiles  # type: ignore
            y_hi = y_pred + nconf_quantiles
            return y_lo, y_hi


class MeanVarCalibrator(BaseCalibrator):
    def __init__(self, weight_func=None, weight_method: str = "equi"):
        super().__init__(weight_func, weight_method)

    def _nconf_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        var_pred: Optional[np.ndarray] = None,
        y_pred_lo: Optional[np.ndarray] = None,
        y_pred_hi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        nconf_scores = np.abs(y_true - y_pred)  # type: ignore
        if var_pred is None:
            raise RuntimeError(
                "Cannot compute nonconformity scores because 'var_pred' is 'None'"
            )
        else:
            # Epsilon addition improves numerical stability
            nconf_scores /= var_pred + EPSILON
            return nconf_scores

    def _calibrate(
        self,
        *,
        nconf_quantiles: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        var_pred: Optional[np.ndarray] = None,
        y_pred_lo: Optional[np.ndarray] = None,
        y_pred_hi: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if y_pred is None:
            raise RuntimeError("Cannot calibrate PIs because 'y_pred' is 'None'")
        elif var_pred is None:
            raise RuntimeError("Cannot calibrate PIs because 'var_pred' is 'None'")
        else:
            y_lo = y_pred - var_pred * nconf_quantiles  # type: ignore
            y_hi = y_pred + var_pred * nconf_quantiles
            return y_lo, y_hi


class QuantileCalibrator(BaseCalibrator):
    def __init__(self, weight_func=None, weight_method: str = "equi"):
        super().__init__(weight_func, weight_method)

    def _nconf_score(
        self,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        var_pred: Optional[np.ndarray] = None,
        y_pred_lo: Optional[np.ndarray] = None,
        y_pred_hi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if y_pred_lo is None:
            raise RuntimeError(
                "Cannot compute nonconformity scores because 'y_pred_lo' is 'None'"
            )
        elif y_pred_hi is None:
            raise RuntimeError(
                "Cannot compute nonconformity scores because 'y_pred_hi' is 'None'"
            )
        else:
            residuals = np.maximum(
                y_pred_lo - y_true,  # type: ignore
                y_true - y_pred_hi,  # type: ignore
            )
            return residuals

    def _calibrate(
        self,
        *,
        nconf_quantiles: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        var_pred: Optional[np.ndarray] = None,
        y_pred_lo: Optional[np.ndarray] = None,
        y_pred_hi: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if y_pred_lo is None:
            raise RuntimeError("Cannot calibrate PIs because 'y_pred_lo' is 'None'")
        elif y_pred_hi is None:
            raise RuntimeError("Cannot calibrate PIs because 'y_pred_hi' is 'None'")
        else:
            y_lo = y_pred_lo - nconf_quantiles  # type: ignore
            y_hi = y_pred_hi + nconf_quantiles
            return y_lo, y_hi


class MetaCalibrator(ABC):
    """Meta calibrator combines the estimations of nonconformity
    scores from each K-Fold calibrator and produces associated prediction
    intervals.

    Attributes:
        kfold_calibrators_dict: dictionarry of calibrators for each K-fold
                                (disjoint calibration subsets). Each calibrator
                                needs to priorly estimate the nonconformity
                                scores w.r.t the associated calibration fold.

    """

    def __init__(self, kfold_calibrators: dict):
        self.kfold_calibrators_dict = kfold_calibrators
        # Sanity checks:
        #   - The collection of calibrators is not None
        if kfold_calibrators is None:
            raise RuntimeError("Calibrators not defined.")
        #   - The calibrators in the collection are not None
        for k, calibrator in self.kfold_calibrators_dict.items():
            if calibrator is None:
                raise RuntimeError(f"Fold {k} calibrator is not defined.")

    def fit(self) -> None:
        """Check if all calibrators have already been fitted."""

        for k, calibrator in self.kfold_calibrators_dict.items():
            if calibrator._residuals is None:
                error_msg = (
                    f"Fold {k} calibrator should have priorly "
                    + "estimated its residuals."
                )
                raise RuntimeError(error_msg)

    @abstractmethod
    def calibrate(
        self,
        *,
        X: np.ndarray,
        kfold_predictors_dict: dict,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute calibrated prediction intervals for new examples X.
        Args:
            X: test features array
            kfold_predictors_dict: dictionnary of predictors trained for each fold on the fit subset.
            alpha: maximum miscoverage target
        Returns:
            Prediction interval's bounds: y_lo, y_hi
        """
        raise NotImplementedError()


class CvPlusCalibrator(MetaCalibrator):
    """Meta calibrator that combines the estimations nonconformity
    scores by each K-Fold calibrator and produces associated prediction
    intervals based on CV+ method.

    Attributes:
        kfold_calibrators_dict: dictionary of calibrators for each K-fold
                                (disjoint calibration subsets). Each calibrator
                                needs to priorly estimate the nonconformity
                                scores w.r.t the associated calibration fold.

    """

    def calibrate(
        self,
        *,
        X,
        kfold_predictors_dict: dict,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute calibrated prediction intervals for new examples.
        Args:
            X: test features array
            kfold_predictors_dict: dictionary of predictors trained for each K-fold
                fitting subset.
            alpha: maximum miscoverage target
        Returns:
            y_lo, y_hi
        """
        concat_residuals_lo = None
        concat_residuals_hi = None

        for k, predictor in kfold_predictors_dict.items():
            # Predictions
            y_pred, _, _, _ = predictor.predict(X)

            if y_pred is None:
                raise RuntimeError("No point predictor provided with cv+.")

            y_pred = np.reshape(y_pred, (len(y_pred), 1))

            # Residuals
            residuals = self.kfold_calibrators_dict[k]._residuals
            residuals = np.reshape(residuals, (1, len(residuals)))

            if concat_residuals_lo is None or concat_residuals_hi is None:
                concat_residuals_lo = y_pred - residuals  # type: ignore
                concat_residuals_hi = y_pred + residuals
            else:
                concat_residuals_lo = np.concatenate(
                    [concat_residuals_lo, y_pred - residuals], axis=1  # type: ignore
                )
                concat_residuals_hi = np.concatenate(
                    [concat_residuals_hi, y_pred + residuals], axis=1
                )

        # sanity check
        if concat_residuals_lo is None or concat_residuals_hi is None:
            raise RuntimeError("This should never happen.")
        else:
            y_lo = np.quantile(concat_residuals_lo, alpha, axis=1, method="lower")
            y_hi = np.quantile(concat_residuals_hi, 1 - alpha, axis=1, method="higher")
            return y_lo, y_hi


# class AggregatorCalibrator(MetaCalibrator):
#     """Meta calibrator that combines the estimations nonconformity
#     scores by each K-Fold calibrator and produces associated prediction
#     intervals. The point predictions on each fold are aggregated by calling agg_func.
#     The PI bounds are computed as the (1-alpha)-th quantiles of the k-fold PI bounds.

#     Attributes:
#         kfold_calibrators_dict: dictionary of calibrators for each K-fold
#                                 (disjoint calibration subsets). Each calibrator
#                                 needs to priorly estimate the nonconformity
#                                 scores w.r.t the associated calibration fold.

#         agg_func: function called to aggregate point predictions.
#     """

#     def __init__(self, kfold_calibrators: dict, agg_func):
#         super().__init__(kfold_calibrators=kfold_calibrators)
#         self.agg_func = agg_func

#     def calibrate(
#         self,
#         X: np.ndarray,
#         kfold_predictors_dict: dict,
#         alpha: float,
#     ) -> tuple[np.ndarray, np.ndarray]:
#         y_preds, y_pred_los, y_pred_his, var_preds = [], [], [], []

#         ## Recover K-fold estimator and predict response for the points X
#         for k, predictor in kfold_predictors_dict.items():
#             (
#                 y_pred,
#                 y_pred_lo,
#                 y_pred_hi,
#                 var_pred,
#             ) = predictor.predict(X)

#             y_preds.append(y_pred)
#             var_preds.append(var_pred)

#             # recover prefitted calibrators on each fold
#             calibrator = self.kfold_calibrators_dict[k]

#             # Compute/calibrate PI
#             (y_pred_lo, y_pred_hi) = calibrator.calibrate(
#                 y_pred=y_pred,
#                 alpha=alpha,
#                 y_pred_lo=y_pred_lo,
#                 y_pred_hi=y_pred_hi,
#                 var_pred=var_pred,
#                 X=X,
#             )

#             y_pred_los.append(y_pred_lo)
#             y_pred_his.append(y_pred_hi)

#         # Number of folds
#         K = len(kfold_predictors_dict.keys())

#         if K == 1:  # 1-Fold, no aggregation required
#             y_pred = y_preds[0]
#             y_pred_lo = y_pred_los[0]
#             y_pred_hi = y_pred_his[0]
#             var_pred = var_preds[0]
#         else:  # K-Fold, aggregate point predictions and PI bounds
#             y_pred = self.agg_func(y_preds)
#             y_pred_lo = np.quantile(y_pred_los, (1 - alpha) * (1 + 1 / K), axis=0)
#             y_pred_hi = np.quantile(y_pred_his, (1 - alpha) * (1 + 1 / K), axis=0)
#             var_pred = self.agg_func(var_preds)

#         return (y_pred, y_pred_lo, y_pred_hi, var_pred)

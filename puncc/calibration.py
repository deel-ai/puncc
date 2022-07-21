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
        # TODO: [Luca] why None? shouldn't X be mandatory?
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
        # TODO: [Luca] why None? shouldn't X be mandatory?
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
        # TODO: [Luca] why None? shouldn't X be mandatory?
        X: np.array = None,
        *args,
        **kwargs,
    ) -> None:
        """Compute residuals on calibration set.
        Args:
            y_true: true values
            y_pred: predicted values
            X: calibration features array
        """
        self._residuals = np.abs(y_true - y_pred)
        if self._w_estimator is not None:
            self._w_calib = self._w_estimator(X)
        self.calib_size = len(X)

    def calibrate(
        self,
        y_pred: np.array,
        alpha: float,
        # TODO: [Luca] why None? shouldn't X be mandatory?
        X: np.array = None,
        *args,
        **kwargs,
    ):
        """Compute calibrated prediction intervals for new examples.
        Args:
            y_pred: predicted values
            alpha: maximum miscoverage target
            X: test features array
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
        y_pred_lower = y_pred - residuals_Qs
        y_pred_upper = y_pred + residuals_Qs
        return y_pred_lower, y_pred_upper


class MeanVarCalibrator(Calibrator):
    def __init__(self, w_estimator=None):
        super().__init__(w_estimator)

    def estimate(
        self,
        y_true: np.array,
        y_pred: np.array,
        sigma_pred: np.array,
        # TODO: [Luca] why None? shouldn't X be mandatory?
        X: np.array = None,
        *args,
        **kwargs,
    ) -> None:
        """Compute residuals on calibration set.
        Args:
            y_true: true values
            y_pred: predicted values
            sigma_pred: predicted absolute deviations
            X: calibration features array
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
        # TODO: [Luca] why None? shouldn't X be mandatory?
        X: np.array = None,
        *args,
        **kwargs,
    ):
        """Compute calibrated prediction intervals for new examples.
        Args:
            y_pred: predicted values
            sigma_pred: predicted absolute deviations
            alpha: maximum miscoverage target
            X: test features array
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
        # TODO: [Luca] why None? shouldn't X be mandatory?
        X: np.array = None,
        *args,
        **kwargs,
    ) -> None:
        """Compute residuals on calibration set.
        Args:
            y_pred: predicted values
            y_pred_lower: lower bound of the prediction interval
            y_pred_upper: upper bound of the prediction interval
            X: calibration features array
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
        # TODO: [Luca] why None? shouldn't X be mandatory?
        X: np.array = None,
        *args,
        **kwargs,
    ):
        """Compute calibrated prediction intervals for new examples.
        Args:
            y_pred_lower: lower bound of the prediction interval
            y_pred_upper: upper bound of the prediction interval
            alpha: maximum miscoverage target
            X: test features array
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


class MetaCalibrator(Calibrator):
    """Meta calibrator that combines the estimations nonconformity
    scores by each K-Fold calibrator and produces associated prediction
    intervals.

    Attributes:
        kfold_calibrators_dict: dictionarry of calibrators for each K-fold
                                (disjoint calibration subsets). Each calibrator
                                needs to priorly estimate the nonconformity
                                scores w.r.t to the associated K-fold
                                calibration set.

    """

    def __init__(self, kfold_calibrators: dict):
        self.kfold_calibrators_dict = kfold_calibrators

    def estimate(self, *args, **kwargs):
        error_msg = (
            "Each KFold calibrator should have priorly " + "estimated the residuals."
        )
        if self.kfold_calibrators_dict is None:
            raise RuntimeError("Calibrators not defined.")
        for calibrator in self.kfold_calibrators_dict.values():
            if calibrator._residuals is None:
                raise RuntimeError(error_msg)

    @abstractmethod
    def calibrate(
        self,
        kfold_predictors: dict,
        alpha: float,
        X: np.array = None,
    ):
        """Compute calibrated prediction intervals for new examples.
        Args:
            kfold_predictors: dictionnary of predictors trained for each K-fold
                fitting subset.
            alpha: maximum miscoverage target
            X: test features array
        Returns:
            y_lower, y_upper
        """
        raise NotImplementedError


class CvPlusCalibrator(MetaCalibrator):
    """Meta calibrator that combines the estimations nonconformity
    scores by each K-Fold calibrator and produces associated prediction
    intervals.

    Attributes:
        kfold_calibrators_dict: dictionarry of calibrators for each K-fold
                                (disjoint calibration subsets). Each calibrator
                                needs to priorly estimate the nonconformity
                                scores w.r.t to the associated K-fold
                                calibration set.

    """

    def calibrate(
        self,
        kfold_predictors: dict,
        alpha: float,
        X: np.array = None,
    ):
        """Compute calibrated prediction intervals for new examples.
        Args:
            kfold_predictors: dictionnary of predictors trained for each K-fold
                fitting subset.
            alpha: maximum miscoverage target
            X: test features array
        Returns:
            y_lower, y_upper
        """
        concat_residuals_lo = None
        concat_residuals_hi = None

        for k, predictor in kfold_predictors.items():
            # Predictions
            y_pred, _, _, _ = predictor.predict(X)
            y_pred = np.reshape(y_pred, (len(y_pred), 1))

            # Residuals
            residuals = self.kfold_calibrators_dict[k]._residuals
            residuals = np.reshape(residuals, (1, len(residuals)))

            if concat_residuals_lo is None or concat_residuals_hi is None:
                concat_residuals_lo = y_pred - residuals
                concat_residuals_hi = y_pred + residuals
            else:
                concat_residuals_lo = np.concatenate(
                    [concat_residuals_lo, y_pred - residuals], axis=1
                )
                concat_residuals_hi = np.concatenate(
                    [concat_residuals_hi, y_pred + residuals], axis=1
                )

        y_pred_lower = np.quantile(
            concat_residuals_lo,
            alpha,
            axis=1,
            method="lower",
        )

        y_pred_upper = np.quantile(
            concat_residuals_hi,
            1 - alpha,
            axis=1,
            method="higher",
        )

        return y_pred_lower, y_pred_upper


class AggregationCalibrator(MetaCalibrator):
    """Meta calibrator that combines the estimations nonconformity
    scores by each K-Fold calibrator and produces associated prediction
    intervals.

    Attributes:
        kfold_calibrators_dict: dictionarry of calibrators for each K-fold
                                (disjoint calibration subsets). Each calibrator
                                needs to priorly estimate the nonconformity
                                scores w.r.t to the associated K-fold
                                calibration set.

    """

    def __init__(self, kfold_calibrators: dict, agg_func):
        super().__init__(kfold_calibrators=kfold_calibrators)
        self.agg_func = agg_func

    def calibrate(
        self,
        kfold_predictors: dict,
        alpha: float,
        # TODO: [Luca] why do we have a None here? without X, this should immediately fail, right?
        X: np.array = None,
    ):
        y_preds, y_pred_lowers, y_pred_uppers, sigma_preds = [], [], [], []

        ## Recover K-fold estimator and predict response for the points X
        for k, predictor in kfold_predictors.items():
            (
                y_pred,
                y_pred_lower,
                y_pred_upper,
                sigma_pred,
            ) = predictor.predict(X)

            y_preds.append(y_pred)
            sigma_preds.append(sigma_pred)

            # TODO: clarify with comments. What is calibrator here?
            # What do we expect?
            calibrator = self.kfold_calibrators_dict[k]

            # TODO: __else__ what? do we append None?
            # Should we check type of calibrator? and raise InputError because we
            # should have got a valid (what?) at instantiation?
            if calibrator is not None:
                calibrator = self.kfold_calibrators_dict[k]
                (y_pred_lower, y_pred_upper) = calibrator.calibrate(
                    y_pred=y_pred,
                    alpha=alpha,
                    y_pred_lower=y_pred_lower,
                    y_pred_upper=y_pred_upper,
                    sigma_pred=sigma_pred,
                    X=X,
                )

            y_pred_lowers.append(y_pred_lower)
            y_pred_uppers.append(y_pred_upper)

        # Number of splits
        K = len(kfold_predictors.keys())

        ## Aggregation of prediction
        if K == 1:  # Simple Split
            y_pred = y_preds[0]
            y_pred_lower = y_pred_lowers[0]
            y_pred_upper = y_pred_uppers[0]
            sigma_pred = sigma_preds[0]
        else:  # K-Fold Split
            y_pred = self.agg_func(y_preds)

            # TODO: fix quantile to use [method='higher']
            y_pred_lower = np.quantile(y_pred_lowers, (1 - alpha) * (1 + 1 / K), axis=0)

            # TODO:
            # WARNING: should this be [alpha], instead of [(1 - alpha)] ??
            y_pred_upper = np.quantile(y_pred_uppers, (1 - alpha) * (1 + 1 / K), axis=0)

            sigma_pred = self.agg_func(sigma_preds)

        # TODO: explicit Exception message
        # TODO: maybe better, raise ValueError/SomethingError
        # [Luca: why is this check even here for? I guess if quantile does not
        # return value, which is due to bad alpha (e.g. too small), which should be checked
        # somewhere else?]
        # [Luca: ok, this could also be to the possibility of having prediction==None above
        # We must handle that first.]
        if True in np.isnan(y_pred_lower):
            raise Exception

        return (y_pred, y_pred_lower, y_pred_upper, sigma_pred)

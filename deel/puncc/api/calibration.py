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
This module implements the core Calibrator interface and different
calibration methods.
"""
from abc import ABC
from abc import abstractmethod
from typing import Optional

import numpy as np

from deel.puncc.api.utils import check_alpha_calib
from deel.puncc.api.utils import EPSILON
from deel.puncc.api.utils import quantile


def tibshirani_weights(*, X, weight_func, weights_calib):
    """Compute and normalize inference weights of the nonconformity distribution
    based on `Tibshirani el al. <http://arxiv.org/abs/1904.06019>`_.

    :param ndarray X: features array
    :param callable weight_func: weight function. By default, equal weights are associated with samples mass density.
    :param ndarray weights_calib: weights assigned to the calibration samples.

    :returns: normalized weights.
    :rtype: ndarray
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
    based on `Barber et al. <https://arxiv.org/abs/2202.13415>`_.

    :param ndarray X: features array
    :param callable, optional weight_func: weight function. By default, equal weights are associated with samples mass density.
    :param ndarray weights_calib: weights assigned to the calibration samples.

    :returns: normalized weights.
    :rtype: ndarray
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
    """Abstract structure of a Calibrator class.


    :param callable weight_func: function that takes as argument an array of features X and returns associated "conformality" weights, defaults to None.
    :param str weight_method: weight normalization method ["equi", "barber", "tibshirani"]; Default value is "equi".

    :raises NotImplementedError: provided :data:`weight_method` is not suitable.
    """

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
            error_msg = f"{weight_method} is not implemented. Choose 'equi', 'barber' or 'tibshirani'."
            raise NotImplementedError(error_msg)

    @abstractmethod
    def _nconf_score(
        self,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        var_pred: Optional[np.ndarray] = None,
        y_pred_lo: Optional[np.ndarray] = None,
        y_pred_hi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Method that specifies and compute nonconformity scores.

        :param ndarray y_true: true values.
        :param ndarray, optional y_pred: predicted values, defaults to None
        :param ndarray, optional var_pred: variability predictions, defaults to None
        :param ndarray, optional y_pred_lo: lower bound of the prediction interval, defaults to None
        :param ndarray, optional y_pred_hi: upper bound of the prediction interval, defaults to None

        .. WARNING::
            All arguments except :data:`y_true` are optional depending of how nonconformity scores are specified.
            Some arguments may be required once the score is determined; in such case, an exception is raised if
            they are not provided.

        :returns: nonconformity scores.
        :rtype: ndarray

        :raises RuntimeError: error if some optional arguments are required.
        """
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
        """Ancillary method called by :meth:`calibrate` to compute prediction intervals w.r.t a nonconformity quantile.

        :param ndarray nconf_quantiles: quantile of nonconformity scores.
        :param ndarray, optional y_pred: predicted values, defaults to None.
        :param ndarray, optional var_pred: variability predictions, defaults to None.
        :param ndarray, optional y_pred_lo: lower bound of the prediction interval, defaults to None.
        :param ndarray, optional y_pred_hi: upper bound of the prediction interval, defaults to None.

        :returns: lower and upper bounds of the prediction intervals.
        :rtype: tuple[ndarray, ndarray]
        """
        raise NotImplementedError()

    def get_weights(self):
        """Getter for the weights associated to new examples

        :returns: weights on inference examples.
        :rtype: ndarray
        """
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
        """Compute and store nonconformity scores on the calibration set.

        :param ndarray y_true: true values.
        :param ndarray y_pred: predicted values.
        :param ndarray X: features array.
        :param ndarray var_pred: variability predictions.
        :param ndarray y_pred_lo: lower bound of the prediction interval.
        :param ndarray y_pred_hi: upper bound of the prediction interval.

        .. WARNING::
            Optional argument :data:`X` should be provided if the class attribute :data:`weight_func` is not default (None).

        :raises RuntimeError: Optional argument :data:`X` not provided but the class attribute :data:`weight_func` is not default (None).

        """
        self._residuals = self._nconf_score(
            y_true=y_true,
            y_pred=y_pred,
            var_pred=var_pred,
            y_pred_lo=y_pred_lo,
            y_pred_hi=y_pred_hi,
        )
        self._len_calib = len(self._residuals)
        if self.weight_method == "equi":  # chosen method is equiprobable weights
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

        :param float alpha: significance level (max miscoverage target).
        :param ndarray y_pred: predicted values.
        :param ndarray X: features array.
        :param ndarray var_pred: variability predictions.
        :param ndarray y_pred_lo: lower bound of the prediction interval.
        :param ndarray y_pred_hi: upper bound of the prediction interval.

        :returns: y_lower, y_upper.
        :rtype: tuple[ndarray, ndarray]

        :raises RuntimeError: :meth:`calibrate` called before :meth:`fit`.
        :raise ValueError: failed check on :data:`alpha` w.r.t size of the calibration set.

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

    :param dict kfold_calibrators_dict: dictionarry of calibrators for each K-fold (disjoint calibration subsets). Each calibrator needs to priorly estimate the nonconformity scores w.r.t the associated calibration fold.
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

        :param ndarray X: test features array.
        :param dict kfold_predictors_dict: dictionnary of predictors trained for each fold on the fit subset.
        :param float alpha: maximum miscoverage target.

        :returns: Prediction interval's bounds y_lo, y_hi.
        :rtype: tuple[ndarray, ndarray]
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
            # The following "upper" and "lower" quantiles follow the formulas
            # given in Section 1.2 of:
            # - Barber et al. 2019, "Predictive inference with the jackknife+"
            #
            # The authors define (with slightly misleading notation for "alpha"):
            #   q_hi(array, alpha) = ceiling( (1-alpha))-th smallest value of sorted(array)
            #   q_lo(array, alpha) = floor(     (alpha))-th smallest value of sorted(array)
            #
            # They point out how:
            #     q_lo(array, alpha) == (-1) * q_hi( (-1) * array, 1-alpha)
            #
            # Here we use the implementation in puncc.utils.quantile that returns
            # the empirical quantiles that yield the same result as in q_hi(.., ..)
            y_lo = (-1) * np.quantile(
                (-1) * concat_residuals_lo, 1 - alpha, axis=1, method="inverted_cdf"
            )
            y_hi = np.quantile(
                concat_residuals_hi, 1 - alpha, axis=1, method="inverted_cdf"
            )
            return y_lo, y_hi

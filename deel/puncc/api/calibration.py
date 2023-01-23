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
This module implements the core Calibrator and provides a collection of non-conformity scores and computations of the prediction sets.
"""
from typing import Iterable
from typing import Optional

import numpy as np

from deel.puncc.api import prediction_sets
from deel.puncc.api.utils import check_alpha_calib
from deel.puncc.api.utils import EPSILON
from deel.puncc.api.utils import quantile


class BaseCalibrator:
    """Calibrator class.

    :param callable nonconf_score_func: nonconformity score function
    :param callable pred_set_func:
    :param callable weight_func: function that takes as argument an array of
    features X and returns associated "conformality" weights, defaults to None.

    :raises NotImplementedError: provided :data:`weight_method` is not suitable.
    """

    @staticmethod
    def barber_weights(*, X, weights_calib):
        """Compute and normalize inference weights of the nonconformity distribution
        based on `Barber et al. <https://arxiv.org/abs/2202.13415>`.

        :param ndarray X: features array.
        :param ndarray weights_calib: weights assigned to the calibration samples.

        :returns: normalized weights.
        :rtype: ndarray
        """

        calib_size = len(weights_calib)
        # Computation of normalized weights
        sum_w_calib = np.sum(weights_calib)
        w_norm = np.zeros(calib_size + 1)

        w_norm[:calib_size] = weights_calib / (sum_w_calib + 1)
        w_norm[calib_size] = 1 / (sum_w_calib + 1)

        return w_norm

    def __init__(
        self,
        *,
        nonconf_score_func,
        pred_set_func,
        weight_func=None,
    ):
        self.nonconf_score_func = nonconf_score_func
        self.pred_set_func = pred_set_func
        self.weight_func = weight_func
        self._len_calib = 0
        self._residuals = None
        self._norm_weights = None

    def fit(
        self,
        *,
        y_true: Iterable,
        y_pred: Iterable,
    ) -> None:
        """Compute and store nonconformity scores on the calibration set.

        :param ndarray|DataFrame|Tensor y_true: true values.
        :param ndarray|DataFrame|Tensor y_pred: predicted values.
        :param Any kwargs: keyword arguments for the nonconformity score function.

        """
        # TODO check structure match in supported types
        self._residuals = self.nonconf_score_func(y_pred, y_true)
        self._len_calib = len(self._residuals)

    def calibrate(
        self,
        alpha: float,
        y_pred: Iterable,
        weights: Optional[Iterable] = None,
    ) -> tuple[Iterable, Optional[Iterable]]:
        """Compute calibrated prediction sets for new examples.

        :param float alpha: significance level (max miscoverage target).
        :param ndarray|DataFrame|Tensor y_pred: predicted values.
        :param Iterable weights: weights to be associated to the non-conformity
                                 scores. Defaults to None when all the scores
                                 are equiprobable.

        :returns: prediction set.
                  In case of regression, returns (y_lower, y_upper).
                  In case of classification, returns (classes,).
        :rtype: tuple[ndarray|DataFrame|Tensor, ndarray|DataFrame|Tensor]

        :raises RuntimeError: :meth:`calibrate` called before :meth:`fit`.
        :raise ValueError: failed check on :data:`alpha` w.r.t size of the calibration set.

        """

        if self._residuals is None:
            raise RuntimeError("Run `fit` method before calling `calibrate`.")

        # Check consistency of alpha w.r.t the size of calibration data
        check_alpha_calib(alpha=alpha, n=self._len_calib)

        # Compute weighted quantiles
        infty_array = np.array([np.inf])
        lemma_residuals = np.concatenate((self._residuals, infty_array))
        residuals_Q = quantile(
            lemma_residuals,
            1 - alpha,
            w=weights,
        )

        return self.pred_set_func(y_pred, scores_quantiles=residuals_Q)

    def set_norm_weights(self, norm_weights):
        """Setter of normalized weights associated to the nonconformity
        scores on the calibration set.

        :returns: normalized weights array.
        :rtype: ndarray
        """
        self._norm_weights = norm_weights

    def get_norm_weights(self):
        """Getter of normalized weights associated to the nonconformity
        scores on the calibration set.

        :param ndarray norm_weights: normalized weights array
        """
        return self._norm_weights


class CvPlusCalibrator:
    """Meta calibrator that combines the estimations nonconformity
    scores by each K-Fold calibrator and produces associated prediction
    intervals based on CV+ method.

    Attributes:
        kfold_calibrators_dict: dictionary of calibrators for each K-fold
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

    def calibrate(
        self,
        *,
        X: Iterable,
        kfold_predictors_dict: dict,
        alpha: float,
    ) -> tuple[Iterable, Iterable]:
        concat_residuals_lo = None
        concat_residuals_hi = None
        """Compute calibrated prediction intervals for new examples X.

        :param ndarray|DataFrame|Tensor X: test features.
        :param dict kfold_predictors_dict: dictionnary of predictors trained for each fold on the fit subset.
        :param float alpha: maximum miscoverage target.

        :returns: y_lower, y_upper.
        :rtype: tuple[ndarray|DataFrame|Tensor, ndarray|DataFrame|Tensor]
        """

        # Init the collection of upper and lower bounds of the K-fold's PIs
        concat_y_lo = None
        concat_y_hi = None

        for k, predictor in kfold_predictors_dict.items():
            # Predictions
            y_pred = predictor.predict(X)

            if y_pred is None:
                raise RuntimeError("No point predictor provided with cv+.")

            y_pred = np.reshape(y_pred, (len(y_pred), 1))

            # Residuals
            residuals = self.kfold_calibrators_dict[k]._residuals
            residuals = np.reshape(residuals, (1, len(residuals)))

            if concat_y_lo is None or concat_y_hi is None:
                concat_y_lo, concat_y_hi = prediction_sets.constant_interval(y_pred, residuals)  # type: ignore
            else:
                y_lo, y_hi = prediction_sets.constant_interval(y_pred, residuals)
                concat_y_lo = np.concatenate(
                    [concat_y_lo, y_lo], axis=1  # type: ignore
                )
                concat_y_hi = np.concatenate([concat_y_hi, y_hi], axis=1)

        # sanity check
        if concat_y_lo is None or concat_y_hi is None:
            raise RuntimeError("This should never happen.")
        else:
            y_lo = (-1) * np.quantile(
                (-1) * concat_y_lo, 1 - alpha, axis=1, method="inverted_cdf"
            )
            y_hi = np.quantile(concat_y_hi, 1 - alpha, axis=1, method="inverted_cdf")
            return y_lo, y_hi

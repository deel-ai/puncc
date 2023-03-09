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
This module implements the core calibrator, providing a structure to estimate the nonconformity scores
on the calibration set and to compute the prediction sets.
"""
import logging
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Tuple

import numpy as np

from deel.puncc.api import prediction_sets
from deel.puncc.api.utils import check_alpha_calib
from deel.puncc.api.utils import EPSILON
from deel.puncc.api.utils import quantile

logger = logging.getLogger(__name__)


class BaseCalibrator:
    """:class:`BaseCalibrator` offers a framework to compute user-defined
    nonconformity scores on calibration dataset(s) (:func:`fit`) and to use for
    constructing and/or calibrating prediction sets (:func:`calibrate`).

    :param callable nonconf_score_func: nonconformity score function.
    :param callable pred_set_func: prediction set construction function.
    :param callable weight_func: function that takes as argument an array of
        features X and returns associated "conformality" weights,
        defaults to None.

    :raises NotImplementedError: provided :data:`weight_method` is not suitable.

    .. _example basecalibrator:

    **Regression calibrator example:**


    Consider a pretrained model :math:`\hat{f}`, a calibration dataset
    :math:`(X_{calib}, y_{calib})` and a test dataset :math:`(X_{test}, y_{test})`.
    The model :math:`\hat{f}` generates predictions on the calibration and test sets:

    .. math::
        y_{pred, calib}=\hat{f}(X_{calib})

    .. math::
        y_{pred, test}=\hat{f}(X_{test})

    Two function need to be defined before instantiating the
    :class:`BaseCalibrator`: a nonconformity score function and a definition of
    how the prediction sets are computed. In the example below, these are
    implemented from scratch but a collection of ready-to-use nonconformity
    scores and prediction sets are provided in the modules :ref:`nonconformity_scores <nonconformity_scores>`
    and :ref:`prediction_sets <prediction_sets>`, respectively.


    .. code-block:: python

        from deel.puncc.api.calibration import BaseCalibrator
        import numpy as np

        # First, we define a nonconformity score function that takes as argument
        # the predicted values y_pred = model(X) and the true labels y_true. In
        # this example, we reimplement the mean absolute deviation that is
        # already defined in `deel.puncc.api.nonconformity_scores.mad`
        def nonconformity_function(y_pred, y_true):
            return np.abs(y_pred - y_true)

        # Prediction sets are computed based on points predictions and
        # the quantiles of nonconformity scores. The function below returns a
        # fixed size interval around the point predictions.
        def prediction_set_function(y_pred, scores_quantile):
            y_lo = y_pred - scores_quantile
            y_hi = y_pred + scores_quantile
            return y_lo, y_hi

        # The calibrator is instantiated by passing the two functions defined
        # above to the constructor.
        calibrator = BaseCalibrator(
            nonconf_score_func=nonconformity_function,
            pred_set_func=prediction_set_function
        )

        # The nonconformity scores are computed by calling the `fit` method
        # on the calibration dataset.
        calibrator.fit(y_pred=y_pred_calib, y_true=y_true_calib)

        # The lower and upper bounds of the prediction interval are then returned
        # by the call to calibrate on the new data w.r.t a risk level of 10%.
        y_pred_low, y_pred_high = calibrator.calibrate(y_pred=y_pred_test, alpha=.1)


    """

    def __init__(
        self,
        *,
        nonconf_score_func: Callable,
        pred_set_func: Callable,
        weight_func: Callable = None,
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

        :param Iterable y_true: true labels.
        :param Iterable y_pred: predicted values.

        """
        # TODO check structure match in supported types
        logger.info(f"Computing nonconformity scores ...")
        logger.debug(f"Shape of y_pred: {y_pred.shape}")
        logger.debug(f"Shape of y_true: {y_true.shape}")
        self._residuals = self.nonconf_score_func(y_pred, y_true)
        self._len_calib = len(self._residuals)
        logger.debug(f"Nonconformity scores computed !")

    def calibrate(
        self,
        *,
        alpha: float,
        y_pred: Iterable,
        weights: Optional[Iterable] = None,
    ) -> Tuple[np.ndarray]:
        """Compute calibrated prediction sets for new examples w.r.t a
        significance level :math:`\\alpha`.

        :param float alpha: significance level (max miscoverage target).
        :param Iterable y_pred: predicted values.
        :param Iterable weights: weights to be associated to the nonconformity
                                 scores. Defaults to None when all the scores
                                 are equiprobable.

        :returns: prediction set.
                  In case of regression, returns (y_lower, y_upper).
                  In case of classification, returns (classes,).
        :rtype: Tuple[ndarray]

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

        return self.pred_set_func(y_pred, scores_quantile=residuals_Q)

    def set_norm_weights(self, norm_weights: np.ndarray) -> None:
        """Setter of normalized weights associated to the nonconformity
        scores on the calibration set.

        :param ndarray norm_weights: normalized weights array

        """
        self._norm_weights = norm_weights

    def get_norm_weights(self) -> np.ndarray:
        """Getter of normalized weights associated to the nonconformity
        scores on the calibration set.

        :returns: normalized weights of nonconformity scores.
        :rtype: np.ndarray
        """
        return self._norm_weights

    @staticmethod
    def barber_weights(weights: np.ndarray) -> np.ndarray:
        """Compute and normalize inference weights of the nonconformity distribution
        based on `Barber et al. <https://arxiv.org/abs/2202.13415>`_.

        :param ndarray weights: weights assigned to the samples.

        :returns: normalized weights.
        :rtype: ndarray
        """

        weights_len = len(weights)

        # Computation of normalized weights
        sum_weights = np.sum(weights)
        w_norm = np.zeros(weights_len + 1)
        w_norm[:weights_len] = weights / (sum_weights + 1)
        w_norm[weights_len] = 1 / (sum_weights + 1)

        return w_norm


class CvPlusCalibrator:
    """Meta calibrator that combines the estimations of nonconformity
    scores by each K-Fold calibrator and produces associated prediction
    intervals based on `CV+ <https://arxiv.org/abs/1905.02928>`_.

    :param dict kfold_calibrators_dict: collection of calibrators for each
        K-fold (disjoint calibration subsets). Each calibrator needs to priorly
        estimate the nonconformity scores w.r.t the associated calibration fold.

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
        """Check if all calibrators have already been fitted.

        :raises RuntimeError: one or more of the calibrators did not estimate
            the nonconformity scores.

        """

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
    ) -> Tuple[np.ndarray]:
        """Compute calibrated prediction intervals for new examples X.

        :param Iterable X: test features.
        :param dict kfold_predictors_dict: dictionnary of predictors trained on each fold.
        :param float alpha: significance level (maximum miscoverage target).

        :returns: y_lower, y_upper.
        :rtype: Tuple[ndarray]
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

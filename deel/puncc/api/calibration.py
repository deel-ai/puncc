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
This module implements the core calibrator, providing a structure to estimate
the nonconformity scores on the calibration set and to compute the prediction
sets.
"""
import logging
import warnings
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Tuple

import numpy as np

from deel.puncc.api.corrections import bonferroni
from deel.puncc.api.utils import alpha_calib_check
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
    The model :math:`\hat{f}` generates predictions on the calibration and
    test sets:

    .. math::
        y_{pred, calib}=\hat{f}(X_{calib})

    .. math::
        y_{pred, test}=\hat{f}(X_{test})

    Two function need to be defined before instantiating the
    :class:`BaseCalibrator`: a nonconformity score function and a definition of
    how the prediction sets are computed. In the example below, these are
    implemented from scratch but a collection of ready-to-use nonconformity
    scores and prediction sets are provided in the modules
    :ref:`nonconformity_scores <nonconformity_scores>` and
    :ref:`prediction_sets <prediction_sets>`, respectively.


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

        # Generate dummy data and predictions
        y_pred_calib = np.random.rand(1000)
        y_true_calib = np.random.rand(1000)
        y_pred_test = np.random.rand(1000)

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
        self._feature_axis = None

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
        logger.info("Computing nonconformity scores ...")
        self._residuals = self.nonconf_score_func(y_pred, y_true)
        self._len_calib = len(self._residuals)
        if (
            y_pred.ndim > 1
        ):  # TODO make sure that ndim can be applied to all types
            self._feature_axis = -1
        logger.debug("Nonconformity scores computed !")

    def calibrate(
        self,
        *,
        alpha: float,
        y_pred: Iterable,
        weights: Optional[Iterable] = None,
        correction: Optional[Callable] = bonferroni,
    ) -> Tuple[np.ndarray]:
        """Compute calibrated prediction sets for new examples w.r.t a
        significance level :math:`\\alpha`.

        :param float alpha: significance level (max miscoverage target).
        :param Iterable y_pred: predicted values.
        :param Iterable weights: weights to be associated to the nonconformity
                                 scores. Defaults to None when all the scores
                                 are equiprobable.
        :param Callable correction: correction for multiple hypothesis testing
                                    in the case of multivariate regression.
                                    Defaults to Bonferroni correction.

        :returns: prediction set.
                  In case of regression, returns (y_lower, y_upper).
                  In case of classification, returns (classes,).
        :rtype: Tuple[ndarray]

        :raises RuntimeError: :meth:`calibrate` called before :meth:`fit`.
        :raise ValueError: failed check on :data:`alpha` w.r.t size of the
            calibration set.

        """
        residuals_Q = self.compute_quantile(
            alpha=alpha, weights=weights, correction=correction
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

    def get_nonconformity_scores(self) -> np.ndarray:
        """Getter of computed nonconformity scores on the calibration set.

        :returns: nonconformity scores.
        :rtype: np.ndarray
        """
        return self._residuals

    def compute_quantile(
        self,
        *,
        alpha: float,
        weights: Optional[Iterable] = None,
        correction: Optional[Callable] = bonferroni,
    ) -> np.ndarray:
        """Compute quantile of scores w.r.t a
        significance level :math:`\\alpha`.

        :param float alpha: significance level (max miscoverage target).
        :param Iterable weights: weights to be associated to the nonconformity
                                 scores. Defaults to None when all the scores
                                 are equiprobable.
        :param Callable correction: correction for multiple hypothesis testing
                                    in the case of multivariate regression.
                                    Defaults to Bonferroni correction.

        :returns: quantile
        :rtype: ndarray

        :raises RuntimeError: :meth:`compute_quantile` called before :meth:`fit`.
        :raise ValueError: failed check on :data:`alpha` w.r.t size of the
            calibration set.

        """

        if self._residuals is None:
            raise RuntimeError("Run `fit` method before calling `calibrate`.")

        alpha = correction(alpha)

        # Check consistency of alpha w.r.t the size of calibration data
        if weights is None:
            alpha_calib_check(alpha=alpha, n=self._len_calib)

        # Compute weighted quantiles
        ## Lemma 1 of Tibshirani's paper (https://arxiv.org/pdf/1904.06019.pdf)
        ## The coverage guarantee holds with 1) the inflated
        ## (1-\alpha)(1+1/n)-th quantile or 2) when adding an infinite term to
        ## the sequence and computing the $(1-\alpha)$-th empirical quantile.
        if self._residuals.ndim > 1:
            infty_array = np.full((1, self._residuals.shape[-1]), np.inf)
        else:
            infty_array = np.array([np.inf])
        lemma_residuals = np.concatenate((self._residuals, infty_array), axis=0)
        residuals_Q = quantile(
            lemma_residuals,
            1 - alpha,
            w=weights,
            feature_axis=self._feature_axis,
        )

        return residuals_Q

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


class ScoreCalibrator:
    """:class:`ScoreCalibrator` offers a framework to compute user-defined
    scores on a calibration dataset (:func:`fit`) and to test the conformity
    of new data points (:func:`is_conformal`) with respect to a significance
    (error) level :math:`\\alpha`. Such calibrator can be used for example
    to calibrate the decision threshold of anomaly detection scores.

    :param callable nonconf_score_func: nonconformity score function.
    :param callable weight_func: function that takes as argument an array of
        data points and returns associated "conformality" weights,
        defaults to None.

    .. _example scorecalibrator:

    **Anomaly detection example:**


    Consider the two moons dataset. We want to detect anomalous points in a new
    sample generated following a uniform distribution. The LOF algorithm is
    used to obtain anomaly scores; then a :class:`ScoreCalibrator` is
    instantiated to decide which scores are conformal (not anomalies) with
    respect to a significance level :math:`\\alpha`.

    .. code-block:: python

        import numpy as np
        from sklearn.datasets import make_moons
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import LocalOutlierFactor
        import matplotlib.pyplot as plt

        from deel.puncc.api.calibration import ScoreCalibrator

        # First, we generate the two moons dataset
        dataset = 4*make_moons(n_samples=1000, noise=0.05,
            random_state=0)[0] - np.array([0.5, 0.25])

        # Split data into proper fitting and calibration sets
        fit_set, calib_set = train_test_split(dataset, train_size=.8)

        # Generate new data points
        rng = np.random.RandomState(42)
        new_samples = rng.uniform(low=-6, high=8, size=(200, 2))

        # Instantiate the LOF anomaly detection algorithm
        algorithm = LocalOutlierFactor(n_neighbors=35, novelty=True)

        # Fit the LOF on the proper fitting dataset
        algorithm.fit(X=fit_set)

        # The nonconformity scores are defined as the LOF (anomaly) scores.
        # By default, score_samples return the opposite of LOF scores.
        ncf = lambda X: -algorithm.score_samples(X)

        # The ScoreCalibrator is instantiated by passing the LOF score function
        # to the constructor
        cad = ScoreCalibrator(nonconf_score_func=ncf)

        # The LOF scores are computed by calling the `fit` method
        # on the calibration dataset
        cad.fit(calib_set)

        # We set the target false detection rate to 1%
        alpha = .01

        # The method `is_conformal` is called on the new data points
        # to test which are conformal (not anomalous) and which are not
        results = cad.is_conformal(z=new_samples, alpha=alpha)
        not_anomalies = new_samples[results]
        anomalies = new_samples[np.invert(results)]

        # Plot the results
        plt.scatter(calib_set[:,0], calib_set[:,1],
                    s=10, label="Inliers")
        plt.scatter(not_anomalies[:, 0], not_anomalies[:, 1], s=40, marker="x",
                    color="blue", label="Normal")
        plt.scatter(anomalies[:, 0], anomalies[:, 1], s=40, marker="x",
                    color="red", label="Anomaly")
        plt.xticks(())
        plt.yticks(())
        plt.legend(loc="lower left")

    """

    def __init__(
        self, nonconf_score_func: Callable, weight_func: Callable = None
    ):
        self.nonconf_score_func = nonconf_score_func
        self._nonconf_scores = None
        self._calib_len = 0
        self.weight_func = weight_func

    def fit(self, z: Iterable):
        """Compute and store nonconformity scores on the calibration set.

        :param Iterable z: calibration dataset.
        """
        self._nonconf_scores = self.nonconf_score_func(z)
        self._calib_len = len(self._nonconf_scores)

    def set_nonconformity_scores(self, scores: np.array):
        """Setter of nonconformity scores. Can be used instead of calling
        :func:`fit` if the nonconformity scores are already computed.

        :param ndarray scores: nonconformity scores.
        """
        if self._nonconf_scores is not None:
            warnings.warn(
                "Warning........... You are overwriting previously computed or provided scores."
            )
        self._nonconf_scores = np.copy(scores)
        self._calib_len = len(self._nonconf_scores)

    def get_nonconformity_scores(self) -> np.ndarray:
        """Getter of computed nonconformity scores on the calibration set.

        :returns: nonconformity scores.
        :rtype: np.ndarray
        """
        return self._nonconf_scores

    def is_conformal(self, z: Iterable, alpha: float) -> np.ndarray:
        """Test if new data points `z` are conformal. The test result is True
        if the new sample is conformal w.r.t a significance level
        :math:`\\alpha` and False otherwise.

        :param Iterable z: new samples.
        :param float alpha: significance level.

        :returns: conformity test results.
        :rtype: np.ndarray[bool]
        """
        if self._nonconf_scores is None:
            raise RuntimeError(
                "Run `fit` or 'set_nonconformity_scores' methods before calling `is_conformal`."
            )

        n = self._calib_len
        weights = None

        if self.weight_func:
            weights = self.weight_func(z)

        q_hat = quantile(
            self._nonconf_scores, q=(1 - alpha) * (n + 1) / n, w=weights
        )

        test_nonconf_scores = self.nonconf_score_func(z)

        return test_nonconf_scores <= q_hat


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
        self._len_calib = None
        self._feature_axis = None

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
        len_calib = 0
        for k, calibrator in self.kfold_calibrators_dict.items():
            kth_nconf_scores = calibrator.get_nonconformity_scores()
            if kth_nconf_scores is None:
                error_msg = (
                    f"Fold {k} calibrator should have priorly "
                    + "been estimated its nonconformity scores."
                )
                raise RuntimeError(error_msg)
            len_calib += len(kth_nconf_scores)
        self._len_calib = len_calib

    def calibrate(
        self,
        *,
        X: Iterable,
        kfold_predictors_dict: dict,
        alpha: float,
    ) -> Tuple[np.ndarray]:
        """Compute calibrated prediction intervals for new examples X.

        :param Iterable X: test features.
        :param dict kfold_predictors_dict: dictionnary of predictors trained
            on each fold.
        :param float alpha: significance level (maximum miscoverage target).

        :returns: y_lower, y_upper.
        :rtype: Tuple[ndarray]
        """
        # Check if all calibrators have already been fitted
        self.fit()

        # Check consistency of alpha w.r.t the size of calibration data
        alpha_calib_check(alpha=alpha, n=self._len_calib)

        # Init the collection of upper and lower bounds of the K-fold's PIs
        concat_y_lo = None
        concat_y_hi = None
        concat_norm_weights = None

        for k, predictor in kfold_predictors_dict.items():
            # Predictions
            y_pred = predictor.predict(X)

            if y_pred is None:
                raise RuntimeError("No prediction obtained with cv+.")

            # Check for multivariate predictions
            if y_pred.ndim > 1:
                self._feature_axis = -1

            # nonconformity scores
            kth_calibrator = self.kfold_calibrators_dict[k]
            nconf_scores = kth_calibrator.get_nonconformity_scores()
            norm_weights = kth_calibrator.get_norm_weights()

            # Reshaping nonconformity scores to broadcast them
            # on y_pred samples when computing the prediction sets
            # Source: R. Barber Section 3 https://arxiv.org/pdf/1905.02928.pdf
            y_pred = y_pred[..., np.newaxis]
            if len(nconf_scores.shape) != 2:
                try:
                    nconf_scores = np.array(nconf_scores)
                    nconf_scores = nconf_scores[np.newaxis, ...]
                except Exception:
                    # @TODO extend the scope beyond castable to ndarrays
                    raise RuntimeError(
                        "Cannot cast nonconformity scores to numpy array."
                    )

            if concat_y_lo is None or concat_y_hi is None:
                (concat_y_lo, concat_y_hi) = kth_calibrator.pred_set_func(
                    y_pred, nconf_scores
                )
                if norm_weights is not None:
                    concat_norm_weights = norm_weights
            else:
                y_lo, y_hi = kth_calibrator.pred_set_func(y_pred, nconf_scores)
                concat_y_lo = np.concatenate([concat_y_lo, y_lo], axis=1)
                concat_y_hi = np.concatenate([concat_y_hi, y_hi], axis=1)
                if norm_weights is not None:
                    concat_norm_weights = np.concatenate(
                        [concat_norm_weights, norm_weights]
                    )

        # sanity check
        if concat_y_lo is None or concat_y_hi is None:
            raise RuntimeError("This should never happen.")

        if concat_norm_weights is None:
            weights = None
        else:
            weights = concat_norm_weights
            infty_array = np.array([np.inf])
            concat_y_lo = np.concatenate([concat_y_lo, infty_array])
            concat_y_hi = np.concatenate([concat_y_hi, infty_array])

        y_lo = -1 * quantile(
            -1 * concat_y_lo,
            (1 - alpha) * (1 + 1 / self._len_calib),
            w=weights,
            axis=1,
            feature_axis=self._feature_axis,
        )
        y_hi = quantile(
            concat_y_hi,
            (1 - alpha) * (1 + 1 / self._len_calib),
            w=weights,
            axis=1,
            feature_axis=self._feature_axis,
        )
        return y_lo, y_hi

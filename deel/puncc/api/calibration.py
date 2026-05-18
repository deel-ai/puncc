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
"""This module implements the core calibrator, providing a structure to estimate
the nonconformity scores on the calibration set and to compute the prediction
sets."""

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
from deel.puncc.api.backend import get_backend, shape2

logger = logging.getLogger(__name__)


class BaseCalibrator:
    """Compute nonconformity scores and construct calibrated prediction sets.

    `BaseCalibrator` offers a framework to compute user-defined nonconformity
    scores on calibration datasets with `fit` and to construct calibrated
    prediction sets with `calibrate`.

    Args:
        nonconf_score_func (callable): Nonconformity score function.
        pred_set_func (callable): Prediction set construction function.
        weight_func (callable): Function that takes an array of features `X`
            and returns associated conformality weights. Defaults to `None`.

    Raises:
        NotImplementedError: Provided `weight_method` is not suitable.

    Examples:
        Consider a pretrained model $\\hat{f}$, a calibration dataset
        $(X_{calib}, y_{calib})$ and a test dataset $(X_{test}, y_{test})$.
        The model $\\hat{f}$ generates predictions on the calibration and test
        sets:

        $$
        y_{pred, calib}=\\hat{f}(X_{calib})
        $$

        $$
        y_{pred, test}=\\hat{f}(X_{test})
        $$

        Two functions need to be defined before instantiating the
        `BaseCalibrator`: a nonconformity score function and a definition of
        how the prediction sets are computed. In the example below, these are
        implemented from scratch, but ready-to-use nonconformity scores and
        prediction sets are provided in the `nonconformity_scores` and
        `prediction_sets` modules.

        ```python
        from deel.puncc.api.calibration import BaseCalibrator
        import numpy as np

        def nonconformity_function(y_pred, y_true):
            return np.abs(y_pred - y_true)

        def prediction_set_function(y_pred, scores_quantile):
            y_lo = y_pred - scores_quantile
            y_hi = y_pred + scores_quantile
            return y_lo, y_hi

        calibrator = BaseCalibrator(
            nonconf_score_func=nonconformity_function,
            pred_set_func=prediction_set_function,
        )

        y_pred_calib = np.random.rand(1000)
        y_true_calib = np.random.rand(1000)
        y_pred_test = np.random.rand(1000)

        calibrator.fit(y_pred=y_pred_calib, y_true=y_true_calib)
        y_pred_low, y_pred_high = calibrator.calibrate(
            y_pred=y_pred_test,
            alpha=0.1,
        )
        ```
    """

    def __init__(
        self,
        *,
        nonconf_score_func: Callable,
        pred_set_func: Callable,
        weight_func: Callable = None,
        **kwargs,
    ):
        del kwargs
        self.nonconf_score_func = nonconf_score_func
        self.pred_set_func = pred_set_func
        self.weight_func = weight_func
        self._len_calib = 0
        self._residuals = None
        self._norm_weights = None
        self._feature_axis = None

    def fit(self, *, y_true: Iterable, y_pred: Iterable, **kwargs) -> None:
        """Compute and store nonconformity scores on the calibration set.

        Args:
            y_true (Iterable): true labels.
            y_pred (Iterable): predicted values."""
        del kwargs
        self._update_feature_axis(y_pred)
        self._residuals = self.nonconf_score_func(y_pred, y_true)
        self._len_calib = len(self._residuals)

    def calibrate(  # pylint: disable=unused-argument
        self,
        *,
        alpha: float,
        X: Optional[Iterable] = None,
        y_pred: Iterable,
        weights: Optional[Iterable] = None,
        correction: Optional[Callable] = bonferroni,
    ) -> Tuple[np.ndarray]:
        """Compute calibrated prediction sets for new examples w.r.t a
                significance level $\\alpha$.

        Args:
            alpha (float): significance level (max miscoverage target).
            X (Iterable): test features, used to compute the weights if a  weight_func is defined.
            y_pred (Iterable): predicted values.
            weights (Iterable): weights to be associated to the nonconformity scores. Defaults to None when all the scores are equiprobable.
            correction (Callable): correction for multiple hypothesis testing in the case of multivariate regression. Defaults to Bonferroni correction.

        Returns:
            Tuple[ndarray]: prediction set. In case of regression, returns (y_lower, y_upper). In case of classification, returns (classes,).

        Raises:
            RuntimeError: `calibrate` called before `fit`.
            ValueError: failed check on :data:`alpha` w.r.t size of the calibration set.
        """
        residuals_Q = self.compute_quantile(
            alpha=alpha, weights=weights, correction=correction
        )

        return self.pred_set_func(y_pred, scores_quantile=residuals_Q)

    def set_norm_weights(self, norm_weights: np.ndarray) -> None:
        """Setter of normalized weights associated to the nonconformity
                scores on the calibration set.

        Args:
            norm_weights (ndarray): normalized weights array"""
        self._norm_weights = norm_weights

    def get_norm_weights(self) -> np.ndarray:
        """Getter of normalized weights associated to the nonconformity
                scores on the calibration set.

        Returns:
            np.ndarray: normalized weights of nonconformity scores."""
        return self._norm_weights

    def get_nonconformity_scores(self) -> np.ndarray:
        """Getter of computed nonconformity scores on the calibration set.

        Returns:
            np.ndarray: nonconformity scores."""
        return self._residuals

    def compute_quantile(
        self,
        *,
        alpha: float,
        weights: Optional[Iterable] = None,
        correction: Optional[Callable] = bonferroni,
    ) -> np.ndarray:
        """Compute quantile of scores w.r.t a
                significance level $\\alpha$.

        Args:
            alpha (float): significance level (max miscoverage target).
            weights (Iterable): weights to be associated to the nonconformity scores. Defaults to None when all the scores are equiprobable.
            correction (Callable): correction for multiple hypothesis testing in the case of multivariate regression. Defaults to Bonferroni correction.

        Returns:
            ndarray: quantile

        Raises:
            RuntimeError: `compute_quantile` called before `fit`.
            ValueError: failed check on :data:`alpha` w.r.t size of the calibration set.
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

    def _update_feature_axis(self, y_pred):
        b = get_backend(y_pred)
        ndim = shape2(b.asarray(y_pred))[0]
        if ndim > 1:
            self._feature_axis = -1

    @staticmethod
    def barber_weights(weights: np.ndarray) -> np.ndarray:
        """Compute and normalize inference weights of the nonconformity distribution
                based on [Barber et al.](https://arxiv.org/abs/2202.13415).

        Args:
            weights (ndarray): weights assigned to the samples.

        Returns:
            ndarray: normalized weights."""

        weights_len = len(weights)

        # Computation of normalized weights
        sum_weights = np.sum(weights)
        w_norm = np.zeros(weights_len + 1)
        w_norm[:weights_len] = weights / (sum_weights + 1)
        w_norm[weights_len] = 1 / (sum_weights + 1)

        return w_norm


class ClasswiseCalibrator(BaseCalibrator):
    """Calibrator for classwise conformal prediction.

        This calibrator computes per-class quantiles of nonconformity scores,
        handling NaN values that indicate scores from other classes.

    Args:
        nonconf_score_func (callable): nonconformity score function.
        pred_set_func (callable): prediction set construction function.
        weight_func (callable): function that takes as argument an array of features X and returns associated "conformality" weights, defaults to None.
    """

    def compute_quantile(
        self,
        *,
        alpha: float,
        weights: Optional[Iterable] = None,
        correction: Optional[Callable] = bonferroni,
    ) -> np.ndarray:
        """Compute per-class quantiles of scores w.r.t a
                significance level $\\alpha$.

        Args:
            alpha (float): significance level (max miscoverage target).
            weights (Iterable): weights to be associated to the nonconformity scores. Defaults to None when all the scores are equiprobable.
            correction (Callable): correction for multiple hypothesis testing. Defaults to Bonferroni correction.

        Returns:
            ndarray: per-class quantiles, shape (n_classes,)

        Raises:
            RuntimeError: `compute_quantile` called before `fit`.
            ValueError: failed check on :data:`alpha` w.r.t size of the calibration set for a given class.
        """
        if self._residuals is None:
            raise RuntimeError("Run `fit` method before calling `calibrate`.")

        alpha = correction(alpha)

        n_classes = self._residuals.shape[1]
        quantiles = np.zeros(n_classes)

        for c in range(n_classes):
            # Get scores for class c (non-NaN values only)
            class_scores = self._residuals[:, c]
            class_scores = class_scores[~np.isnan(class_scores)]

            if len(class_scores) == 0:
                quantiles[c] = np.inf
                logger.warning(
                    f"No calibration samples for class {c}, setting quantile to inf."
                )
                continue

            # Check consistency of alpha w.r.t the size of calibration data for this class
            if weights is None:
                alpha_calib_check(alpha=alpha, n=len(class_scores))

            # Add infinity for coverage guarantee (Tibshirani's lemma)
            class_scores_with_inf = np.concatenate([class_scores, [np.inf]])

            quantiles[c] = np.quantile(
                class_scores_with_inf, 1 - alpha, method="inverted_cdf"
            )

        return quantiles


class LeveragedCalibrator(BaseCalibrator):
    """Calibrator for leverage-weighted conformal prediction.

        This calibrator computes quantiles of nonconformity scores with weights
        derived from the leverage function, as proposed in `Fadnavis
        <https://arxiv.org/abs/2602.12693>`_.

    Args:
        nonconf_score_func (callable): nonconformity score function.
        pred_set_func (callable): prediction set construction function.
        weight_func (callable): function that takes as argument an array of features X and returns associated weighted leverage-based "conformality" weights, defaults to identity function.
        leverage_func (Callable): function to compute covariates leverage score."""

    def __init__(
        self,
        *,
        nonconf_score_func: Callable,
        pred_set_func: Callable,
        weight_func: Callable = lambda x: x,
        leverage_func: Callable = lambda x: x,
    ):
        super().__init__(
            nonconf_score_func=nonconf_score_func,
            pred_set_func=pred_set_func,
        )
        self.lweight_func = weight_func
        # No other weighting than leverage-based weighting
        # is used in this calibrator
        self.weight_func = None
        self.leverage_func = leverage_func
        self.wlf = lambda x: x

    def fit(self, *, X: Iterable, y_true: Iterable, y_pred: Iterable) -> None:
        """Compute and store nonconformity scores on the calibration set.

        Args:
            X (Iterable): input features.
            y_true (Iterable): true labels.
            y_pred (Iterable): predicted values."""
        # Weigthed leverage function to compute the weights
        # for the nonconformity scores
        self._update_feature_axis(y_pred)
        self.wlf = lambda x: self.lweight_func(self.leverage_func(x))
        self._residuals = self.nonconf_score_func(
            X, y_pred, y_true, weight_func=self.wlf
        )
        self._len_calib = len(self._residuals)

    def calibrate(
        self,
        *,
        alpha: float,
        X: Iterable,
        y_pred: Iterable,
        weights: Optional[Iterable] = None,
        correction: Optional[Callable] = bonferroni,
    ) -> Tuple:
        residuals_Q = self.compute_quantile(
            alpha=alpha, weights=weights, correction=correction
        )
        return self.pred_set_func(
            y_pred, scores_quantile=residuals_Q, weights=1 / self.wlf(X)
        )


class ScoreCalibrator:
    """Compute and calibrate user-defined conformity scores.

    `ScoreCalibrator` computes user-defined scores on a calibration dataset
    with `fit` and tests the conformity of new data points with
    `is_conformal` at a significance level $\\alpha$. It can be used, for
    example, to calibrate the decision threshold of anomaly detection scores.

    Args:
        nonconf_score_func (callable): Nonconformity score function.
        weight_func (callable): Function that takes an array of data points
            and returns associated conformality weights. Defaults to `None`.

    Examples:
        Consider the two moons dataset. We want to detect anomalous points in a
        new sample generated from a uniform distribution. The LOF algorithm is
        used to obtain anomaly scores; then a `ScoreCalibrator` is instantiated
        to decide which scores are conformal with respect to a significance
        level $\\alpha$.

        ```python
        import numpy as np
        from sklearn.datasets import make_moons
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import LocalOutlierFactor
        import matplotlib.pyplot as plt

        from deel.puncc.api.calibration import ScoreCalibrator

        dataset = (
            4 * make_moons(n_samples=1000, noise=0.05, random_state=0)[0]
            - np.array([0.5, 0.25])
        )
        fit_set, calib_set = train_test_split(dataset, train_size=0.8)

        rng = np.random.RandomState(42)
        new_samples = rng.uniform(low=-6, high=8, size=(200, 2))

        algorithm = LocalOutlierFactor(n_neighbors=35, novelty=True)
        algorithm.fit(X=fit_set)

        ncf = lambda X: -algorithm.score_samples(X)
        cad = ScoreCalibrator(nonconf_score_func=ncf)
        cad.fit(calib_set)

        alpha = 0.01
        results = cad.is_conformal(z=new_samples, alpha=alpha)
        not_anomalies = new_samples[results]
        anomalies = new_samples[np.invert(results)]

        plt.scatter(calib_set[:, 0], calib_set[:, 1], s=10, label="Inliers")
        plt.scatter(
            not_anomalies[:, 0],
            not_anomalies[:, 1],
            s=40,
            marker="x",
            color="blue",
            label="Normal",
        )
        plt.scatter(
            anomalies[:, 0],
            anomalies[:, 1],
            s=40,
            marker="x",
            color="red",
            label="Anomaly",
        )
        plt.xticks(())
        plt.yticks(())
        plt.legend(loc="lower left")
        ```
    """

    def __init__(self, nonconf_score_func: Callable, weight_func: Callable = None):
        self.nonconf_score_func = nonconf_score_func
        self._nonconf_scores = None
        self._calib_len = 0
        self.weight_func = weight_func

    def fit(self, z: Iterable):
        """Compute and store nonconformity scores on the calibration set.

        Args:
            z (Iterable): calibration dataset."""
        self._nonconf_scores = self.nonconf_score_func(z)
        self._calib_len = len(self._nonconf_scores)

    def set_nonconformity_scores(self, scores: np.array):
        """Setter of nonconformity scores. Can be used instead of calling
                `fit` if the nonconformity scores are already computed.

        Args:
            scores (ndarray): nonconformity scores."""
        if self._nonconf_scores is not None:
            warnings.warn(
                "Warning........... You are overwriting previously computed or provided scores."
            )
        self._nonconf_scores = np.copy(scores)
        self._calib_len = len(self._nonconf_scores)

    def get_nonconformity_scores(self) -> np.ndarray:
        """Getter of computed nonconformity scores on the calibration set.

        Returns:
            np.ndarray: nonconformity scores."""
        return self._nonconf_scores

    def is_conformal(self, z: Iterable, alpha: float) -> np.ndarray:
        """Test if new data points `z` are conformal. The test result is True
                if the new sample is conformal w.r.t a significance level
                $\\alpha$ and False otherwise.

        Args:
            z (Iterable): new samples.
            alpha (float): significance level.

        Returns:
            np.ndarray[bool]: conformity test results."""
        if self._nonconf_scores is None:
            raise RuntimeError(
                "Run `fit` or 'set_nonconformity_scores' methods before calling `is_conformal`."
            )

        n = self._calib_len
        weights = None

        if self.weight_func:
            weights = self.weight_func(z)

        q_hat = quantile(self._nonconf_scores, q=(1 - alpha) * (n + 1) / n, w=weights)

        test_nonconf_scores = self.nonconf_score_func(z)

        return test_nonconf_scores <= q_hat


class CvPlusCalibrator:
    """Meta calibrator that combines the estimations of nonconformity
        scores by each K-Fold calibrator and produces associated prediction
        intervals based on [CV+](https://arxiv.org/abs/1905.02928).

    Args:
        kfold_calibrators_dict (dict): collection of calibrators for each K-fold (disjoint calibration subsets). Each calibrator needs to priorly estimate the nonconformity scores w.r.t the associated calibration fold.
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

        Raises:
            RuntimeError: one or more of the calibrators did not estimate the nonconformity scores.
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

        Args:
            X (Iterable): test features.
            kfold_predictors_dict (dict): dictionnary of predictors trained on each fold.
            alpha (float): significance level (maximum miscoverage target).

        Returns:
            Tuple[ndarray]: y_lower, y_upper."""
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
                concat_y_lo, concat_y_hi = kth_calibrator.pred_set_func(
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

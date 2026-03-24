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
This module implements usual conformal regression wrappers.
"""
from copy import deepcopy

import numpy as np
from sklearn.utils import resample

from deel.puncc.api.nonconformity_scores import absolute_difference, scaled_ad, cqr_score
from deel.puncc.api.prediction_sets import constant_interval, scaled_interval, cqr_interval
from deel.puncc.api.conformal_predictor import ConformalPredictor, StaticConformalPredictor


class SplitCP(StaticConformalPredictor):
    nc_score_function=absolute_difference()
    pred_set_function=constant_interval()

class CQR(StaticConformalPredictor):
    nc_score_function=cqr_score()
    pred_set_function=cqr_interval()

# TODO : put the eps somewhere else
class LocallyAdaptiveCP(ConformalPredictor):
    def __init__(self, model,
                 weight_function=None,
                 eps:float=1e-12):
        super().__init__(
            model=model,
            nc_score_function=scaled_ad(eps=eps),
            pred_set_function=scaled_interval(eps=eps),
            weight_function=weight_function,
        )

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We implement hereafter methods related to CP, with a relaxation
of the exchangeability assumption.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


class EnbPI:
    """Ensemble batch prediction intervals method

    :param BasePredictor predictor: object implementing '.fit()' and
        '.predict()' methods
    :param int B: number of bootstrap models
    :param func agg_func_loo: aggregation function of LOO estimators.
    :param int random_state: determines random generation.

    .. NOTE::

        *Xu et al.* defined two aggregation functions of leave-one-out estimators:

            - For `EnbPI v1 <http://proceedings.mlr.press/v139/xu21h.html>`_:
              :code:`lambda x, *args: np.quantile(x, alpha, *args)`
            - For `EnbPI v2 <https://arxiv.org/abs/2010.09107v12>`_: :code:`np.mean`

    Example::

        import numpy as np

        from deel.puncc.regression import EnbPI
        from deel.puncc.api.prediction import BasePredictor

        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor

        from deel.puncc.metrics import regression_mean_coverage
        from deel.puncc.metrics import regression_sharpness


        # Generate a random regression problem
        X, y = make_regression(n_samples=1000, n_features=4, n_informative=2,
                    random_state=0, shuffle=False)

        # Split data into train and test
        X, X_test, y, y_test = train_test_split(
            X, y, test_size=.2, random_state=0
        )

        # Create rf regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
        # Wrap model in a predictor
        rf_predictor = BasePredictor(rf_model)
        # CP method initialization
        enbpi = EnbPI(
            rf_predictor,
            B=30,
            agg_func_loo=np.mean,
            random_state=0,
        )

        # The call to `fit` trains the model and computes the nonconformity
        # scores on the oob calibration sets
        enbpi.fit(X, y)

        # The predict method infers prediction intervals with respect to
        # the significance level alpha = 20%
        y_pred, y_pred_lower, y_pred_upper = enbpi.predict(
            X_test, alpha=.2, y_true=y_test, s=None
        )

        # Compute marginal coverage and average width of the prediction intervals
        coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
        width = regression_sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)

    """

    def __init__(
        self, predictor, B: int, agg_func_loo=np.mean, random_state=None
    ):
        self.predictor = predictor
        self.B = B
        # Aggregation function of LOO predictions
        self.agg_func_loo = agg_func_loo
        # Initialisation of residuals list
        self.residuals = []
        # Boostrapped models list for estimations
        self._boot_predictors = None
        # Randome seed
        self.random_state = random_state

        # Privates
        self._oob_dict = None
        self._boot_dict = None
        self._oob_matrix = None

    def _compute_residuals(self, y_pred, y_true):
        """Residual computation formula.

        :param ndarray y_pred: predicted values.
        :param ndarray y_true: true values.
        :param ndarray sigma_pred: predicted variability.

        :returns: residuals.
        :rtype: ndarray
        """
        return absolute_difference()(y_pred, y_true)

    def _compute_pi(self, y_pred, w):
        """Compute prediction intervals.

        :param ndarray y_pred: predicted values.
        :param ndarray w: residuals' quantiles.

        :returns: prediction intervals.
        :rtype: tuple[ndarray]

        """

        return constant_interval()(y_pred, w)

    def _compute_boot_residuals(self, boot_pred, y_true):
        """Compute residuals w.r.t the boostrap aggregation.

        :param ndarray boot_pred: bootstrapped predicted values.
        :param ndarray y_true: true targets.

        :returns: residuals.
        :rtype: list[ndarray]

        """
        # Approximation of LOO predictions:
        #   For each training sample X_i, the LOO estimate is built from
        #   averaging the predictions of bootstrap models whose OOB include X_i
        loo_pred = (self._oob_matrix * boot_pred.T).sum(-1)
        residuals = absolute_difference()(
            y_pred=loo_pred, y_true=y_true
        )
        return list(residuals)

    def _compute_loo_predictions(self, boot_pred):
        """Compute Leave-One-Out (LOO) predictions from bootstrapped predicitons.

        :param ndarray boot_pred: bootstrapped predicted values.

        :returns: LOO prediction.
        :rtype: ndarray

        """
        return np.matmul(self._oob_matrix, boot_pred)

    def fit(self, X, y, **kwargs):
        """Fit B bootstrap models on the bootstrap bags and respectively
        compute/store residuals on out-of-bag samples.

        :param ndarray X: training feature set
        :param ndarray y: training label set
        :param dict kwargs: fit arguments for the underlying model

        :raises RuntimeError: empty out-of-bag.

        """
        self._oob_dict = {}  # Key: b. Value: out of bag weighted index
        self._boot_predictors = []  # f^_b for b in [1,B]
        T = len(X)  # Number of samples to be considered during training
        horizon_indices = np.arange(T)

        # === (1) === Do bootstrap sampling, reference OOB samples===
        self._boot_dict = {}

        for b in range(self.B):
            # Ensure we don't have pathological bootstrap sampling
            # In case bootstrap is identical to original set, OOB is empty.
            oob_is_empty = True
            random_state_b = (
                None if self.random_state is None else self.random_state + b
            )

            boot = None  # Initialization
            oob_units = None  # Initialization

            # Randomly sample bootstrap sets until the out-of-bag is not empty
            while oob_is_empty:
                boot = resample(
                    horizon_indices,
                    replace=True,
                    n_samples=T,
                    random_state=random_state_b,
                )

                if boot is None:  # sanity check
                    raise RuntimeError("Bootstrap dataset is empty.")

                oob_units = np.setdiff1d(horizon_indices, boot)
                oob_is_empty = len(oob_units) == 0
            # OOB is not empty, proceed
            self._boot_dict[b] = boot
            self._oob_dict[b] = oob_units

        # Create oob_matrix, rows for every i-th training sample
        # and columns for each j-th bootstrap model.
        # Cell value is > 0 if i-th sample is in the j-th OOB set
        # and 0 otherwise.
        self._oob_matrix = np.zeros((T, self.B))

        for i in range(T):
            oobs_for_i_th_unit = [
                1 if i in self._oob_dict[b] else 0 for b in range(self.B)
            ]

            # Verify OOB-ness for all i-th training samples;
            # raise an exception otherwise.
            if np.sum(oobs_for_i_th_unit) == 0:
                raise RuntimeError(
                    f"Training sample {i} is included in all boostrap sets."
                    + ' Increase "B", the number of boostrap models.'
                )

            self._oob_matrix[i] = oobs_for_i_th_unit

        # oob matrix normalization: the sum of each rows is made equal to 1.
        self._oob_matrix /= np.tile(
            np.sum(self._oob_matrix, axis=1), (self.B, 1)
        ).transpose()

        # === (2) === Fit predictors on bootstrapped samples
        # print(" === step 1/2: fitting bootstrap estimators ...")

        for b in range(self.B):
            # Retrieve list of indexes of previously bootstrapped sample
            boot = self._boot_dict[b]
            boot_predictor = self.predictor.copy()  # Instantiate model
            boot_predictor.fit(X[boot], y[boot], **kwargs)  # fit predictor
            self._boot_predictors.append(boot_predictor)  # Store fitted model

        # === (3) === Residuals computation
        # print(" === step 2/2: computing nonconformity scores ...")
        # Predictions on X by each bootstrap estimator
        boot_preds = [
            self._boot_predictors[b].predict(X) for b in range(self.B)
        ]
        boot_preds = np.array(boot_preds)
        residuals = self._compute_boot_residuals(boot_preds, y)
        self.residuals += residuals

    def predict(
        self, X_test, alpha=0.1, y_true=None, s=None
    ) -> tuple[np.ndarray]:
        """Estimate conditional mean and interval prediction.

        :param ndarray X_test: features of new samples.
        :param ndarray y_true: if not None, residuals update based on
            seasonality is performed.
        :param float alpha: target maximum miscoverage.
        :param int s: Number of online samples necessary to update the
            residuals sequence.

        :returns: A tuple composed of y_pred (conditional mean), y_pred_lower
            (lower PI bound) and y_pred_upper (upper PI bound).
        :rtype: tuple[ndarray]

        """
        y_pred_upper_list = []
        y_pred_lower_list = []
        y_pred_list = []
        updated_residuals = list(deepcopy(self.residuals))

        # WARNING: following the paper of Xu et al 2021,
        # we should __NOT__ look for the (1-alpha)(1+1/N) empirical quantile, unlike with
        # proper Conformal Prediction: __it seems__ like we only care about estimating the
        # (1-alpha) quantile.
        #
        # That is, we do not need to compute the quantile from the empirical CDF of
        # errors, but we can use estimation techniques.
        #
        # Here, using the default implementation of numpy.quantile(), we use
        # the argument: np.quantile(..., method='linear').
        #
        # TODO: go back to EnbPI-v1 paper and double check what above.
        #
        res_quantile = np.quantile(self.residuals, (1 - alpha), method="linear")

        if y_true is None or (y_true is not None and s is None):
            n_batches = 1
            s = len(X_test)
        elif y_true is not None and s is not None:
            n_batches = len(y_true) // s
        else:
            raise RuntimeError("Cannot determine batch size.")

        if self._boot_predictors is None:  # Sanity check
            raise RuntimeError("Fatal error: _boot_predictors is None.")

        # Inference is performed by batch
        for i in np.arange(n_batches):
            if i == n_batches - 1:
                X_batch = X_test[i * s :]
                y_true_batch = y_true[i * s :] if y_true is not None else None
            else:
                X_batch = X_test[i * s : (i + 1) * s]
                y_true_batch = (
                    y_true[i * s : (i + 1) * s] if y_true is not None else None
                )
            # Matrix containing batch predictions of each bootstrap model
            boot_preds = np.array(
                [
                    self._boot_predictors[b].predict(X_batch)
                    for b in range(self.B)
                ]
            )
            # Approximation of LOO predictions
            loo_preds = self._compute_loo_predictions(boot_preds)
            # Ensemble prediction based on the aggregation of LOO estimations
            y_pred_batch = self.agg_func_loo(loo_preds, axis=0)
            if y_pred_batch.shape == ():  # Case predicting a single outcome
                y_pred_batch = y_pred_batch[np.newaxis, ...]
            y_pred_batch_lower, y_pred_batch_upper = self._compute_pi(
                y_pred_batch, res_quantile
            )
            # Update prediction / PI lists for the current batch
            y_pred_upper_list += list(y_pred_batch_upper)
            y_pred_lower_list += list(y_pred_batch_lower)
            y_pred_list += list(y_pred_batch)

            # Update residuals
            if y_true is not None:
                residuals = self._compute_residuals(y_pred_batch, y_true_batch)
                updated_residuals = updated_residuals[s:]
                updated_residuals += list(residuals)
                res_quantile = np.quantile(updated_residuals, (1 - alpha))

        return (
            np.array(y_pred_list),
            np.array(y_pred_lower_list),
            np.array(y_pred_upper_list),
        )


class AdaptiveEnbPI(EnbPI):
    """Locally adaptive version ensemble batch prediction intervals method.

    :param MeanVarPredictor predictor: object implementing '.fit()' and
        '.predict()' methods
    :param int B: number of bootstrap models
    :param func agg_func_loo: aggregation function of LOO estimators.
    :param int random_state: determines random generation.

    .. note::

        *Xu et al.* defined two aggregation functions of leave-one-out estimators:

            - For `EnbPI v1 <http://proceedings.mlr.press/v139/xu21h.html>`_:
              :code:`lambda x, *args: np.quantile(x, alpha, *args)`
            - For `EnbPI v2 <https://arxiv.org/abs/2010.09107v12>`_: :code:`np.mean`

    Example::

        from deel.puncc.regression import AdaptiveEnbPI
        from deel.puncc.api.prediction import MeanVarPredictor

        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor

        from deel.puncc.metrics import regression_mean_coverage
        from deel.puncc.metrics import regression_sharpness


        # Generate a random regression problem
        X, y = make_regression(n_samples=1000, n_features=4, n_informative=2,
                    random_state=0, shuffle=False)

        # Split data into train and test
        X, X_test, y, y_test = train_test_split(
            X, y, test_size=.2, random_state=0
        )

        # Create two models mu (mean) and sigma (dispersion)
        mean_model = RandomForestRegressor(n_estimators=100, random_state=0)
        sigma_model = RandomForestRegressor(n_estimators=100, random_state=0)

        # Wrap models in a mean/variance predictor
        mean_var_predictor = MeanVarPredictor([mean_model, sigma_model])

        # CP method initialization
        aenbpi = AdaptiveEnbPI(
            mean_var_predictor,
            B=30,
            agg_func_loo=np.mean,
            random_state=0,
        )

        # The call to `fit` trains the model and computes the nonconformity
        # scores on the oob calibration sets
        aenbpi.fit(X, y)

        # The predict method infers prediction intervals with respect to
        # the significance level alpha = 20%
        y_pred, y_pred_lower, y_pred_upper = aenbpi.predict(
            X_test, alpha=.2, y_true=y_test, s=None
        )

        # Compute marginal coverage and average width of the prediction intervals
        coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
        width = regression_sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)

    """

    def _compute_pi(self, y_pred, w) -> tuple[np.ndarray]:
        """Compute prediction intervals.

        :param ndarray y_pred: predicted values and variabilities.
        :param ndarray w: residuals' quantiles.

        :returns: prediction intervals.
        :rtype: tuple[ndarray]

        """

        return scaled_interval()(y_pred, w)

    def _compute_residuals(self, y_pred, y_true):
        """Residual computation formula.

        :param ndarray y_pred: predicted values.
        :param ndarray y_true: true values.
        :param ndarray sigma_pred: predicted variability.

        :returns: residuals.
        :rtype: ndarray

        """
        return scaled_ad()(y_pred, y_true)

    def _compute_boot_residuals(self, boot_pred, y_true):
        loo_pred = (self._oob_matrix * boot_pred[:, :, 0].T).sum(-1)
        loo_sigma = (self._oob_matrix * boot_pred[:, :, 1].T).sum(-1)
        y_pred = np.stack((loo_pred, loo_sigma), axis=-1)
        residuals = self._compute_residuals(y_pred=y_pred, y_true=y_true)
        return list(residuals)

    def _compute_loo_predictions(self, boot_pred):
        """Compute Leave-One-Out (LOO) predictions from bootstrapped predicitons.

        :param ndarray boot_pred: bootstrapped predicted values.

        :returns: LOO prediction.
        :rtype: ndarray

        """
        loo_mean = np.matmul(self._oob_matrix, boot_pred[:, :, 0])
        loo_sigma = np.matmul(self._oob_matrix, boot_pred[:, :, 1])
        return np.stack((loo_mean, loo_sigma), axis=-1)

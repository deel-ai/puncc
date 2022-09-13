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
This module implements usual conformal prediction wrappers.
"""
from copy import deepcopy
from typing import Tuple

import numpy as np
from sklearn.utils import resample
from tqdm import tqdm

from deel.puncc.api.calibration import MeanCalibrator
from deel.puncc.api.calibration import MeanVarCalibrator
from deel.puncc.api.calibration import QuantileCalibrator
from deel.puncc.api.conformalization import ConformalPredictor
from deel.puncc.api.prediction import MeanPredictor
from deel.puncc.api.prediction import MeanVarPredictor
from deel.puncc.api.prediction import QuantilePredictor
from deel.puncc.api.splitting import IdSplitter
from deel.puncc.api.splitting import KFoldSplitter


class BaseSplit:
    """Interface of conformal prediction methods based on a split calibration/test scheme.

    :param deel.puncc.prediction.BasePredictor predictor: point-based or interval-based model wrapper.
    :param deel.puncc.calibration.BaseCalibrator calibrator: nonconformity computation strategy and interval predictor.
    :param deel.puncc.prediction.BaseSplitter splitter: fit/calibration split strategy.
    :param boom train: if False, prediction model(s) will not be trained and will be used as is.
    """

    def __init__(self, predictor, calibrator, train):
        self.predictor = predictor
        self.calibrator = calibrator
        self.train = train

    def fit(
        self,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ):
        """This method fits the models to the fit data (X_fit, y_fit)
        and computes residuals on (X_calib, y_calib).

        :param ndarray X_fit: features from the fit dataset.
        :param ndarray y_fit: labels from the fit dataset.
        :param ndarray X_calib: features from the calibration dataset.
        :param ndarray y_calib: labels from the calibration dataset.
        """
        id_gen = IdSplitter(X_fit, y_fit, X_calib, y_calib)
        self.conformal_predictor = ConformalPredictor(
            predictor=self.predictor,
            calibrator=self.calibrator,
            splitter=id_gen,
        )
        self.conformal_predictor.fit(X=None, y=None)  # type: ignore

    def predict(self, X_test, alpha):
        raise NotImplementedError


class SplitCP(BaseSplit):
    """Split conformal prediction method.

    :param object mu_model: conditional mean model.
    :param bool train: if False, prediction model(s) will not be trained and will be used as is. Defaults to True.

    """

    def __init__(self, mu_model, train=True):
        super().__init__(MeanPredictor(mu_model), MeanCalibrator(), train=train)

    def predict(self, X_test, alpha) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate conditional mean and prediction interval
        (w.r.t target miscoverage alpha) for new samples.

        :param ndarray X_test: features of new samples.
        :param float alpha: target maximum miscoverage.

        :returns: A tuple composed of y_pred (conditional mean), y_pred_lower (lower PI bound) and y_pred_upper (upper PI bound).
        :rtype: tuple[ndarray, ndarray, ndarray]
        """
        (
            y_pred,
            y_lo,
            y_hi,
            var_pred,
        ) = self.conformal_predictor.predict(X_test, alpha=alpha)

        return y_pred, y_lo, y_hi


class WeightedSplitCP(BaseSplit):
    """Weighted split conformal prediction method.

    :param object mu_model: conditional mean model.
    :param object w_estimator: weight estimator of nonconformity scores distribution, defaults to None (equal weights).
    :param str weight_method: method to normalize weights: ['barber', 'tibshirani'], defaults to 'barber'
    :param bool train: if False, prediction model(s) will not be trained and will be used as is. Defaults to True.
    """

    def __init__(self, mu_model, w_estimator=None, weight_method="barber", train=True):

        self.predictor = MeanPredictor(mu_model)

        if weight_method not in ("barber", "tibshirani"):
            error_msg = f"{weight_method} is not implemented. Please choose either 'barber' or 'tibshirani'"
            raise NotImplementedError(error_msg)

        self.calibrator = MeanCalibrator(
            weight_func=w_estimator, weight_method=weight_method
        )
        super().__init__(
            predictor=self.predictor, calibrator=self.calibrator, train=train
        )

    def predict(self, X_test, alpha) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate conditional mean and prediction interval
        (w.r.t target miscoverage alpha) for new samples.

        :param ndarray X_test: features of new samples.
        :param float alpha: target maximum miscoverage.

        :returns: A tuple composed of y_pred (conditional mean), y_pred_lower (lower PI bound) and y_pred_upper (upper PI bound).
        :rtype: tuple[ndarray, ndarray, ndarray]
        """
        (
            y_pred,
            y_lo,
            y_hi,
            var_pred,
        ) = self.conformal_predictor.predict(X_test, alpha=alpha)
        return y_pred, y_lo, y_hi


class LocallyAdaptiveCP(BaseSplit):
    """Locally adaptive conformal prediction method.

    :param object mu_model: conditional mean model.
    :param object var_model: mean absolute deviation model.
    :param bool train: if False, prediction model(s) will not be trained and will be used as is. Defaults to True.
    """

    def __init__(self, mu_model, var_model, train=True):

        self.predictor = MeanVarPredictor(mu_model=mu_model, var_model=var_model)
        self.calibrator = MeanVarCalibrator()
        super().__init__(
            predictor=self.predictor, calibrator=self.calibrator, train=train
        )

    def predict(
        self, X_test, alpha
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Estimate conditional mean, mean absolute deviation and prediction interval (w.r.t target miscoverage alpha) for new samples.

        :param ndarray X_test: features of new samples.
        :param float alpha: target maximum miscoverage.

        :returns: A tuple composed of y_pred (conditional mean), y_pred_lower (lower PI bound), y_pred_upper (upper PI bound) and var_pred (absolute mean deviation).
        :rtype: tuple[ndarray, ndarray, ndarray, ndarray]
        """
        (
            y_pred,
            y_pred_lower,
            y_pred_upper,
            var_pred,
        ) = self.conformal_predictor.predict(X_test, alpha=alpha)
        return (y_pred, y_pred_lower, y_pred_upper, var_pred)


class CQR(BaseSplit):
    """Conformalized quantile regression method.

    :param object q_hi_model: higher quantile model.
    :param object q_lo_model: lower quantile model.
    :param bool train: if False, prediction model(s) will not be trained and will be used as is. Defaults to True.
    """

    def __init__(self, q_hi_model, q_lo_model, train: bool = True):

        self.predictor = QuantilePredictor(q_hi_model=q_hi_model, q_lo_model=q_lo_model)
        self.calibrator = QuantileCalibrator()
        super().__init__(
            predictor=self.predictor, calibrator=self.calibrator, train=train
        )

    def predict(
        self, X_test: np.ndarray, alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate prediction intervals (w.r.t target miscoverage alpha) for new samples.

        :param ndarray X_test: features of new samples.
        :param float alpha: target maximum miscoverage.

        :returns: A tuple composed of y_pred_lower (lower PI bound) and y_pred_upper (upper PI bound).
        :rtype: tuple[ndarray, ndarray]
        """
        (
            y_pred,
            y_pred_lower,
            y_pred_upper,
            var_pred,
        ) = self.conformal_predictor.predict(X_test, alpha=alpha)
        return y_pred_lower, y_pred_upper


class CvPlus:
    """Cross-validation plus method.

    :param object mu_model: conditional mean model.
    :param int K: number of training/calibration folds.
    :param int random_state: seed to control random folds.
    """

    def __init__(self, mu_model, K: int, random_state=None):

        self.predictor = MeanPredictor(mu_model)
        self.calibrator = MeanCalibrator()
        kfold_splits = KFoldSplitter(K=K, random_state=random_state)
        self.conformal_predictor = ConformalPredictor(
            predictor=self.predictor,
            calibrator=self.calibrator,
            splitter=kfold_splits,
            method="cv+",
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """This method fits the ensemble models based on the K-fold plan.
        The out-of-bag folds are used to computes residuals on (X_calib, y_calib).

        :param ndarray X_train: features from the train dataset.
        :param ndarray y_train: labels from the train dataset.
        """
        self.conformal_predictor.fit(X_train, y_train)

    def predict(
        self, X_test: np.ndarray, alpha: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate conditional mean and prediction interval
        (w.r.t target miscoverage alpha) for new samples.

        :param ndarray X_test: features of new samples.
        :param float alpha: target maximum miscoverage.

        :returns: A tuple composed of y_pred (conditional mean), y_pred_lower (lower PI bound) and y_pred_upper (upper PI bound).
        :rtype: tuple[ndarray, ndarray, ndarray]
        """
        (
            y_pred,
            y_pred_lower,
            y_pred_upper,
            _,
        ) = self.conformal_predictor.predict(X_test, alpha=alpha)
        return (
            y_pred,
            y_pred_lower,
            y_pred_upper,
        )


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We implement hereafter methods related to CP, with a relaxation
of the exchangeability assumption.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


class EnbPI:
    """Ensemble batch prediction intervals method

    :param object model: object implementing '.fit()' and '.predict()' methods
    :param int B: number of bootstrap models
    :param func agg_func_loo: aggregation function of LOO estimators.
    :param int random_state: determines random generation.

    .. note::
        *Xu et al.* defined two aggregation functions of leave-one-out estimators:
            * For `EnbPI v1 <http://proceedings.mlr.press/v139/xu21h.html>`_: :code:`lambda x, *args: np.quantile(x, alpha, *args)`
            * For `EnbPI v2 <https://arxiv.org/abs/2010.09107v12>`_: :code:`np.mean`
    """

    def __init__(self, model, B: int, agg_func_loo=np.mean, random_state=None):
        self.model = model
        self.B = B
        # Aggregation function of LOO predictions
        self.agg_func_loo = agg_func_loo
        # Initialisation of residuals list
        self.residuals = list()
        # Boostrapped models list for conditional mean estimation
        self._boot_estimators = None
        # Boostrapped models list for mean absolute deviation estimation
        self._boot_sigma_estimators = None
        # Randome seed
        self.random_state = random_state

    def _fit_callback(self, X_train=None, y_train=None, b=None):
        """Callback process for each iteration of the ensemble training.
        To be defined if further processing is needed.

        :param ndarray X_train: batch training features
        :param ndarray y_train: batch training targets
        :param int b: iteration index of the bagging
        """
        pass

    def _predict_callback(self, X_test=None):
        """Callback process for additional predictions.
        To be defined if further processing is needed.

        :param ndarray X_train: batch training features.
        :param ndarray y_train: batch training targets.
        :param int b: iteration index of the bagging.
        """
        pass

    def _compute_pi(self, y_pred, w, *args):
        """Compute prediction intervals.

        :param ndarray y_pred: predicted values.
        :param ndarray w: residuals' quantiles.

        :returns: prediction intervals.
        :rtype: tuple[ndarray, ndarray]
        """
        y_pred_batch_upper = y_pred + w
        y_pred_batch_lower = y_pred - w
        return y_pred_batch_upper, y_pred_batch_lower

    def _compute_residuals(self, y_pred, y_true, *args):
        """Residual computation formula.

        :param ndarray y_pred: predicted values.
        :param ndarray y_true: true values.

        :returns: residuals.
        :rtype: ndarray
        """
        return np.abs(y_true - y_pred)

    def _compute_boot_residuals(self, X_train, y_train, boot_estimators, *args):
        """Compute residuals w.r.t the boostrap aggregation.
        Args:
        :param ndarray X_train: train features.
        :param ndarray y_train: train targets.
        :param list[objects] boot_estimators: list of bootstrap models.

        :returns: residuals.
        :rtype: list[ndarray]
        """
        # Predictions on X_train by each bootstrap estimator
        boot_preds = [boot_estimators[b].predict(X_train) for b in range(self.B)]
        boot_preds = np.array(boot_preds)
        # Approximation of LOO predictions:
        #   For each training sample X_i, the LOO estimate is built from
        #   averaging the predictions of bootstrap models whose OOB include X_i
        loo_pred = (self._oob_matrix * boot_preds.T).sum(-1)
        residuals = self._compute_residuals(y_pred=loo_pred, y_true=y_train)
        return list(residuals)

    def fit(self, X_train, y_train):
        """Fit B bootstrap models on the bootstrap bags and respectively compute/store residuals on out-of-bag samples.

        :param ndarray X_train: training feature set
        :param ndarray y_train: training label set

        :raises RuntimeError: in case of empty out-of-bag.
        """
        self._oob_dict = dict()  # Key: b. Value: out of bag weighted index
        self._boot_estimators = list()  # f^_b for b in [1,B]
        self._boot_sigma_estimators = list()  # sigma^_b for b in [1,B]
        T = len(X_train)  # Number of samples to be considered during training
        horizon_indices = np.arange(T)

        # === (1) === Do bootstrap sampling, reference OOB samples===
        self._boot_dict = dict()

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
        for i in tqdm(range(T)):
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
            else:
                self._oob_matrix[i] = oobs_for_i_th_unit
        # oob matrix normalization: the sum of each rows is made equal to 1.
        self._oob_matrix /= np.tile(
            np.sum(self._oob_matrix, axis=1), (self.B, 1)
        ).transpose()

        # === (2) === Fit predictors on bootstrapped samples
        print(" === step 1/2: fitting bootstrap estimators ...")
        for b in tqdm(range(self.B)):
            # Retrieve list of indexes of previously bootstrapped sample
            boot = self._boot_dict[b]
            boot_estimator = deepcopy(self.model)  # Instantiate model
            boot_estimator.fit(X_train[boot], y_train[boot])  # fit predictor
            self._boot_estimators.append(boot_estimator)  # Store fitted model
            self._fit_callback(
                X_train, y_train, b
            )  # Callback for further processes, if needed

        # === (3) === Residuals computation
        print(" === step 2/2: computing nonconformity scores ...")
        residuals = self._compute_boot_residuals(
            X_train,
            y_train,
            self._boot_estimators,
            self._boot_sigma_estimators,
        )
        self.residuals += residuals

    def predict(self, X_test, alpha=0.1, y_true=None, s=None):
        """Estimate conditional mean and interval prediction.

        :param ndarray X_test: features of new samples.
        :param ndarray y_true: if not None, residuals update based on seasonality is performed.
        :param float alpha: target maximum miscoverage.
        :param int s: Number of online samples necessary to update the residuals sequence.

        :returns: A tuple composed of y_pred (conditional mean), y_pred_lower (lower PI bound) and y_pred_upper (upper PI bound).
        :rtype: tuple[ndarray, ndarray, ndarray]
        """
        y_pred_upper_list = list()
        y_pred_lower_list = list()
        y_pred_list = list()
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

        if self._boot_estimators is None:  # Sanity check
            raise RuntimeError("Fatal error: _boot_estimators is None.")

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
                [self._boot_estimators[b].predict(X_batch) for b in range(self.B)]
            )
            # Approximation of LOO predictions
            loo_preds = np.matmul(self._oob_matrix, boot_preds)
            # Ensemble prediction based on the aggregation of LOO estimations
            y_pred_batch = self.agg_func_loo(loo_preds, axis=0)
            # Auxiliary prediction computed by the predict callback
            # Will be passed as argument for computing prediction intervals
            aux_pred_batch = self._predict_callback(X_batch)

            y_pred_batch_upper, y_pred_batch_lower = self._compute_pi(
                y_pred_batch, res_quantile, aux_pred_batch
            )

            # Update prediction / PI lists for the current batch
            y_pred_upper_list += list(y_pred_batch_upper)
            y_pred_lower_list += list(y_pred_batch_lower)
            y_pred_list += list(y_pred_batch)

            # TODO: see comment above. We probably should remove the
            # correction (1-alpha)(1+1/ncalib) to just (1-alpha).
            # Xu uses different theory, not needing inflating quantiles
            # to get finite sample guarantees and all the conformal machinery
            # in place
            #
            # Update residuals
            if y_true is not None:
                residuals = self._compute_residuals(
                    y_pred_batch, y_true_batch, aux_pred_batch
                )
                updated_residuals = updated_residuals[s:]
                updated_residuals += list(residuals)
                res_quantile = np.quantile(
                    updated_residuals,
                    (1 - alpha) * (1 + 1 / len(updated_residuals)),
                )

        return (
            np.array(y_pred_list),
            np.array(y_pred_lower_list),
            np.array(y_pred_upper_list),
        )


class AdaptiveEnbPI(EnbPI):
    """Locally adaptive version ensemble batch prediction intervals method.

    :param object model: object implementing '.fit()' and '.predict()' methods
    :param object dispersion_model: variability model
    :param int B: number of bootstrap models
    :param func agg_func_loo: aggregation function of LOO estimators.
    :param int random_state: determines random generation.

    .. note::
        *Xu et al.* defined two aggregation functions of leave-one-out estimators:
            * For `EnbPI v1 <http://proceedings.mlr.press/v139/xu21h.html>`_: :code:`lambda x, *args: np.quantile(x, alpha, *args)`
            * For `EnbPI v2 <https://arxiv.org/abs/2010.09107v12>`_: :code:`np.mean`
    """

    def __init__(
        self,
        model,
        dispersion_model,
        B,
        agg_func_loo=np.mean,
        random_state=None,
    ):
        super().__init__(model, B, agg_func_loo, random_state=random_state)
        self.dispersion_model = dispersion_model

    def _compute_pi(self, y_pred, w, sigma_pred):
        """Compute prediction intervals.

        :param ndarray y_pred: predicted values.
        :param ndarray w: residuals' quantiles.
        :param ndarray sigma_pred: predicted variability.

        :returns: prediction intervals.
        :rtype: tuple[ndarray, ndarray]
        """

        y_pred_batch_upper = y_pred + w * sigma_pred
        y_pred_batch_lower = y_pred - w * sigma_pred
        return y_pred_batch_upper, y_pred_batch_lower

    def _compute_residuals(self, y_pred, y_true, sigma_pred):
        """Residual computation formula.

        :param ndarray y_pred: predicted values.
        :param ndarray y_true: true values.
        :param ndarray sigma_pred: predicted variability.

        :returns: residuals.
        :rtype: ndarray
        """
        return np.abs(y_true - y_pred) / sigma_pred

    def _compute_boot_residuals(
        self, X_train, y_train, boot_estimators, boot_disp_estimators
    ):
        boot_preds = np.array(
            [boot_estimators[b].predict(X_train) for b in range(self.B)]
        )
        boot_sigmas = np.array(
            [boot_disp_estimators[b].predict(X_train) for b in range(self.B)]
        )
        loo_pred = (self._oob_matrix * boot_preds.T).sum(-1)
        loo_sigma = (self._oob_matrix * boot_sigmas.T).sum(-1)

        residuals = self._compute_residuals(
            y_pred=loo_pred, sigma_pred=loo_sigma, y_true=y_train
        )
        return list(residuals)

    def _fit_callback(self, X_train, y_train, b):
        # retrieve list of indexes of previously bootstrapped sample
        boot = self._boot_dict[b]
        if self._boot_estimators is None:  # Sanity check
            raise RuntimeError("Fatal error: _boot_estimators is None.")
        f_hat_b = self._boot_estimators[b]
        # Fit dispersion predictor
        sigma_hat_b = deepcopy(self.dispersion_model)  # Instantiate model
        sigmas = np.abs(y_train[boot] - f_hat_b.predict(X_train[boot]))
        sigma_hat_b.fit(X_train[boot], sigmas)
        # Sanity check
        if self._boot_sigma_estimators is None:
            raise RuntimeError("Fatal error: _boot_estimators is None.")
        self._boot_sigma_estimators.append(sigma_hat_b)  # Store fitted model

    def _predict_callback(self, X_test=None):
        # Sanity check
        if self._boot_sigma_estimators is None:
            raise RuntimeError("Fatal error: _boot_estimators is None.")
        sigma_bs = np.array(
            [self._boot_sigma_estimators[b].predict(X_test) for b in range(self.B)]
        )
        sigma_phi_x_loos = np.matmul(self._oob_matrix, sigma_bs)
        sigma_pred = np.mean(sigma_phi_x_loos, axis=0)
        return sigma_pred

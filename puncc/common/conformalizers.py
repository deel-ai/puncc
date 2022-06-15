"""
This module implements usual conformal prediction wrappers.
"""


from copy import deepcopy
import numpy as np
from typing import Tuple
from tqdm import tqdm
from sklearn.utils import resample
from puncc.calibration import (
    MeanCalibrator,
    MeanVarCalibrator,
    QuantileCalibrator,
)
from puncc.prediction import MeanPredictor, MeanVarPredictor, QuantilePredictor
from puncc.conformalization import ConformalPredictor
from puncc.splitting import IdSplitter, KFoldSplitter
from puncc.utils import agg_func


class BaseSplit:
    """
    Attributes:
        predictor: point-based or interval-based model wrapper
        calibrator: nonconformity computation strategy and interval predictor
        splitter: fit/calibration split strategy
        train: if False, prediction model(s) will not be trained and will
               be used as is

    """

    def __init__(self, train):
        self.predictor = None
        self.calibrator = None
        self.conformal_predictor = None
        self.train = train

    def fit(
        self,
        X_fit: np.array,
        y_fit: np.array,
        X_calib: np.array,
        y_calib: np.array,
    ):
        """This method fits the models to the fit data (X_fit, y_fit)
        and computes residuals on (X_calib, y_calib).

        Args:
            X_fit: features from the fit dataset
            y_fit: labels from the fit dataset
            X_calib: features from the calibration dataset
            y_calib: labels from the calibration dataset

        """
        id_gen = IdSplitter(X_fit, y_fit, X_calib, y_calib)
        self.conformal_predictor = ConformalPredictor(
            predictor=self.predictor,
            calibrator=self.calibrator,
            splitter=id_gen,
        )
        self.conformal_predictor.fit(None, None)

    def predict(self, X_test, alpha):
        raise NotImplementedError


class SplitCP(BaseSplit):
    """Split Conformal Prediction wrapper."""

    def __init__(self, mu_model, train=True):
        """Constructor.

        Args:
            mu_model: Conditional mean model
            train: if False, prediction model(s) will not be trained and will
                   be used as is
        """
        super().__init__(train=train)
        self.predictor = MeanPredictor(mu_model)
        self.calibrator = MeanCalibrator()
        self.conformal_predictor = None

    def predict(self, X_test, alpha) -> Tuple[np.array]:
        """Estimate conditional mean and prediction interval
        (w.r.t target miscoverage alpha) for new samples.

        Args:
            X_test: features of new samples
            alpha: target miscoverage

        Returns:
            A tuple composed of:
                y_pred (conditional mean),
                y_pred_lower (lower PI bound),
                y_pred_upper (upper PI bound)
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


class WeightedSplitCP(BaseSplit):
    """Split Conformal Prediction wrapper."""

    def __init__(self, mu_model, w_estimator, train=True):
        """Constructor.

        Args:
            mu_model: Conditional mean model
            w_estimator: weight estimator of nonconformity scores distribution
            train: if False, prediction model(s) will not be trained and will
                   be used as is
        """
        super().__init__(train=train)
        self.predictor = MeanPredictor(mu_model)
        self.calibrator = MeanCalibrator(w_estimator=w_estimator)
        self.conformal_predictor = None

    def predict(self, X_test, alpha) -> Tuple[np.array]:
        """Estimate conditional mean and prediction interval
        (w.r.t target miscoverage alpha) for new samples.

        Args:
            X_test: features of new samples
            alpha: target miscoverage

        Returns:
            A tuple composed of:
                y_pred (conditional mean),
                y_pred_lower (lower PI bound),
                y_pred_upper (upper PI bound)
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


class LocallyAdaptiveCP(BaseSplit):
    """Locally Adaptive Conformal Prediction wrapper."""

    def __init__(self, mu_model, sigma_model, train=True):
        """Constructor.

        Args:
            mu_model: conditional mean model
            sigma_model: mean absolute deviation model
            train: if False, prediction model(s) will not be trained and will
                   be used as is
        """
        super().__init__(train=train)
        self.predictor = MeanVarPredictor(
            mu_model=mu_model, sigma_model=sigma_model
        )
        self.calibrator = MeanVarCalibrator()
        self.conformal_predictor = None

    def predict(self, X_test, alpha) -> Tuple[np.array]:
        """Estimate conditional mean, mean absolute deviation,
        and prediction interval (w.r.t target miscoverage alpha)
        for new samples.

        Args:
            X_test: features of new samples
            alpha: target miscoverage

        Returns:
            A tuple composed of:
                y_pred (conditional mean),
                y_pred_lower (lower PI bound),
                y_pred_upper (upper PI bound),
                sigma_pred (mean absolute deviation)
        """
        (
            y_pred,
            y_pred_lower,
            y_pred_upper,
            sigma_pred,
        ) = self.conformal_predictor.predict(X_test, alpha=alpha)
        return (y_pred, y_pred_lower, y_pred_upper, sigma_pred)


class CQR(BaseSplit):
    """Conformalized Quantile Regression wrapper."""

    def __init__(self, q_hi_model, q_lo_model, train=True):
        """Constructor.

        Args:
            q_hi_model: higher quantile model
            q_lo_model: lower quantile model
            train: if False, prediction model(s) will not be trained and will
                   be used as is
        """
        super().__init__(train=train)
        self.predictor = QuantilePredictor(
            q_hi_model=q_hi_model, q_lo_model=q_lo_model
        )
        self.calibrator = QuantileCalibrator()
        self.conformal_predictor = None

    def predict(self, X_test, alpha):
        """Estimate prediction interval (w.r.t target miscoverage alpha)
        for new samples.

        Args:
            X_test: features of new samples
            alpha: target miscoverage

        Returns:
            A tuple composed of:
                y_pred_lower (lower PI bound),
                y_pred_upper (upper PI bound),
        """
        (
            _,
            y_pred_lower,
            y_pred_upper,
            _,
        ) = self.conformal_predictor.predict(X_test, alpha=alpha)
        return (y_pred_lower, y_pred_upper)


class CvPlus:
    """Cross-validation plus wrapper."""

    def __init__(self, mu_model, K, train=True, random_state=None):
        """Constructor.

        Args:
            mu_model: conditional mean model
            K: number of training/calibration folds
            train: if False, prediction model(s) will not be trained and will
                   be used as is
            random_state: seed to control random folds

        """
        self.predictor = MeanPredictor(mu_model)
        self.calibrator = MeanCalibrator()
        kfold_splits = KFoldSplitter(K=K, random_state=random_state)
        self.conformal_predictor = ConformalPredictor(
            predictor=self.predictor,
            calibrator=self.calibrator,
            splitter=kfold_splits,
            agg_func=agg_func,
        )

    def fit(self, X_train, y_train):
        """This method fits the ensemble models based on the K-fold plan.
        The out-of-bag folds are used to computes residuals
        on (X_calib, y_calib).

        Args:
            X_train: features from the train dataset
            y_train: labels from the train dataset
        """
        self.conformal_predictor.fit(X_train, y_train)

    def predict(self, X_test, alpha):
        """Estimate conditional mean and prediction interval
        (w.r.t target miscoverage alpha) for new samples.

        Args:
            X_test: features of new samples
            alpha: target miscoverage

        Returns:
            A tuple composed of:
                y_pred (conditional mean),
                y_pred_lower (lower PI bound),
                y_pred_upper (upper PI bound)
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
    """Ensemble Batch Prediction Intervals method"""

    def __init__(
        self, model, B: int, agg_func_loo=np.mean, random_state: int = None
    ):
        """constructor

        Args:
            model: object implementing '.fit()' and '.predict()' methods
            B: number of bootstrap models
            agg_func_loo: aggregation function of LOO estimators.
              - For EnbPI v1 ICML 2021
                http://proceedings.mlr.press/v139/xu21h.html:
                lambda x, *args: np.quantile(x, (1-alpha)*(1+1/len(x)), *args)
              - For EnbPI v2 (https://arxiv.org/abs/2010.09107v12): np.mean
            random_state: determines random generation
        """
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
        To be defined by subclasses if further processing is needed.

        Args:
            X_train: batch training features
            y_train: batch training targets
            b: iteration index of the bagging
        """
        pass

    def _predict_callback(self, X_test=None):
        """Callback process for additional predictions.
        To be defined by subclasses if further processing is needed.

        Args:
            X_train: batch training features
            y_train: batch training targets
            b: iteration index of the bagging
        """
        pass

    def _compute_pi(self, y_pred, w, *args):
        """Compute prediction intervals.
        To be modified by subclasses if different processing is needed."""
        y_pred_batch_upper = y_pred + w
        y_pred_batch_lower = y_pred - w
        return y_pred_batch_upper, y_pred_batch_lower

    def _compute_residuals(self, y_pred, y_true, *args):
        """Residual computation formula"""
        return np.abs(y_true - y_pred)

    def _compute_boot_residuals(
        self, X_train, y_train, boot_estimators, *args
    ):
        """Compute residuals w.r.t the boostrap aggregation.
        Args:
            X_train: train features
            y_train: train targets
            boot_estimators: list of bootstrap models
        """
        # Predictions on X_train by each bootstrap estimator
        boot_preds = [
            boot_estimators[b].predict(X_train) for b in range(self.B)
        ]
        boot_preds = np.array(boot_preds)
        # Approximation of LOO predictions:
        #   For each training sample X_i, the LOO estimate is built from
        #   averaging the predictions of bootstrap models whose OOB include X_i
        loo_pred = (self._oob_matrix * boot_preds.T).sum(-1)
        residuals = self._compute_residuals(y_pred=loo_pred, y_true=y_train)
        return list(residuals)

    def fit(self, X_train, y_train):
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
            while oob_is_empty:
                boot = resample(
                    horizon_indices,
                    replace=True,
                    n_samples=T,
                    random_state=self.random_state + b,
                )
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
                raise Exception(
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

        Args:
            X_test: features of new samples
            alpha: miscoverage level, acceptable statistical error
            y_true: if not None, residuals update based on seasonality is
                    performed
            s: Number of online samples necessary to update the residuals
               sequence
        """
        y_pred_upper_list = list()
        y_pred_lower_list = list()
        y_pred_list = list()
        updated_residuals = list(deepcopy(self.residuals))

        # Residuals 1-alpha th quantile
        res_quantile = np.quantile(
            self.residuals, (1 - alpha) * (1 + 1 / len(self.residuals))
        )

        if y_true is None or (y_true is not None and s is None):
            n_batches = 1
            s = len(X_test)

        elif y_true is not None and s is not None:
            n_batches = len(y_true) // s

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
                    self._boot_estimators[b].predict(X_batch)
                    for b in range(self.B)
                ]
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
    # """Ensemble Batch Prediction Intervals method, Locally Adaptive version
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
        To be modified by subclasses if different processing is needed."""
        y_pred_batch_upper = y_pred + w * sigma_pred
        y_pred_batch_lower = y_pred - w * sigma_pred
        return y_pred_batch_upper, y_pred_batch_lower

    def _compute_residuals(self, y_pred, y_true, sigma_pred):
        return np.abs(y_true - y_pred) / sigma_pred

    def _compute_boot_residuals(
        self, X_train, y_train, boot_estimators, boot_disp_estimators
    ):
        """Compute residuals w.r.t the boostrapping.
        Args:
            X_train: train features
            y_train: train targets
            boot_estimators: list of bootstrap models
            boot_disp_estimators: list of bootstrap dispersion models
        """
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
        f_hat_b = self._boot_estimators[b]
        # Fit dispersion predictor
        sigma_hat_b = deepcopy(self.dispersion_model)  # Instantiate model
        sigmas = np.abs(y_train[boot] - f_hat_b.predict(X_train[boot]))
        sigma_hat_b.fit(X_train[boot], sigmas)
        self._boot_sigma_estimators.append(sigma_hat_b)  # Store fitted model

    def _predict_callback(self, X_test=None):
        sigma_bs = np.array(
            [
                self._boot_sigma_estimators[b].predict(X_test)
                for b in range(self.B)
            ]
        )
        sigma_phi_x_loos = np.matmul(self._oob_matrix, sigma_bs)
        sigma_pred = np.mean(sigma_phi_x_loos, axis=0)
        return sigma_pred

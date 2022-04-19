"""
This module implements usual conformal prediction wrappers.
"""


from puncc.calibration import MeanCalibrator, MeanVarCalibrator, QuantileCalibrator
from puncc.predictor import MeanPredictor, MeanVarPredictor, QuantilePredictor
from puncc.conformalization import ConformalPredictor
from copy import deepcopy
import numpy as np
from typing import Tuple
from tqdm import tqdm
from sklearn.utils import resample
from puncc.utils import identity_split, kfold_random_split


class BaseSplit:
    """
    Attributes:
        predictor: point-based or interval-based model wrapper
        calibrator: nonconformity computation strategy and interval predictor
        splitter: fit/calibration split strategy
        train: if False, predictor model will not be trained and will be used as is

    """

    def __init__(self, train):
        self.predictor = None
        self.calibrator = None
        self.conformal_predictor = None
        self.train = train

    def fit(
        self, X_fit: np.array, y_fit: np.array, X_calib: np.array, y_calib: np.array
    ):
        """This method fits the models to the fit data (X_fit, y_fit)
        and computes residuals on (X_calib, y_calib).

        Args:
            X_fit: features from the fit dataset
            y_fit: labels from the fit dataset
            X_calib: features from the calibration dataset
            y_calib: labels from the calibration dataset

        """
        id_gen = identity_split(X_fit, y_fit, X_calib, y_calib)
        self.conformal_predictor = ConformalPredictor(
            predictor=self.predictor,
            calibrator=self.calibrator,
            splitter=id_gen,
            train=self.train,
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
        """
        super().__init__(train=train)
        self.predictor = MeanPredictor(mu_model)
        self.calibrator = MeanCalibrator()
        self.conformal_predictor = None

    def predict(self, X_test, alpha) -> Tuple[np.array]:
        """Estimate conditional mean and prediction interval (w.r.t target miscoverage alpha)
        for new samples.

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
        """
        super().__init__(train=train)
        self.predictor = MeanVarPredictor(
            mu_model=mu_model, sigma_model=sigma_model, gaussian=False
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
        """
        super().__init__(train=train)
        self.predictor = QuantilePredictor(q_hi_model=q_hi_model, q_lo_model=q_lo_model)
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

    def __init__(self, mu_model, K, train=True):
        """Constructor.

        Args:
            mu_model: conditionnal mean model
            K: number of training/calibration folds
        """
        self.predictor = MeanPredictor(mu_model)
        self.calibrator = MeanCalibrator()
        kfold_gen = kfold_random_split(K=K)
        self.conformal_predictor = ConformalPredictor(
            predictor=self.predictor,
            calibrator=self.calibrator,
            splitter=kfold_gen,
            train=train,
        )

    def fit(self, X_train, y_train):
        """This method fits the ensemble models based on the K-fold plan.
        The out-of-bag folds are used to computes residuals on (X_calib, y_calib).

        Args:
            X_train: features from the train dataset
            y_train: labels from the train dataset
        """
        self.conformal_predictor.fit(X_train, y_train)

    def predict(self, X_test, alpha):
        """Estimate conditional mean and prediction interval (w.r.t target miscoverage alpha)
        for new samples.

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


class EnbPI:
    """Ensemble Batch Prediction Intervals method.

    Args:
        model: model (will be duplicated) to be used for ensemble training
        T: time horizon (from most recent training sample) considered the ensemble learning
        B: Number of bootstrap models
    """

    def __init__(self, model, T, B):
        self.model = model
        self.T = T
        self.B = B

    def fit(self, X_train, y_train):
        # Key: b as bootstrap index. Value: out of bag data
        self._oob_dict = dict()
        # Key: b as bootstrap index. Value: bootstrap fitting data
        self._boot_dict = dict()
        self._boot_estimators = list()  # f^_b for b in [1,B]
        horizon_indices = np.arange(self.T)
        print("Step 1/3 ...")
        # Construct bootstrap and out-of-bag sets
        # This is isolated from the fitting loop to identify empty oobs first.
        # Saves time in case an exception is fired.
        for b in range(self.B):
            # Get bootstrap samples with replacement
            boot = resample(horizon_indices, replace=True, n_samples=self.T)
            self._boot_dict[b] = boot
            # Out of bag samples
            self._oob_dict[b] = [x for x in horizon_indices if x not in boot]
            # Raise exception is oob is empty
            if len(self._oob_dict[b]) == 0:
                raise Exception(
                    f"Training sample {b} is included in all boostrap sets. Increase number of boostrap models."
                )
        # Bootstrapping
        for b in tqdm(range(self.B)):
            f_hat_b = deepcopy(self.model)  # Instanciate model
            # Fit model on most recent data
            f_hat_b.fit(X_train[-self._boot_dict[b]], y_train[-self._boot_dict[b]])
            # Store model
            self._boot_estimators.append(f_hat_b)

        # Bootsrap matrix: lines = time index, columns = oob set
        # Cell value is 1 if time index is in the oob set
        print("Step 2/3 ...")
        self._oob_matrix = np.zeros((self.T, self.B))
        for i in tqdm(range(self.T)):
            self._oob_matrix[i] = [
                1 if i in self._oob_dict[b] else 0 for b in range(self.B)
            ]

        self.residuals = list()  # list of residuals
        # Residuals computation
        print("Step 3/3 ...")
        for i in tqdm(range(self.T)):
            f_bs = list()
            x_i = X_train[[-i]]
            for b in range(self.B):
                if self._oob_matrix[i][b] == 1:
                    f_bs.append(self._boot_estimators[b].predict(x_i))
            # Case where i is in all bootstrap sample sets (Sb)
            # Sanity check, this should have been caught before
            if len(f_bs) == 0:
                raise Exception(
                    f"Training sample {i} is included in all boostrap sets. Increase number of boostrap models."
                )
            f_phi_loo_x = np.mean(f_bs)
            # self.residuals.append(np.abs(y_train.iloc[-i] - f_phi_loo_x))
            self.residuals.append(np.abs(y_train[-i] - f_phi_loo_x))
        # Weighted oob matrix
        self._oob_matrix = (
            self._oob_matrix
            / np.tile(np.sum(self._oob_matrix, axis=1), (self.B, 1)).transpose()
        )

    def predict(self, X_test, alpha=0.1, y_true=None, s=None):
        """Estimate conditionnal mean and interval prediction.

        Args:
            y_true: if not None, residuals update based on seasonality is performed
            s: Number of online samples necessary to update the residuals sequence
        """
        y_pred_upper_list = list()
        y_pred_lower_list = list()
        y_pred_list = list()
        online_residuals = list()
        updated_residuals = deepcopy(self.residuals)
        # Time checkpoint for residuals update
        t_checkpoint = 0

        w = np.quantile(
            updated_residuals, (1 - alpha) * (1 + 1 / len(updated_residuals))
        )
        for t in tqdm(range(X_test.shape[0])):
            f_bs = np.array(
                [self._boot_estimators[b].predict(X_test[[t]]) for b in range(self.B)]
            )
            # Compute estimate quantile
            f_phi_x_loos = np.matmul(self._oob_matrix, f_bs)
            y_pred = np.quantile(
                f_phi_x_loos, (1 - alpha) * (1 + 1 / len(f_phi_x_loos))
            )
            # Compute residual quantile
            y_pred_upper = y_pred + w
            y_pred_lower = y_pred - w
            # Update prediction / PI lists for the current example
            y_pred_upper_list.append(y_pred_upper)
            y_pred_lower_list.append(y_pred_lower)
            y_pred_list.append(y_pred)
            if y_true is not None:
                online_residuals.append(np.abs(y_pred - y_true[t]))
                # Update residuals based on seasonality
                if t > 0 and (t - t_checkpoint) % s == 0:
                    t_checkpoint = t
                    updated_residuals = updated_residuals[s:]  # Remove s first elements
                    # Concatenate with fresh residuals
                    updated_residuals += online_residuals
                    online_residuals = []  # Reset online residuals
                    # Compute new residuals quantile
                    w = np.quantile(
                        updated_residuals,
                        (1 - alpha) * (1 + 1 / len(updated_residuals)),
                    )

        return (
            np.array(y_pred_list),
            np.array(y_pred_lower_list),
            np.array(y_pred_upper_list),
        )


class AdaptiveEnbPI:
    # """Ensemble Batch Prediction Intervals method, Locally Adaptive version

    def __init__(
        self, model, T, B, dispersion_model, aggregation_predictor_version_2=True
    ):
        """constructor

        Args:
            model: regression predictor, with '.fit()' and '.predict()' methods
            T: _description_
            B: _description_
            dispersion_model: regression predictor, with '.fit()' and '.predict()' methods
            aggregation_predictor_version_2: if True, uses version 2 of EnbPI algorithm https://arxiv.org/abs/2010.09107v12
        """
        self.model = model
        self.T = T
        self.B = B

        # estimate absolute deviation for Y|X=x --> ad = |y - y_pred|
        self.dispersion_model = dispersion_model

        # version 1: http://proceedings.mlr.press/v139/xu21h.html
        #           --> y_pred is 1-alpha quantile of bootstrapped predictions
        # version 2: https://arxiv.org/pdf/2010.09107.pdf
        #           --> y_pred is aggregation (mean) of bootstrapped predictions
        self.aggregation_predictor_version_2 = aggregation_predictor_version_2

    def fit(self, X_train, y_train):
        self._oob_dict = dict()  # Key: b. Value: out of bag data
        self._boot_estimators = list()  # f^_b for b in [1,B]
        self._boot_dispersion_estimators = list()  # f^_b for b in [1,B]
        horizon_indices = np.arange(self.T)

        # === (1) === Do bootstrap sampling, check OOB condition ===
        self._boot_dict = dict()

        # WARNING: this loop could written as a recursion. PLEASE DO NOT DO IT.
        for b in range(self.B):
            # ensure we don't have pathological
            # bootstrap sample: identical to original sample
            oob_sample_is_empty = True

            while oob_sample_is_empty:
                boot = resample(horizon_indices, replace=True, n_samples=self.T)
                oob_units = np.setdiff1d(horizon_indices, boot)
                oob_sample_is_empty = not (
                    len(oob_units) > 0
                )  # non-empty if at least 1 elem

                if not oob_sample_is_empty:
                    # if bootstrap sample is exactly as original sample, we
                    # exclude and repeat sampling: we need at least one unit to be OOB.
                    # This is an extremely unlikely case, but not impossible.
                    self._boot_dict[b] = boot
                    self._oob_dict[b] = np.setdiff1d(horizon_indices, boot)

        # == create oob_matrix, one row for every i-th training sample
        #   Bootstrap matrix: lines = time index, columns = oob set
        #   Cell value is 1 if time index is in the oob set
        #
        self._oob_matrix = np.zeros((self.T, self.B))

        for i in tqdm(range(self.T)):
            oobs_for_i_th_unit = [
                1 if i in self._oob_dict[b] else 0 for b in range(self.B)
            ]

            # verify OOB-ness for all i-th training samples, raise if not.
            # This check can only be done after the self.B samples have been drawn
            if np.sum(oobs_for_i_th_unit) == 0:
                raise Exception(
                    f'Training sample {i} is included in all boostrap sets.\
                        Increase number "B=" of boostrap models.'
                )
            else:
                self._oob_matrix[i] = oobs_for_i_th_unit

        # === (2) === Fit predictors on bootstrapped samples
        print(" === step 1/2: fit predictors on boostrapped data")

        for b in tqdm(range(self.B)):
            # retrieve list of indexes of previously bootstrapped sample
            boot = self._boot_dict[b]

            # == (1) fit point predictor
            f_hat_b = deepcopy(self.model)  # Instantiate model

            # WARNING: verify X_train[-boot] indexing. we are missing the 0-th element
            # --> when x[-s], if 0 is in s, then we always get the 0-th, not
            # the last element!
            f_hat_b.fit(X_train[-boot], y_train[-boot])
            self._boot_estimators.append(f_hat_b)  # Store fitted model

            # == (2) Fit dispersion predictor
            ad_hat_b = deepcopy(self.dispersion_model)  # Instantiate model

            # WARNING: verify X_train[-boot] indexing. we are missing the 0-th element
            # --> when x[-s], if 0 is in s, then we always get the 0-th, not
            # the last element!
            ads = np.abs(y_train[-boot] - f_hat_b.predict(X_train[-boot]))
            ad_hat_b.fit(X_train[-boot], ads)

            self._boot_dispersion_estimators.append(ad_hat_b)  # Store fitted model

        # === (3) === Residuals computation
        print(" === step 2/2: compute nonconformity scores")
        self.residuals = list()  # list of residuals

        # TODO: parallelize from here
        # from joblib import Parallel, delayed
        # Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
        #
        for i in tqdm(range(self.T)):
            f_bs = list()
            ad_bs = list()
            x_i = X_train[[-i]]

            for b in range(self.B):
                if self._oob_matrix[i][b] == 1:
                    f_bs.append(self._boot_estimators[b].predict(x_i))
                    ad_bs.append(self._boot_dispersion_estimators[b].predict(x_i))

            f_phi_loo_x = np.mean(f_bs)
            ad_phi_loo_x = np.mean(ad_bs)

            self.residuals.append(np.abs(y_train[-i] - f_phi_loo_x) / ad_phi_loo_x)

        # Weighted oob matrix
        self._oob_matrix = (
            self._oob_matrix
            / np.tile(np.sum(self._oob_matrix, axis=1), (self.B, 1)).transpose()
        )

    def predict(self, X_test, alpha=0.1, y_true=None, s=None):
        """Estimate conditionnal mean and interval prediction.

        Args:
            alpha: miscoverage level, acceptable statistical error
            y_true: if not None, residuals update based on seasonality is performed
            s: Number of online samples necessary to update the residuals sequence
        """
        y_pred_upper_list = list()
        y_pred_lower_list = list()
        y_pred_list = list()
        online_residuals = list()
        updated_residuals = deepcopy(self.residuals)

        # Time checkpoint for residuals update
        t_checkpoint = 0

        w = np.quantile(
            updated_residuals, (1 - alpha) * (1 + 1 / len(updated_residuals))
        )

        for t in tqdm(range(X_test.shape[0])):
            f_bs = np.array(
                [self._boot_estimators[b].predict(X_test[[t]]) for b in range(self.B)]
            )
            ad_bs = np.array(
                [
                    self._boot_dispersion_estimators[b].predict(X_test[[t]])
                    for b in range(self.B)
                ]
            )

            # Compute estimate quantile
            f_phi_x_loos = np.matmul(self._oob_matrix, f_bs)
            ad_phi_x_loos = np.matmul(self._oob_matrix, ad_bs)

            if self.aggregation_predictor_version_2 == False:
                # EnbPI v1: ICML 2021 http://proceedings.mlr.press/v139/xu21h.html
                y_pred = np.quantile(
                    f_phi_x_loos, (1 - alpha) * (1 + 1 / len(f_phi_x_loos))
                )
            else:
                # TODO: implement Lines 13 (modified emp. quantile for conformalization)
                # EnbPI v2: inference pred by aggregation (https://arxiv.org/abs/2010.09107v12)
                y_pred = np.mean(f_phi_x_loos)

            # Compute residual quantile
            y_pred_upper = y_pred + w * np.mean(ad_phi_x_loos)
            y_pred_lower = y_pred - w * np.mean(ad_phi_x_loos)

            # Update prediction / PI lists for the current example
            y_pred_upper_list.append(y_pred_upper)
            y_pred_lower_list.append(y_pred_lower)
            y_pred_list.append(y_pred)

            if y_true is not None:
                # TODO: ([Luca] I think)
                # uses old dispersion predictor to normalize new y_true points
                # ad_bs_test = np.array(
                #     [
                #         self._boot_dispersion_estimators[b].predict(X_test[[t]])
                #         for b in range(self.B)
                #     ]
                # )

                ad_phi_x_loos = np.matmul(self._oob_matrix, ad_bs)
                ad_estimate = np.mean(ad_phi_x_loos)
                online_residuals.append(np.abs(y_pred - y_true[t]) / ad_estimate)

                # Update residuals based on seasonality
                if t > 0 and (t - t_checkpoint) % s == 0:
                    t_checkpoint = t
                    updated_residuals = updated_residuals[s:]  # Remove s first elements

                    # Concatenate with fresh residuals
                    updated_residuals += online_residuals
                    online_residuals = []  # Reset online residuals

                    # Compute new residuals quantile
                    w = np.quantile(
                        updated_residuals,
                        (1 - alpha) * (1 + 1 / len(updated_residuals)),
                    )

        return (
            np.array(y_pred_list),
            np.array(y_pred_lower_list),
            np.array(y_pred_upper_list),
        )

"""
This module provides the canvas for conformal prediction.
"""


from puncc.calibration import Calibrator
from puncc.prediction import BasePredictor
import numpy as np
from typing import Iterable, Callable
from copy import deepcopy
from puncc.utils import agg_func
import matplotlib.pyplot as plt


class CvAggregation:
    """This class enables to aggregate predictions and calibrations
    from different K-folds."""

    def __init__(self, K, agg_func=agg_func) -> None:
        self.K = K  # Number of K-folds
        self.predictors = dict()
        self.calibrators = dict()
        self.agg_func = agg_func

    def append_predictor(self, id, predictor):
        self.predictors[id] = deepcopy(predictor)

    def append_calibrator(self, id, calibrator):
        self.calibrators[id] = deepcopy(calibrator)

    def get_residuals(self):
        """Get a dictionnary of residuals computed on the K-folds
        Returns:
            Dict of residual. Key: K-fold index, value: residuals iterable.
        """
        return {
            k: calibrator._residuals
            for k, calibrator in self.calibrators.items()
        }

    def get_weights(self):
        """Get a dictionnary of normalized weights computed on the K-folds
        Returns:
            Dict of normalized weights. Key: K-fold index,
                                        value: residuals iterable.
        """
        return {
            k: calibrator.weights for k, calibrator in self.calibrators.items()
        }

    def predict(self, X, alpha):
        assert (
            self.predictors.keys() == self.calibrators.keys()
        ), "K-fold predictors are not well calibrated."

        y_preds, y_pred_lowers, y_pred_uppers, sigma_preds = (
            list(),
            list(),
            list(),
            list(),
        )

        for k in self.predictors.keys():
            y_pred, y_pred_lower, y_pred_upper, sigma_pred = self.predictors[
                k
            ].predict(X)

            y_preds.append(y_pred)
            sigma_preds.append(sigma_pred)

            if self.calibrators[k] is not None:
                y_pred_lower, y_pred_upper = self.calibrators[k].calibrate(
                    y_pred=y_pred,
                    alpha=alpha,
                    y_pred_lower=y_pred_lower,
                    y_pred_upper=y_pred_upper,
                    sigma_pred=sigma_pred,
                    X=X,
                )

            y_pred_lowers.append(y_pred_lower)
            y_pred_uppers.append(y_pred_upper)

        if self.K == 1:  # Simple Split
            y_pred = y_preds[0]
            y_pred_lower = y_pred_lowers[0]
            y_pred_upper = y_pred_uppers[0]
            sigma_pred = sigma_preds[0]
        else:  # K-Fold Split
            y_pred = self.agg_func(y_preds)
            y_pred_lower = np.quantile(
                y_pred_lowers, (1 - alpha) * (1 + 1 / self.K), axis=0
            )

            y_pred_upper = np.quantile(
                y_pred_uppers, (1 - alpha) * (1 + 1 / self.K), axis=0
            )
            sigma_pred = self.agg_func(sigma_preds)
        if True in np.isnan(y_pred_lower):
            raise Exception
        return (y_pred, y_pred_lower, y_pred_upper, sigma_pred)


class ConformalPredictor:
    """Conformal predictor class.
    Attributes:
        predictor: point-based or interval-based model wrapper
        calibrator: nonconformity computation strategy and interval predictor
        splitter: fit/calibration split strategy
        train: if False, prediction model(s) will not be (re)trained
    """

    def __init__(
        self,
        calibrator: Calibrator,
        predictor: BasePredictor,
        splitter: Iterable,
        agg_func: Callable = agg_func,
        train: bool = True,
    ):
        self.calibrator = calibrator
        self.predictor = predictor
        self.splitter = splitter
        self.agg_func = agg_func
        self.train = train

    def get_residuals(self):
        return self._cv_aggregation.get_residuals()

    def get_weights(self):
        return self._cv_aggregation.get_weights()

    def fit(self, X, y, **kwargs):

        splits = self.splitter(X, y)

        predictor = deepcopy(self.predictor)
        calibrator = deepcopy(self.calibrator)

        self._cv_aggregation = CvAggregation(
            K=len(splits), agg_func=self.agg_func
        )

        if len(splits) > 1:
            if not self.train:
                raise Exception(
                    "Model already trained. This is inconsistent with the"
                    + "cross-validation strategy."
                )

        for i, (X_fit, y_fit, X_calib, y_calib) in enumerate(splits):
            if self.train:
                predictor.fit(X_fit, y_fit, **kwargs)  # Fit K-fold predictor
            self._cv_aggregation.append_predictor(i, predictor)

            if self.calibrator is not None:
                (
                    y_pred,
                    y_pred_lower,
                    y_pred_upper,
                    sigma_pred,
                ) = predictor.predict(X_calib)
                calibrator.estimate(
                    y_true=y_calib,
                    y_pred=y_pred,
                    y_pred_lower=y_pred_lower,
                    y_pred_upper=y_pred_upper,
                    sigma_pred=sigma_pred,
                    X=X_calib,
                )
            self._cv_aggregation.append_calibrator(i, calibrator)

    def predict(self, X: np.array, alpha, **kwargs):
        return self._cv_aggregation.predict(X, alpha)

    def hist_residuals(self, alpha, xlim=None, delta_space=0.03, **kwargs):
        """Visualize the histogram of residuals. If the miscoverage rate
        'alpha' is given, plot the corresponding residual quantile.
        """
        residuals_dict = self.get_residuals()
        K = len(residuals_dict.keys())
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["ytick.labelsize"] = 12
        plt.rcParams["xtick.labelsize"] = 12
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["legend.fontsize"] = 15
        figsize = kwargs["figsize"] if "figsize" in kwargs.keys() else (15, 6)
        del kwargs["figsize"]  # Remove figsize entry
        if K == 1:
            fig = plt.figure(figsize=figsize)
            residuals = residuals_dict[0]
            max_residual = np.max(residuals)
            plt.hist(residuals, **kwargs)
            min_ylim, max_ylim = plt.ylim()
            if alpha:
                residuals_Q = np.quantile(
                    residuals, (1 - alpha) * (1 + 1 / len(residuals))
                )
                plt.axvline(
                    residuals_Q, color="k", linestyle="dashed", linewidth=2
                )
                plt.text(
                    residuals_Q * 0.98,
                    max_ylim * (-delta_space),
                    r"$\mathbf{\delta^{\alpha}}$",
                )
            plt.xlabel("Residuals")
            if xlim:
                plt.xlim(xlim)
            else:
                plt.xlim([0, max_residual])
            if "density" in kwargs.keys() and kwargs["density"] is True:
                plt.ylabel("Occurence Ratio")
                plt.yticks(np.arange(0, 1.1, step=0.1))
            else:
                plt.ylabel("Occurence")
        else:
            fig, ax = plt.subplots(
                nrows=K // 2 + K % 2, ncols=2, figsize=figsize
            )
            ax = ax.flatten()
            for k in residuals_dict.keys():
                residuals = residuals_dict[k]
                ax[k].hist(residuals, **kwargs)
        plt.grid(True)
        # print(plt.xlim())
        plt.show()

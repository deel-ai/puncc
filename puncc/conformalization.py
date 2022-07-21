#
# This module provides the canvas for conformal prediction.
#


from puncc.calibration import (
    Calibrator,
    CvPlusCalibrator,
    AggregationCalibrator,
)
from puncc.prediction import BasePredictor
import numpy as np
from typing import Iterable, Callable
from copy import deepcopy
from puncc.utils import agg_func
import matplotlib.pyplot as plt


class CvAggregation:
    """This class enables to aggregate predictions and calibrations
    from different K-folds.

    Attributes:
        K: number of folds
        predictors: collection of predictors fitted on the K-folds
        calibrators: collection of calibrators fitted on the K-folds
        agg_func: function called to aggregate the predictions of the K-folds
            estimators. Used only when method is 'aggregation'.
        method: method to handle the ensemble prediction and calibration.
            - 'aggregation': aggregate the k-fold estimators using
                agg_func. The PI bounds are computed as the
                quantiles of the k-fold PI bounds.
            - 'cv+': follow cv+ procedure to construct PIs based on the
                k-fold estimators
    """

    def __init__(
        self,
        K: int,
        agg_func: Callable = agg_func,
        method: str = "aggregation",
    ) -> None:
        self.K = K  # Number of K-folds
        self.predictors = dict()
        self.calibrators = dict()
        self.agg_func = agg_func

        if method not in ("aggregation", "cv+"):
            return NotImplemented(
                f"Method {method} is not implemented. "
                + "Please choose between 'aggregation' and 'cv+'."
            )
        self.method = method

    def append_predictor(self, id, predictor):
        self.predictors[id] = deepcopy(predictor)

    def append_calibrator(self, id, calibrator):
        self.calibrators[id] = deepcopy(calibrator)

    def get_residuals(self):
        """Get a dictionnary of residuals computed on the K-folds
        Returns:
            Dict of residual. Key: K-fold index, value: residuals iterable.
        """
        return {k: calibrator._residuals for k, calibrator in self.calibrators.items()}

    def get_weights(self):
        """Get a dictionnary of normalized weights computed on the K-folds
        Returns:
            Dict of normalized weights. Key: K-fold index,
                                        value: residuals iterable.
        """
        return {k: calibrator.weights for k, calibrator in self.calibrators.items()}

    def predict(self, X, alpha):
        assert (
            self.predictors.keys() == self.calibrators.keys()
        ), "K-fold predictors are not well calibrated."

        if self.method == "cv+":
            cvp_calibrator = CvPlusCalibrator(self.calibrators)
            y_pred_lower, y_pred_upper = cvp_calibrator.calibrate(
                self.predictors, alpha=alpha, X=X
            )
            return (None, y_pred_lower, y_pred_upper, None)

        elif self.method == "aggregation":
            agg_calibrator = AggregationCalibrator(self.calibrators, self.agg_func)
            return agg_calibrator.calibrate(self.predictors, alpha=alpha, X=X)

        else:
            return RuntimeError(
                f"Method {self.method} is not implemented"
                + "Please choose between 'aggregation' and 'cv+'."
            )


class ConformalPredictor:
    """Conformal predictor class.
    Attributes:
        predictor: point-based or interval-based model wrapper
        calibrator: nonconformity computation strategy and interval predictor
        splitter: fit/calibration split strategy
        train: if False, prediction model(s) will not be (re)trained
        agg_func: In case the splitter is a K-fold-like strategy, agg_func is
            called to aggregate the predictions of the K-folds
            estimators. Used only when method is 'aggregation'.
        method: method to handle the ensemble prediction and calibration
            in case the splitter is a K-fold-like strategy.
                - 'aggregation': aggregate the k-fold estimators using
                    agg_func. The PI bounds are computed as the
                    quantiles of the k-fold PI bounds.
                - 'cv+': follow cv+ procedure to construct PIs based on the
                    k-fold estimators
    """

    def __init__(
        self,
        calibrator: Calibrator,
        predictor: BasePredictor,
        splitter,  # <-- [TODO] what type do we expect here? Not [Iterable]
        agg_func: Callable = agg_func,
        method: str = "aggregation",
        train: bool = True,
    ):
        self.calibrator = calibrator
        self.predictor = predictor
        self.splitter = splitter
        self.agg_func = agg_func
        self.method = method
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
            K=len(splits), agg_func=self.agg_func, method=self.method
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

                # TODO: [sigma_pred] not in calibrator.estimate() definition.
                # Fix estimate() definition to accept sigma_pred or change the function itself?
                # Maybe this stuff hangs around from previous versions?
                calibrator.estimate(
                    y_true=y_calib,
                    y_pred=y_pred,
                    y_pred_lower=y_pred_lower,
                    y_pred_upper=y_pred_upper,
                    sigma_pred=sigma_pred,
                    X=X_calib,
                )

            self._cv_aggregation.append_calibrator(i, calibrator)

    def predict(self, X: np.ndarray, alpha, **kwargs):
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
                    residuals,
                    (1 - alpha) * (1 + 1 / len(residuals)),
                    method="higher",
                )
                plt.axvline(residuals_Q, color="k", linestyle="dashed", linewidth=2)
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
            fig, ax = plt.subplots(nrows=K // 2 + K % 2, ncols=2, figsize=figsize)
            ax = ax.flatten()
            for k in residuals_dict.keys():
                residuals = residuals_dict[k]
                ax[k].hist(residuals, **kwargs)
        plt.grid(True)
        # print(plt.xlim())
        plt.show()

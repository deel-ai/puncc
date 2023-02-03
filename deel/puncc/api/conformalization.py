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
This module provides the canvas for conformal prediction.
"""
from copy import deepcopy
from typing import Iterable

import numpy as np

from deel.puncc.api.calibration import BaseCalibrator
from deel.puncc.api.calibration import CvPlusCalibrator
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.api.splitting import BaseSplitter


class ConformalPredictor:
    """Conformal predictor class.

    :param deel.puncc.api.prediction.BasePredictor predictor: model wrapper.
    :param deel.puncc.api.prediction.BaseCalibrator calibrator: nonconformity computation strategy and interval predictor.
    :param deel.puncc.api.prediction.BaseSplitter splitter: fit/calibration split strategy.
    :param str method: method to handle the ensemble prediction and calibration in case the splitter is a K-fold-like strategy. Defaults to 'cv+' to follow cv+ procedure.
    :param bool train: if False, prediction model(s) will not be (re)trained. Defaults to True.

    .. WARNING::
        if a K-Fold-like splitter is provided with the :data:`train` attribute set to True, an exception is raised.
        The models have to be trained during the call :meth:`fit`.

    """

    def __init__(
        self,
        calibrator: BaseCalibrator,
        predictor: BasePredictor,
        splitter: BaseSplitter,
        method: str = "cv+",
        train: bool = True,
    ):
        self.calibrator = calibrator
        self.predictor = predictor
        self.splitter = splitter
        self.method = method
        self.train = train
        self._cv_cp_agg = None

    def get_residuals(self):
        """Getter for computed nonconformity scores on the calibration(s) set(s).

        :returns: dictionary of nonconformity scores indexed by the fold index.
        :rtype: dict

        :raises RuntimeError: :meth:`fit` needs to be called before :meth:`get_residuals`.
        """

        if self._cv_cp_agg is None:
            return RuntimeError("Error: call 'fit' method first.")

        return self._cv_cp_agg.get_residuals()

    def get_weights(self):
        """Getter for weights associated to calibration samples.

        :returns: dictionary of weights indexed by the fold index.
        :rtype: dict

        :raises RuntimeError: :meth:`fit` needs to be called before :meth:`get_weights`.
        """

        if self._cv_cp_agg is None:
            return RuntimeError("Error: call 'fit' method first.")

        return self._cv_cp_agg.get_weights()

    def fit(
        self,
        X: Iterable,
        y: Iterable,
        **kwargs,
    ) -> None:
        """Fit the model(s) and estimate the nonconformity scores.

        If the splitter is an instance of :class:`deel.puncc.splitting.KFoldSplitter`, the fit operates on each fold separately. Thereafter, the predictions and nonconformity scores are combined accordingly to an aggregation method (cv+ by default).

        :param ndarray|DataFrame|Tensor X: features.
        :param ndarray|DataFrame|Tensor y: labels.
        :param dict kwargs: options configuration for the training.

        :raises RuntimeError: inconsistencies between the train status of the model(s) and the :data:`train` class attribute.
        """
        # Get split folds. Each fold split is a iterable of a quadruple that
        # contains fit and calibration data.
        splits = self.splitter(X, y)

        # The Cross validation aggregator will aggregate the predictors and
        # calibrators fitted on each of the K splits.
        self._cv_cp_agg = CrossValCpAggregator(K=len(splits), method=self.method)

        # In case of multiple split folds, the predictor require training.
        # Having 'self.train' set to False is therefore an inconsistency
        if len(splits) > 1 and not self.train:
            raise RuntimeError(
                "Model already trained. This is inconsistent with the"
                + "cross-validation strategy."
            )

        # Core loop: for each split (that contains fit and calib data):
        #   1- The predictor f_i is fitted of (X_fit, y_fit) (if necessary)
        #   2- y_pred is predicted by f_i
        #   3- The calibrator is fitted to approximate the distribution of nonconformity scores
        for i, (X_fit, y_fit, X_calib, y_calib) in enumerate(splits):
            # Make local copies of the structure of the predictor and the calibrator.
            # In case of a K-fold like splitting strategy, these structures are
            # inherited by the predictor/calibrator used in each fold.
            predictor = self.predictor.copy()
            calibrator = deepcopy(self.calibrator)

            if self.train:
                predictor.fit(X_fit, y_fit, **kwargs)  # Fit K-fold predictor
            # Make sure that predictor is already trained if train arg is False
            elif self.train is False and predictor.is_trained is False:
                raise RuntimeError(
                    f"'train' argument is set to 'False' but model is not pre-trained"
                )

            # Call predictor to estimate predictions
            y_pred = predictor.predict(X_calib)
            # Fit calibrator
            calibrator.fit(y_true=y_calib, y_pred=y_pred)

            # Compute normalized weights of the nonconformity scores
            # if a weight function is provided
            if calibrator.weight_func:
                weights = calibrator.weight_func(X_calib)
                norm_weights = calibrator.barber_weights(weights=weights)
                # Store the mornalized weights
                calibrator.set_norm_weights(norm_weights)
            # Add predictor and calibrator to the collection that is used later
            # by the predict method

            self._cv_cp_agg.append_predictor(i, predictor)
            self._cv_cp_agg.append_calibrator(i, calibrator)

    def predict(self, X: Iterable, alpha: float) -> Iterable:
        """Predict point, interval and variability estimates for X data.

        :param ndarray|DataFrame|Tensor X: features.
        :param float alpha: significance level (max miscoverage target).

        :returns: y_pred.
        :rtype: ndarray|DataFrame|Tensor
        """

        if self._cv_cp_agg is None:
            raise RuntimeError("Error: call 'fit' method first.")

        return self._cv_cp_agg.predict(X, alpha)


class CrossValCpAggregator:
    """This class enables to aggregate predictions and calibrations
    from different K-folds.


    :param int K: number of folds
    :param dict _predictors: collection of predictors fitted on the K-folds
    :param dict _calibrators: collection of calibrators fitted on the K-folds
    :param str method: method to handle the ensemble prediction and calibration, defaults to 'cv+'.
    """

    def __init__(
        self,
        K: int,
        method: str = "cv+",
    ):
        self.K = K  # Number of K-folds
        self._predictors = dict()
        self._calibrators = dict()

        if method not in ("cv+"):
            return NotImplemented(
                f"Method {method} is not implemented. " + "Please choose 'cv+'."
            )

        self.method = method

    def append_predictor(self, id, predictor):
        self._predictors[id] = predictor.copy()

    def append_calibrator(self, id, calibrator):
        self._calibrators[id] = deepcopy(calibrator)

    def get_residuals(self):
        """Get a dictionnary of residuals computed on the K-folds.

        :returns: dictionary of residual indexed by the K-fold number.
        :rtype: dict
        """
        return {k: calibrator._residuals for k, calibrator in self._calibrators.items()}

    def get_weights(self):
        """Get a dictionnary of normalized weights computed on the K-folds.

        :returns: dictionary of normalized weights indexed by the K-fold number.
        :rtype: dict
        """
        return {
            k: calibrator.get_weights() for k, calibrator in self._calibrators.items()
        }

    def predict(self, X: Iterable, alpha: float) -> Iterable:  #  type: ignore
        """Predict point, interval and variability estimates for X data.

        :param ndarray|DataFrame|Tensor X: features.
        :param float alpha: significance level (max miscoverage target).

        :returns: y_pred, y_lower, y_higher.
        :rtype: Tuple[ndarray|DataFrame|Tensor]
        """
        assert (
            self._predictors.keys() == self._calibrators.keys()
        ), "K-fold predictors are not well calibrated."

        K = len(self._predictors.keys())  # Number of folds

        # No cross-val strategy if K = 1
        if K == 1:

            for k in self._predictors.keys():
                predictor = self._predictors[k]
                calibrator = self._calibrators[k]
                # Get normalized weights of the nonconformity scores
                norm_weights = calibrator.get_norm_weights()
                y_pred = predictor.predict(X=X)
                set_pred = calibrator.calibrate(
                    alpha=alpha, y_pred=y_pred, weights=norm_weights
                )
                return (y_pred, *set_pred)

        else:
            y_pred = None

            if self.method == "cv+":
                cvp_calibrator = CvPlusCalibrator(self._calibrators)
                set_pred = cvp_calibrator.calibrate(
                    X=X,
                    kfold_predictors_dict=self._predictors,
                    alpha=alpha,
                )
                return (y_pred, *set_pred)  # type: ignore

            else:
                raise RuntimeError(
                    f"Method {self.method} is not implemented." + "Please choose 'cv+'."
                )

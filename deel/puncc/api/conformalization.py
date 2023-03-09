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
import logging
from copy import deepcopy
from typing import Iterable
from typing import Tuple

import numpy as np

from deel.puncc.api.calibration import BaseCalibrator
from deel.puncc.api.calibration import CvPlusCalibrator
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.api.splitting import BaseSplitter

logger = logging.getLogger(__name__)


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


    Conformal Regression example:
    =============================

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

    def get_residuals(self) -> dict:
        """Getter for computed nonconformity scores on the calibration(s) set(s).

        :returns: dictionary of nonconformity scores indexed by the fold index.
        :rtype: dict

        :raises RuntimeError: :meth:`fit` needs to be called before :meth:`get_residuals`.
        """

        if self._cv_cp_agg is None:
            return RuntimeError("Error: call 'fit' method first.")

        return self._cv_cp_agg.get_residuals()

    def get_weights(self) -> dict:
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

        :param Iterable X: features.
        :param Iterable y: labels.
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
                logger.info(f"Fitting model on fold {i}")
                predictor.fit(X_fit, y_fit, **kwargs)  # Fit K-fold predictor

            # Make sure that predictor is already trained if train arg is False
            elif self.train is False and predictor.is_trained is False:
                raise RuntimeError(
                    f"'train' argument is set to 'False' but model is not pre-trained"
                )

            # Call predictor to estimate predictions
            logger.info(f"Model predictions on X_calib fold {i}")
            y_pred = predictor.predict(X_calib)
            logger.debug(f"Shape of y_pred")

            # Fit calibrator
            logger.info(f"Fitting calibrator on fold {i}")
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
            logger.info(
                f"Adding {i}-th K-fold predictor and calibrator to the"
                + "cross-validation aggregator"
            )
            self._cv_cp_agg.append_predictor(i, predictor)
            self._cv_cp_agg.append_calibrator(i, calibrator)

    def predict(self, X: Iterable, alpha: float) -> Tuple[np.ndarray]:
        """Predict point, and interval estimates for X data.

        :param Iterable X: features.
        :param float alpha: significance level (max miscoverage target).

        :returns: y_pred, y_lower, y_higher.
        :rtype: Tuple[ndarray]
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

    def get_residuals(self) -> dict:
        """Get a dictionnary of residuals computed on the K-folds.

        :returns: dictionary of residual indexed by the K-fold number.
        :rtype: dict
        """
        return {k: calibrator._residuals for k, calibrator in self._calibrators.items()}

    def get_weights(self) -> dict:
        """Get a dictionnary of normalized weights computed on the K-folds.

        :returns: dictionary of normalized weights indexed by the K-fold number.
        :rtype: dict
        """
        return {
            k: calibrator.get_weights() for k, calibrator in self._calibrators.items()
        }

    def predict(self, X: Iterable, alpha: float) -> Tuple[np.ndarray]:  #  type: ignore
        """Predict point, interval and variability estimates for X data.

        :param Iterable X: features.
        :param float alpha: significance level (max miscoverage target).

        :returns: y_pred, y_lower, y_higher.
        :rtype: Tuple[Iterable]
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
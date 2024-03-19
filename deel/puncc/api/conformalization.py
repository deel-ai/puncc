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
import pickle
from copy import deepcopy
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from deel.puncc.api.calibration import BaseCalibrator
from deel.puncc.api.calibration import CvPlusCalibrator
from deel.puncc.api.corrections import bonferroni
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.api.prediction import DualPredictor
from deel.puncc.api.splitting import BaseSplitter
from deel.puncc.api.splitting import IdSplitter

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """Conformal predictor class.

    :param deel.puncc.api.prediction.BasePredictor | object predictor:
        underlying model to be conformalized. The model can directly be
        passed as argument if it already has `fit` and `predict` methods.
    :param deel.puncc.api.prediction.BaseCalibrator calibrator: nonconformity
        computation strategy and set predictor.
    :param deel.puncc.api.prediction.BaseSplitter splitter: fit/calibration
        split strategy. The splitter can be set to None if the underlying
        model is pretrained.
    :param str method: method to handle the ensemble prediction and calibration
        in case the splitter is a K-fold-like strategy. Defaults to 'cv+' to
        follow cv+ procedure.
    :param bool train: if False, prediction model(s) will not be (re)trained.
        Defaults to True.

    .. WARNING::
        if a K-Fold-like splitter is provided with the :data:`train` attribute
        set to True, an exception is raised.
        The models have to be trained during the call :meth:`fit`.


    **Conformal Regression example:**

    .. code-block:: python

        from sklearn import linear_model
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        from deel.puncc.api.conformalization import ConformalPredictor
        from deel.puncc.api.prediction import BasePredictor
        from deel.puncc.api.calibration import BaseCalibrator
        from deel.puncc.api.splitting import KFoldSplitter
        from deel.puncc.api import nonconformity_scores
        from deel.puncc.api import prediction_sets

        # Generate a random regression problem
        X, y = make_regression(n_samples=1000, n_features=4, n_informative=2,
                                random_state=0, shuffle=False)

        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=0
        )

        # Regression linear model
        model = linear_model.LinearRegression()

        # Definition of a predictor. Note that it is not required to wrap
        # the model here because it already implements fit and predict methods
        predictor = BasePredictor(model)

        # Definition of a calibrator, built for a given nonconformity scores
        # and a procedure to build the prediction sets
        calibrator = BaseCalibrator(nonconf_score_func=nonconformity_scores.mad,
                                pred_set_func=prediction_sets.constant_interval)

        # Definition of a K-fold splitter that produces
        # 20 folds of fit/calibration
        kfold_splitter = KFoldSplitter(K=20, random_state=42)

        # Conformal predictor requires the three components instantiated
        # previously. Our choice of calibrator and splitter yields a cv+ procedure
        conformal_predictor = ConformalPredictor(predictor=predictor,
                                                calibrator=calibrator,
                                                splitter=kfold_splitter,
                                                train=True)

        # Fit model and compute nonconformity scores
        conformal_predictor.fit(X_train, y_train)

        # The lower and upper bounds of the prediction interval are predicted
        # by the call to predict on the new data w.r.t a risk level of 10%.
        # Besides, there is no aggregate point prediction in cv+ so y_pred is None.
        y_pred , y_lower, y_upper = conformal_predictor.predict(X_test, alpha=.1)

    """

    def __init__(
        self,
        calibrator: BaseCalibrator,
        predictor: Union[BasePredictor, object],
        splitter: BaseSplitter,
        method: str = "cv+",
        train: bool = True,
    ):
        self.calibrator = calibrator

        if isinstance(predictor, (BasePredictor, DualPredictor)):
            self.predictor = predictor

        elif not hasattr(predictor, "predict"):
            raise RuntimeError(
                "Provided model has no predict method. "
                + "Use a BasePredictor or a DualPredictor to build "
                + "a compatible predictor."
            )

        elif train and not hasattr(predictor, "fit"):
            raise RuntimeError(
                "Provided model is not trained and has no fit method. "
                + "Use a BasePredictor or a DualPredictor to build "
                + "a compatible predictor."
            )

        else:
            self.predictor = BasePredictor(predictor, is_trained=not train)

        if train and splitter is None:
            raise RuntimeError(
                "The splitter argument is None but train is set to True. "
                + "Please provide a correct splitter to train the underlying "
                + "model."
            )

        if method != "cv+":
            raise RuntimeError(
                f"Method {method} is not implemented." + "Please choose 'cv+'."
            )

        self.splitter = splitter
        self.method = method
        self.train = train
        self._cv_cp_agg = None

    def get_nonconformity_scores(self) -> dict:
        """Getter for computed nonconformity scores on the calibration(s) set(s).

        :returns: dictionary of nonconformity scores indexed by the fold index.
        :rtype: dict

        :raises RuntimeError: :meth:`fit` needs to be called before
            :meth:`get_nonconformity_scores`.
        """

        if self._cv_cp_agg is None:
            raise RuntimeError("Error: call 'fit' method first.")

        return self._cv_cp_agg.get_nonconformity_scores()

    def get_weights(self) -> dict:
        """Getter for weights associated to calibration samples.

        :returns: dictionary of weights indexed by the fold index.
        :rtype: dict

        :raises RuntimeError: :meth:`fit` needs to be called before
            :meth:`get_weights`.
        """

        if self._cv_cp_agg is None:
            raise RuntimeError("Error: call 'fit' method first.")

        return self._cv_cp_agg.get_weights()

    def fit(
        self,
        X: Iterable,
        y: Iterable,
        use_cached: bool = False,
        **kwargs,
    ) -> None:
        """Fit the model(s) and estimate the nonconformity scores.

        If the splitter is an instance of
        :class:`deel.puncc.splitting.KFoldSplitter`, the fit operates on each
        fold separately. Thereafter, the predictions and nonconformity scores
        are combined accordingly to an aggregation method (cv+ by default).

        :param Iterable X: features.
        :param Iterable y: labels.
        :param bool use_cached: if set, enables to add the previously computed
            nonconformity scores (if any) to the pool estimated in the current
            call to `fit`. The aggregation follows the CV+
            procedure.
        :param dict kwargs: options configuration for the training.

        :raises RuntimeError: inconsistencies between the train status of the
            model(s) and the :data:`train` class attribute.
        """
        # Get split folds. Each fold split is a iterable of a quadruple that
        # contains fit and calibration data.
        if self.splitter is None:
            splits = IdSplitter(X, y, X, y)()
        else:
            splits = self.splitter(X, y)

        # The Cross validation aggregator will aggregate the predictors and
        # calibrators fitted on each of the K splits.
        if self._cv_cp_agg is None or use_cached is False:
            cached_len = 0
            self._cv_cp_agg = CrossValCpAggregator(
                K=len(splits), method=self.method
            )
        else:
            cached_len = self._cv_cp_agg.K
            self._cv_cp_agg.K = cached_len + len(splits)

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
        #   3- The calibrator is fitted to approximate the distribution of
        #      nonconformity scores
        for i, (X_fit, y_fit, X_calib, y_calib) in enumerate(splits):
            # Make local copies of the structure of the predictor and the calibrator.
            # In case of a K-fold like splitting strategy, these structures are
            # inherited by the predictor/calibrator used in each fold.
            if len(splits) > 1:
                predictor = self.predictor.copy()
                calibrator = deepcopy(self.calibrator)
            else:
                predictor = self.predictor
                calibrator = self.calibrator

            if self.train:
                if self.splitter is None:
                    raise RuntimeError(
                        "The splitter argument is None but train is set to "
                        + "True. Please provide a correct splitter to train "
                        + "the underlying model."
                    )
                logger.info(f"Fitting model on fold {i+cached_len}")
                predictor.fit(X_fit, y_fit, **kwargs)  # Fit K-fold predictor

            # Make sure that predictor is already trained if train arg is False
            elif self.train is False and predictor.get_is_trained() is False:
                raise RuntimeError(
                    "'train' argument is set to 'False' but model(s) not pre-trained."
                )

            else:  # Skipping training
                logger.info("Skipping training.")

            # Call predictor to estimate predictions
            logger.info(f"Model predictions on X_calib fold {i+cached_len}")
            y_pred = predictor.predict(X_calib)
            logger.debug("Shape of y_pred")

            # Fit calibrator
            logger.info(f"Fitting calibrator on fold {i+cached_len}")
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
            self._cv_cp_agg.append_predictor(i + cached_len, predictor)
            self._cv_cp_agg.append_calibrator(i + cached_len, calibrator)

    def predict(
        self,
        X: Iterable,
        alpha: float,
        correction_func: Optional[Callable] = bonferroni,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Predict point, and interval estimates for X data.

        :param Iterable X: features.
        :param float alpha: significance level (max miscoverage target).
        :param Callable correction_func: correction for multiple hypothesis
            testing in the case of multivariate regression. Defaults to
            Bonferroni correction.

        :returns: (y_pred, y_lower, y_higher) or (y_pred, pred_set).
        :rtype: Union[Tuple[np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray]]
        """
        if self._cv_cp_agg is None:
            raise RuntimeError("Error: call 'fit' method first.")

        return self._cv_cp_agg.predict(X, alpha, correction_func)

    def save(self, path):
        """Serialize current conformal predictor and write it into file.

        :param str path: file path.
        """
        with open(path, "wb") as output_file:
            pickle.dump(self.__dict__, output_file)

    @staticmethod
    def load(path):
        """Load conformal predictor from a file.

        :param str path: file path.

        :returns: loaded conformal predictor instance.
        :rtype: ConformalPredictor
        """
        with open(path, "rb") as input_file:
            saved_dict = pickle.load(input_file)

        loaded_cp = ConformalPredictor(
            calibrator=None, predictor=BasePredictor(None), splitter=object()
        )
        loaded_cp.__dict__.update(saved_dict)
        return loaded_cp


class CrossValCpAggregator:
    """This class enables to aggregate predictions and calibrations
    from different K-folds.


    :param int K: number of folds
    :param dict _predictors: collection of predictors fitted on the K-folds
    :param dict _calibrators: collection of calibrators fitted on the K-folds
    :param str method: method to handle the ensemble prediction and
        calibration, defaults to 'cv+'.
    """

    def __init__(
        self,
        K: int,
        method: str = "cv+",
    ):
        self.K = K  # Number of K-folds
        self._predictors = {}
        self._calibrators = {}

        if method not in ("cv+"):
            raise NotImplementedError(
                f"Method {method} is not implemented. " + "Please choose 'cv+'."
            )

        self.method = method

    def append_predictor(self, key, predictor):
        """Add predictor in kfold predictors dictionnary.

        :param int key: key of the predictor.
        :param BasePredictor|DualPredictor predictor: predictor to be appended.

        """
        self._predictors[key] = predictor.copy()

    def append_calibrator(self, key, calibrator):
        """Add calibrator in kfold calibrators dictionnary.

        :param int key: key of the calibrator.
        :param BaseCalibrator predictor: calibrator to be appended.

        """
        self._calibrators[key] = deepcopy(calibrator)

    def get_nonconformity_scores(self) -> dict:
        """Get a dictionnary of residuals computed on the K-folds.

        :returns: dictionary of residual indexed by the K-fold number.
        :rtype: dict
        """
        return {
            k: calibrator.get_nonconformity_scores()
            for k, calibrator in self._calibrators.items()
        }

    def get_weights(self) -> dict:
        """Get a dictionnary of normalized weights computed on the K-folds.

        :returns: dictionary of normalized weights indexed by the K-fold number.
        :rtype: dict
        """
        return {
            k: calibrator.get_weights()
            for k, calibrator in self._calibrators.items()
        }

    def predict(
        self,
        X: Iterable,
        alpha: float,
        correction_func: Optional[Callable] = bonferroni,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:  #  type: ignore
        """Predict point, interval and variability estimates for X data.

        :param Iterable X: features.
        :param float alpha: significance level (max miscoverage target).
        :param Callable correction_func: correction for multiple hypothesis
            testing in the case of multivariate regression. Defaults to
            Bonferroni correction.

        :returns: y_pred, y_lower, y_higher.
        :rtype: Union[Tuple[np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray]]
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
                    alpha=alpha,
                    y_pred=y_pred,
                    weights=norm_weights,
                    correction=correction_func,
                )
                return (y_pred, *set_pred)

        else:  # TODO support multireg with CV
            y_pred = None

            if self.method == "cv+":
                cvp_calibrator = CvPlusCalibrator(self._calibrators)
                set_pred = cvp_calibrator.calibrate(
                    X=X,
                    kfold_predictors_dict=self._predictors,
                    alpha=alpha,
                )
                return (y_pred, *set_pred)

            raise RuntimeError(
                f"Method {self.method} is not implemented."
                + "Please choose 'cv+'."
            )

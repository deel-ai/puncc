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
from typing import Iterable
from typing import Optional
from typing import Tuple

import numpy as np
from sklearn.utils import resample
from tqdm import tqdm

from deel.puncc.api.calibration2 import BaseCalibrator
from deel.puncc.api.calibration2 import NonConformityScores
from deel.puncc.api.calibration2 import PredictionSets
from deel.puncc.api.conformalization2 import ConformalPredictor
from deel.puncc.api.prediction2 import BasePredictor
from deel.puncc.api.splitting2 import IdSplitter
from deel.puncc.api.splitting2 import KFoldSplitter


class SplitCP:
    """Split conformal prediction method.

    :param BasePredictor predictor: a predictor implementing fit and predict.
    :param bool train: if False, prediction model(s) will not be trained and will be used as is. Defaults to True.
    :param callable weight_func: function that takes as argument an array of
    features X and returns associated "conformality" weights, defaults to None.

    """

    def __init__(self, predictor, train=True, *, weight_func=None):
        self.predictor = predictor
        self.calibrator = BaseCalibrator(
            nonconf_score_func=NonConformityScores.Regression.MAD,
            pred_set_func=PredictionSets.Regression.CONSTANT_INTERVAL,
            weight_func=weight_func,
        )
        self.train = train

    def fit(
        self,
        X_fit: Iterable,
        y_fit: Iterable,
        X_calib: Iterable,
        y_calib: Iterable,
        **kwargs: Optional[dict],
    ):
        """This method fits the models to the fit data (X_fit, y_fit)
        and computes residuals on (X_calib, y_calib).

        :param ndarray|DataFrame|Tensor X_fit: features from the fit dataset.
        :param ndarray|DataFrame|Tensor y_fit: labels from the fit dataset.
        :param ndarray|DataFrame|Tensor X_calib: features from the calibration dataset.
        :param ndarray|DataFrame|Tensor y_calib: labels from the calibration dataset.
        :param dict kwargs: predict configuration to be passed to the model's predict method.
        """
        self.conformal_predictor = ConformalPredictor(
            predictor=self.predictor,
            calibrator=self.calibrator,
            splitter=IdSplitter(X_fit, y_fit, X_calib, y_calib),
        )
        self.conformal_predictor.fit(X=None, y=None, **kwargs)  # type: ignore

    def predict(self, X_test: Iterable, alpha) -> Tuple[Iterable, Iterable, Iterable]:
        """Conformal interval predictions (w.r.t target miscoverage alpha)
        for new samples.

        :param ndarray|DataFrame|Tensor X_test: features of new samples.
        :param float alpha: target maximum miscoverage.

        :returns: y_pred, y_lower, y_higher
        :rtype: Tuple[ndarray|DataFrame|Tensor]
        """

        if not hasattr(self, "conformal_predictor"):
            raise RuntimeError("Fit method should be called before predict.")

        (
            y_pred,
            y_lo,
            y_hi,
        ) = self.conformal_predictor.predict(X_test, alpha=alpha)

        return y_pred, y_lo, y_hi


class LocallyAdaptiveCP(SplitCP):
    """Locally adaptive conformal prediction method.

    :param BasePredictor predictor: a predictor implementing fit and predict.
    Must embed two models for point and dispersion estimations respectively.
    :param bool train: if False, prediction model(s) will not be trained and
    will be used as is. Defaults to True.
    :param callable weight_func: function that takes as argument an array of
    features X and returns associated "conformality" weights, defaults to None.
    """

    def __init__(self, predictor, train=True, *, weight_func=None):
        self.predictor = predictor
        self.calibrator = BaseCalibrator(
            nonconf_score_func=NonConformityScores.Regression.SCALED_MAD,
            pred_set_func=PredictionSets.Regression.SCALED_INTERVAL,
            weight_func=weight_func,
        )
        self.train = train


class CQR(SplitCP):
    """Conformalized quantile regression method.

    :param QuantilePredictor predictor: a predictor implementing fit and predict.
    Must embed two models for lower and upper quantiles estimations respectively.
    :param bool train: if False, prediction model(s) will not be trained and
    will be used as is. Defaults to True.
    :param callable weight_func: function that takes as argument an array of
    features X and returns associated "conformality" weights, defaults to None.
    """

    def __init__(self, predictor, train=True, *, weight_func=None):
        self.predictor = predictor
        self.calibrator = BaseCalibrator(
            nonconf_score_func=NonConformityScores.Regression.CQR_SCORE,
            pred_set_func=PredictionSets.Regression.CQR_INTERVAL,
            weight_func=weight_func,
        )
        self.train = train


class CvPlus:
    """Cross-validation plus method.

    :param BasePredictor predictor: a predictor implementing fit and predict.
    :param int K: number of training/calibration folds.
    :param int random_state: seed to control random folds.
    """

    def __init__(self, predictor, *, K: int, random_state=None):

        self.predictor = predictor
        self.calibrator = BaseCalibrator(
            nonconf_score_func=NonConformityScores.Regression.MAD,
            pred_set_func=PredictionSets.Regression.CONSTANT_INTERVAL,
            weight_func=None,
        )
        self.splitter = KFoldSplitter(K=K, random_state=random_state)

    def fit(
        self,
        X_train: Iterable,
        y_train: Iterable,
        **kwargs: Optional[dict],
    ):
        """This method fits the ensemble models based on the K-fold scheme.
        The out-of-bag folds are used to computes residuals on (X_calib, y_calib).

        :param ndarray|DataFrame|Tensor X_train: features from the train dataset.
        :param ndarray|DataFrame|Tensor y_train: labels from the train dataset.
        :param dict kwargs: predict configuration to be passed to the model's predict method.
        """
        self.conformal_predictor = ConformalPredictor(
            predictor=self.predictor,
            calibrator=self.calibrator,
            splitter=self.splitter,
            method="cv+",
        )
        self.conformal_predictor.fit(X=X_train, y=y_train, **kwargs)

    def predict(self, X_test: Iterable, alpha) -> Tuple[Iterable, Iterable, Iterable]:
        """Conformal interval predictions (w.r.t target miscoverage alpha)
        for new samples.

        :param ndarray|DataFrame|Tensor X_test: features of new samples.
        :param float alpha: target maximum miscoverage.

        :returns: y_pred, y_lower, y_higher
        :rtype: Tuple[ndarray|DataFrame|Tensor]
        """

        if not hasattr(self, "conformal_predictor"):
            raise RuntimeError("Fit method should be called before predict.")

        (
            y_pred,
            y_lo,
            y_hi,
        ) = self.conformal_predictor.predict(X_test, alpha=alpha)

        return y_pred, y_lo, y_hi

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
import numpy as np
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from deel.puncc.api import nonconformity_scores
from deel.puncc.api import prediction_sets
from deel.puncc.api.calibration import BaseCalibrator
from deel.puncc.api.conformalization import ConformalPredictor
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.api.splitting import KFoldSplitter


def test_conformalpredictor():
    # Generate a random regression problem
    X, y = make_regression(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        random_state=0,
        shuffle=False,
    )

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Regression linear model
    model = linear_model.LinearRegression()

    # Definition of a predictor
    predictor = BasePredictor(model)

    # Definition of a calibrator, built for a given nonconformity scores
    # and a procedure to build the prediction sets
    calibrator = BaseCalibrator(
        nonconf_score_func=nonconformity_scores.mad,
        pred_set_func=prediction_sets.constant_interval,
    )

    # Definition of a K-fold splitter that produces
    # 20 folds of fit/calibration
    kfold_splitter = KFoldSplitter(K=20, random_state=42)

    # Conformal predictor requires the three components instantiated
    # previously. Our choice of calibrator and splitter yields a cv+ procedure
    conformal_predictor = ConformalPredictor(
        predictor=predictor,
        calibrator=calibrator,
        splitter=kfold_splitter,
        train=True,
    )

    # Fit model and compute nonconformity scores
    conformal_predictor.fit(X_train, y_train)

    # The lower and upper bounds of the prediction interval are predicted
    # by the call to predict on the new data w.r.t a risk level of 10%.
    # Besides, there is no aggregate point prediction in cv+ so y_pred is None.
    y_pred, y_lower, y_upper = conformal_predictor.predict(X_test, alpha=0.1)

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
import os
import unittest

import numpy as np
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from deel.puncc.api import nonconformity_scores
from deel.puncc.api import prediction_sets
from deel.puncc.api.calibration import BaseCalibrator
from deel.puncc.api.conformalization import ConformalPredictor
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.api.prediction import DualPredictor
from deel.puncc.api.splitting import KFoldSplitter
from deel.puncc.api.splitting import RandomSplitter

np.random.seed(0)


class ConformalPredictorCheck(unittest.TestCase):
    def setUp(self):
        # Generate a random regression problem
        X, y = make_regression(
            n_samples=1000,
            n_features=4,
            n_informative=2,
            random_state=0,
            shuffle=False,
        )

        # Split data into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Split train data in proper training and calibration
        self.X_fit, self.X_calib, self.y_fit, self.y_calib = train_test_split(
            self.X_train, self.y_train, test_size=0.5, random_state=0
        )

        # Regression linear model
        model = linear_model.LinearRegression()

        # Definition of a predictor
        self.predictor = BasePredictor(model)

        # Definition of a calibrator
        self.calibrator = BaseCalibrator(
            nonconf_score_func=nonconformity_scores.absolute_difference,
            pred_set_func=prediction_sets.constant_interval,
        )

        # Definition of a dual calibrator
        self.dual_calibrator = BaseCalibrator(
            nonconf_score_func=nonconformity_scores.cqr_score,
            pred_set_func=prediction_sets.cqr_interval,
        )

        # Random splitter
        self.random_splitter = RandomSplitter(ratio=0.2)

        # Definition of a K-fold splitter
        self.kfold_splitter = KFoldSplitter(K=20, random_state=42)

    def test_proper_fit_predict(self):
        # Conformal predictor
        conformal_predictor = ConformalPredictor(
            predictor=self.predictor,
            calibrator=self.calibrator,
            splitter=self.random_splitter,
        )

        # Fit model and compute nonconformity scores
        conformal_predictor.fit(self.X_train, self.y_train)

        # Predict
        y_pred, y_lower, y_upper = conformal_predictor.predict(
            self.X_test, alpha=0.1
        )

    def test_bad_fit_predict(self):
        # Conformal predictor
        conformal_predictor = ConformalPredictor(
            predictor=self.predictor,
            calibrator=self.calibrator,
            splitter=self.random_splitter,
        )

        # Predict without fitting
        with self.assertRaises(RuntimeError):
            y_pred, y_lower, y_upper = conformal_predictor.predict(
                self.X_test, alpha=0.1
            )

    def test_pretrained_predictor(self):
        # Predictor initialized with trained model
        model = linear_model.LinearRegression()
        model.fit(self.X_fit, self.y_fit)
        trained_predictor = BasePredictor(model, is_trained=True)
        notrained_predictor = BasePredictor(model, is_trained=False)

        # Conformal predictor (good)
        conformal_predictor = ConformalPredictor(
            predictor=trained_predictor,
            calibrator=self.calibrator,
            splitter=self.random_splitter,
            train=False,
        )
        # Compute nonconformity scores
        conformal_predictor.fit(self.X_train, self.y_train)

        # Conformal predictor (bad)
        with self.assertRaises(RuntimeError):
            conformal_predictor = ConformalPredictor(
                predictor=notrained_predictor,
                calibrator=self.calibrator,
                splitter=self.random_splitter,
                train=False,
            )
            # Compute nonconformity scores
            conformal_predictor.fit(self.X_train, self.y_train)

        # Conformalization with no splitter (good)
        conformal_predictor = ConformalPredictor(
            predictor=trained_predictor,
            calibrator=self.calibrator,
            splitter=None,
            train=False,
        )
        conformal_predictor.fit(self.X_calib, self.y_calib)
        conformal_predictor.predict(self.X_test, alpha=0.1)

        # Conformalization with no splitter (bad)
        with self.assertRaises(RuntimeError):
            conformal_predictor = ConformalPredictor(
                predictor=notrained_predictor,
                calibrator=self.calibrator,
                splitter=None,
                train=False,
            )
            conformal_predictor.fit(self.X_calib, self.y_calib)

        # Conformalization with no splitter and train set to True (bad)
        with self.assertRaises(RuntimeError):
            conformal_predictor = ConformalPredictor(
                predictor=notrained_predictor,
                calibrator=self.calibrator,
                splitter=None,
                train=True,
            )
            conformal_predictor.fit(self.X_calib, self.y_calib)

    def test_pretrained_dualpredictor(self):
        # Predictor initialized with trained model
        model1 = linear_model.LinearRegression()
        model2 = linear_model.LinearRegression()
        model1.fit(self.X_fit, self.y_fit)
        model2.fit(self.X_fit, self.y_fit)
        trained_predictor = DualPredictor(
            [model1, model2], is_trained=[True, True]
        )
        notrained_predictor = DualPredictor(
            [model1, model2], is_trained=[False, True]
        )

        # Conformal predictor (good)
        conformal_predictor = ConformalPredictor(
            predictor=trained_predictor,
            calibrator=self.dual_calibrator,
            splitter=self.random_splitter,
            train=False,
        )
        # Compute nonconformity scores
        conformal_predictor.fit(self.X_train, self.y_train)

        # Conformal predictor (bad)
        with self.assertRaises(RuntimeError):
            conformal_predictor = ConformalPredictor(
                predictor=notrained_predictor,
                calibrator=self.dual_calibrator,
                splitter=self.random_splitter,
                train=False,
            )
            # Compute nonconformity scores
            conformal_predictor.fit(self.X_train, self.y_train)

        # Conformalization with no splitter (good)
        conformal_predictor = ConformalPredictor(
            predictor=trained_predictor,
            calibrator=self.dual_calibrator,
            splitter=None,
            train=False,
        )
        conformal_predictor.fit(self.X_calib, self.y_calib)
        conformal_predictor.predict(self.X_test, alpha=0.1)

        # Conformalization with no splitter (bad)
        with self.assertRaises(RuntimeError):
            conformal_predictor = ConformalPredictor(
                predictor=notrained_predictor,
                calibrator=self.dual_calibrator,
                splitter=None,
                train=False,
            )
            conformal_predictor.fit(self.X_calib, self.y_calib)

        # Conformalization with no splitter and train set to True (bad)
        with self.assertRaises(RuntimeError):
            conformal_predictor = ConformalPredictor(
                predictor=notrained_predictor,
                calibrator=self.dual_calibrator,
                splitter=None,
                train=True,
            )
            conformal_predictor.fit(self.X_calib, self.y_calib)

    def test_get_nconf_scores_split(self):
        # Conformal predictor
        conformal_predictor = ConformalPredictor(
            predictor=self.predictor,
            calibrator=self.calibrator,
            splitter=self.random_splitter,
        )

        # Fit model and compute nonconformity scores
        conformal_predictor.fit(self.X_train, self.y_train)

        # Get nconf scores
        nconf_scores = conformal_predictor.get_nonconformity_scores()

        self.assertIs(type(nconf_scores), dict)
        self.assertEqual(len(nconf_scores), 1)

    def test_get_nconf_scores_kfold(self):
        # Conformal predictor
        conformal_predictor = ConformalPredictor(
            predictor=self.predictor,
            calibrator=self.calibrator,
            splitter=self.kfold_splitter,
        )

        # Fit model and compute nonconformity scores
        conformal_predictor.fit(self.X_train, self.y_train)

        # Get nconf scores
        nconf_scores = conformal_predictor.get_nonconformity_scores()

        self.assertIs(type(nconf_scores), dict)
        self.assertEqual(len(nconf_scores), 20)

    def test_save_load(self):
        # Conformal predictor
        conformal_predictor = ConformalPredictor(
            predictor=self.predictor,
            calibrator=self.calibrator,
            splitter=self.kfold_splitter,
        )

        # Fit model and compute nonconformity scores
        conformal_predictor.fit(self.X_train, self.y_train)

        # Save copy of the conformal predictor
        conformal_predictor.save("my_cp.pkl")

        # load conformal predictor from file
        loaded_conformal_predictor = ConformalPredictor.load("my_cp.pkl")

        # Predict on X_test
        y_pred, y_pred_lo, y_pred_hi = conformal_predictor.predict(
            self.X_test, alpha=0.9
        )
        l_y_pred, l_y_pred_lo, l_y_pred_hi = loaded_conformal_predictor.predict(
            self.X_test, alpha=0.9
        )

        np.testing.assert_array_equal(y_pred, l_y_pred)
        np.testing.assert_array_equal(y_pred_lo, l_y_pred_lo)
        np.testing.assert_array_equal(y_pred_hi, l_y_pred_hi)

        os.remove("my_cp.pkl")

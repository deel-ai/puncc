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
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from deel.puncc.api.prediction import BasePredictor
from deel.puncc.classification import APS
from deel.puncc.classification import RAPS
from deel.puncc.metrics import classification_mean_coverage
from deel.puncc.metrics import classification_mean_size


def test_raps():
    # Generate a random regression problem
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_classes=2,
        random_state=0,
        shuffle=False,
    )

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Split train data into fit and calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0
    )

    # One hot encoding of classes
    y_fit_cat = to_categorical(y_fit)
    y_calib_cat = to_categorical(y_calib)
    y_test_cat = to_categorical(y_test)

    # Create rf classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=0)

    # Create a wrapper of the random forest model to redefine its predict method
    # into logits predictions. Make sure to subclass BasePredictor.
    # Note that we needed to build a new wrapper (over BasePredictor) only because
    # the predict(.) method of RandomForestClassifier does not predict logits.
    # Otherwise, it is enough to use BasePredictor (e.g., neural network with softmax).
    class RFPredictor(BasePredictor):
        def predict(self, X, **kwargs):
            return self.model.predict_proba(X, **kwargs)

    # Wrap model in the newly created RFPredictor
    rf_predictor = RFPredictor(rf_model)

    # CP method initialization
    raps_cp = RAPS(rf_predictor, k_reg=2, lambd=1)

    # The call to `fit` trains the model and computes the nonconformity
    # scores on the calibration set
    raps_cp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)

    # The predict method infers prediction intervals with respect to
    # the significance level alpha = 20%
    y_pred, set_pred = raps_cp.predict(X_test, alpha=0.2)

    # Compute marginal coverage
    coverage = classification_mean_coverage(y_test, set_pred)
    size = classification_mean_size(set_pred)

    assert y_pred is not None
    assert set_pred is not None


def test_aps():
    # Generate a random regression problem
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_classes=2,
        random_state=0,
        shuffle=False,
    )

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Split train data into fit and calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0
    )

    # One hot encoding of classes
    y_fit_cat = to_categorical(y_fit)
    y_calib_cat = to_categorical(y_calib)
    y_test_cat = to_categorical(y_test)

    # Create rf classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=0)

    # Create a wrapper of the random forest model to redefine its predict method
    # into logits predictions. Make sure to subclass BasePredictor.
    # Note that we needed to build a new wrapper (over BasePredictor) only because
    # the predict(.) method of RandomForestClassifier does not predict logits.
    # Otherwise, it is enough to use BasePredictor (e.g., neural network with softmax).
    class RFPredictor(BasePredictor):
        def predict(self, X, **kwargs):
            return self.model.predict_proba(X, **kwargs)

    # Wrap model in the newly created RFPredictor
    rf_predictor = RFPredictor(rf_model)

    # CP method initialization
    aps_cp = APS(rf_predictor)

    # The call to `fit` trains the model and computes the nonconformity
    # scores on the calibration set
    aps_cp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)

    # The predict method infers prediction intervals with respect to
    # the significance level alpha = 20%
    y_pred, set_pred = aps_cp.predict(X_test, alpha=0.2)

    # Compute marginal coverage
    coverage = classification_mean_coverage(y_test, set_pred)
    size = classification_mean_size(set_pred)

    assert y_pred is not None
    assert set_pred is not None

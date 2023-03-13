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
import pytest
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split

from deel.puncc.api.prediction import BasePredictor
from deel.puncc.api.prediction import DualPredictor
from deel.puncc.api.prediction import MeanVarPredictor
from deel.puncc.metrics import regression_mean_coverage
from deel.puncc.metrics import regression_sharpness
from deel.puncc.regression import AdaptiveEnbPI
from deel.puncc.regression import CQR
from deel.puncc.regression import CVPlus
from deel.puncc.regression import EnbPI
from deel.puncc.regression import LocallyAdaptiveCP
from deel.puncc.regression import SplitCP


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_split_cp(diabetes_data, alpha, random_state):

    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data

    # split train data into fit and calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, random_state=random_state
    )

    tf.keras.utils.set_random_seed(0)

    # Create NN predictor
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1))
    compile_kwargs = {"optimizer": "sgd", "loss": "mse"}
    predictor = BasePredictor(model, **compile_kwargs)

    ## Conformal predictor
    split_cp = SplitCP(predictor)
    kwargs = {"batch_size": 64, "epochs": 5}

    ## Fitting
    split_cp.fit(X_fit, y_fit, X_calib, y_calib, **kwargs)

    ## Predict
    y_pred, y_pred_lower, y_pred_upper = split_cp.predict(X_test, alpha=alpha)

    assert y_pred is not None
    assert not (True in np.isnan(y_pred))
    assert not (True in np.isnan(y_pred_lower))
    assert not (True in np.isnan(y_pred_upper))

    # Compute marginal coverage
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(
        y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper
    )


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_ne_split_cp(diabetes_data, alpha, random_state):

    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data

    # split train data into fit and calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, random_state=random_state
    )

    def w_estimator_gen(gamma):
        def w_estimator(X):
            return [gamma ** (len(X) + 1 - i) for i in range(len(X))]

        return w_estimator

    tf.keras.utils.set_random_seed(0)
    # Create NN predictor
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1))
    compile_kwargs = {"optimizer": "sgd", "loss": "mse"}
    predictor = BasePredictor(model, **compile_kwargs)

    # CP method initialization
    w_split_cp = SplitCP(predictor, weight_func=w_estimator_gen(0.95))

    # The fit method trains the model and computes the residuals on the
    # calibration set
    kwargs = {"batch_size": 64, "epochs": 5}
    w_split_cp.fit(X_fit, y_fit, X_calib, y_calib, **kwargs)  # type: ignore

    # The predict method infers prediction intervals with respect to
    # the risk alpha
    y_pred, y_pred_lower, y_pred_upper = w_split_cp.predict(X_test, alpha=alpha)

    assert y_pred is not None
    assert not (True in np.isnan(y_pred))
    assert not (True in np.isnan(y_pred_lower))
    assert not (True in np.isnan(y_pred_upper))

    # Compute marginal coverage
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(
        y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper
    )


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_locally_adaptive_cp(diabetes_data, alpha, random_state):

    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data

    # split train data into fit and calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, random_state=random_state
    )

    tf.keras.utils.set_random_seed(0)

    # Create NN regression object
    mu_model = tf.keras.Sequential()
    mu_model.add(tf.keras.layers.Dense(1))
    compile_kwargs1 = {"optimizer": "sgd", "loss": "mse"}

    # Create NN regression model
    var_model = tf.keras.Sequential()
    var_model.add(tf.keras.layers.Dense(10, activation="relu"))
    var_model.add(tf.keras.layers.Dense(1, activation="relu"))
    var_model.add(tf.keras.layers.Lambda(lambda x: tf.abs(x)))
    compile_kwargs2 = {"optimizer": "rmsprop", "loss": "mse"}

    # Create predictor
    predictor = MeanVarPredictor(
        models=[mu_model, var_model],
        compile_args=[compile_kwargs1, compile_kwargs2],
    )

    # CP method initialization
    la_cp = LocallyAdaptiveCP(predictor)

    # Fit and conformalize
    kwargs1 = {"batch_size": 64, "epochs": 5}
    kwargs2 = {"batch_size": 64, "epochs": 4}
    la_cp.fit(X_fit, y_fit, X_calib, y_calib, dictargs=[kwargs1, kwargs2])  # type: ignore
    y_pred, y_pred_lower, y_pred_upper = la_cp.predict(X_test, alpha=alpha)

    assert y_pred is not None
    assert not (True in np.isnan(y_pred))
    assert not (True in np.isnan(y_pred_lower))
    assert not (True in np.isnan(y_pred_upper))

    # Compute marginal coverage
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(
        y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper
    )


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_cqr(diabetes_data, alpha, random_state):

    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data

    # split train data into fit and calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, random_state=random_state
    )

    tf.keras.utils.set_random_seed(0)

    gbr_params = {
        "n_estimators": 250,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_samples_leaf": 9,
        "min_samples_split": 9,
        "random_state": random_state,
    }

    # Lower quantile model
    q_lo_model = tf.keras.Sequential()
    q_lo_model.add(tf.keras.layers.Dense(10, activation="relu"))
    q_lo_model.add(tf.keras.layers.Dense(10, activation="relu"))
    q_lo_model.add(tf.keras.layers.Dense(1, activation="relu"))
    loss_lo = tfa.losses.PinballLoss(tau=alpha / 2)
    compile_kwargs1 = {"optimizer": "sgd", "loss": loss_lo}

    # Upper quantile model
    q_hi_model = tf.keras.Sequential()
    q_hi_model.add(tf.keras.layers.Dense(10, activation="relu"))
    q_hi_model.add(tf.keras.layers.Dense(10, activation="relu"))
    q_hi_model.add(tf.keras.layers.Dense(1, activation="relu"))
    loss_hi = tfa.losses.PinballLoss(tau=1 - alpha / 2)
    compile_kwargs2 = {"optimizer": "sgd", "loss": loss_hi}

    # Wrap models in predictor
    predictor = DualPredictor(
        models=[q_lo_model, q_hi_model],
        compile_args=[compile_kwargs1, compile_kwargs2],
    )
    # CP method initialization
    crq = CQR(predictor)

    # Fit and conformalize
    kwargs1 = {"batch_size": 64, "epochs": 5}
    kwargs2 = {"batch_size": 64, "epochs": 4}
    crq.fit(X_fit, y_fit, X_calib, y_calib, dictargs=[kwargs1, kwargs2])  # type: ignore
    _, y_pred_lower, y_pred_upper = crq.predict(X_test, alpha=alpha)

    assert not (True in np.isnan(y_pred_lower))
    assert not (True in np.isnan(y_pred_upper))

    # Compute marginal coverage
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(
        y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper
    )


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_cv_plus(diabetes_data, alpha, random_state):

    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data

    tf.keras.utils.set_random_seed(0)

    # Create NN model and wrap it by a predictor
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1))
    compile_kwargs = {"optimizer": "sgd", "loss": "mse"}
    predictor = BasePredictor(model, **compile_kwargs)

    # CP method initialization
    cv_cp = CVPlus(predictor, K=20, random_state=random_state)

    # Fit and conformalize
    kwargs = {"batch_size": 64, "epochs": 5}
    cv_cp.fit(X_train, y_train, **kwargs)
    _, y_pred_lower, y_pred_upper = cv_cp.predict(X_test, alpha=alpha)

    # Compute marginal coverage
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(
        y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper
    )


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_enbpi(diabetes_data, alpha, random_state):

    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data

    tf.keras.utils.set_random_seed(0)

    # Create NN model and wrap it in a predictor
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1))
    compile_kwargs = {"optimizer": "sgd", "loss": "mse"}
    predictor = BasePredictor(model, **compile_kwargs)

    # Create EnbPI object
    enbpi = EnbPI(
        predictor, B=30, agg_func_loo=np.mean, random_state=random_state
    )

    # Fit and conformalize
    kwargs = {"batch_size": 64, "epochs": 5}
    enbpi.fit(X_train, y_train, **kwargs)
    y_pred, y_pred_lower, y_pred_upper = enbpi.predict(
        X_test, alpha=alpha, y_true=y_test, s=None
    )

    assert y_pred is not None
    assert not (True in np.isnan(y_pred))
    assert not (True in np.isnan(y_pred_lower))
    assert not (True in np.isnan(y_pred_upper))

    # Compute marginal coverage
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(
        y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper
    )


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_adaptive_enbpi(diabetes_data, alpha, random_state):

    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data

    tf.keras.utils.set_random_seed(0)

    # Create NN regression object
    mu_model = tf.keras.Sequential()
    mu_model.add(tf.keras.layers.Dense(1))
    compile_kwargs1 = {"optimizer": "sgd", "loss": "mse"}

    # Create NN regression model
    var_model = tf.keras.Sequential()
    var_model.add(tf.keras.layers.Dense(10, activation="relu"))
    var_model.add(tf.keras.layers.Dense(1, activation="relu"))
    var_model.add(tf.keras.layers.Lambda(lambda x: tf.abs(x)))
    compile_kwargs2 = {"optimizer": "rmsprop", "loss": "mse"}

    # Create predictor
    predictor = MeanVarPredictor(
        models=[mu_model, var_model],
        compile_args=[compile_kwargs1, compile_kwargs2],
    )

    # Build AdaptiveEnbPI object
    aenbpi = AdaptiveEnbPI(
        predictor,
        B=30,
        agg_func_loo=np.mean,
        random_state=random_state,
    )

    # Fit and conformalize
    kwargs1 = {"batch_size": 64, "epochs": 5}
    kwargs2 = {"batch_size": 64, "epochs": 4}
    aenbpi.fit(X_train, y_train, dictargs=[kwargs1, kwargs2])
    y_pred, y_pred_lower, y_pred_upper = aenbpi.predict(
        X_test, alpha=alpha, y_true=y_test, s=None
    )

    assert y_pred is not None
    assert not (True in np.isnan(y_pred))
    assert not (True in np.isnan(y_pred_lower))
    assert not (True in np.isnan(y_pred_upper))

    # Compute marginal coverage
    coverage = regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = regression_sharpness(
        y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper
    )

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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from deel.puncc import ops
from deel.puncc.core.cross_conformal import CVPlusRegressor
from deel.puncc.core.predictors import MultiPredictorStack, SklearnWrapper
from deel.puncc.core.samplers import IIDBootstrapSampler
from deel.puncc.core.split import WeightedQuantileMixin
from deel.puncc.metrics import regression_mean_coverage, regression_sharpness
from deel.puncc.regression.sequential import EnbPIRegressor
from deel.puncc.regression.split import (
    CQR,
    LeverageWeightedCP,
    LocallyAdaptiveCP,
    SplitConformalRegression,
)
from deel.puncc.core.calibration import CalibrationContext
from deel.puncc.exceptions import NotCalibratedError
from tests._utils import tensor, to_numpy


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(320, 3)).astype(np.float32)
    noise = rng.normal(scale=0.6, size=320)

    y = (2.0 * X[:, 0] - 0.6 * X[:, 1] + 0.4 * X[:, 2] + noise).astype(np.float32)

    X, y = tensor(X, "float32"), tensor(y, "float32")
    fit_data = X[:150], y[:150]
    calib_data = X[150:260], y[150:260]
    test_data = X[260:], y[260:]

    return fit_data, calib_data, test_data


def assert_valid_intervals(result, n_samples):
    prediction = to_numpy(result.prediction)
    intervals = to_numpy(result.prediction_set)

    assert prediction.shape[0] == n_samples
    assert intervals.shape == (n_samples, 2)

    assert np.all(np.isfinite(prediction))
    assert np.all(np.isfinite(intervals))
    assert np.all(intervals[:, 0] <= intervals[:, 1])

    return prediction, intervals


def assert_reasonable_coverage(y_true, intervals):
    lower = intervals[:, 0]
    upper = intervals[:, 1]

    coverage = regression_mean_coverage(to_numpy(y_true), lower, upper)
    width = regression_sharpness(lower, upper)

    assert 0.7 <= coverage <= 1.0
    assert np.isfinite(to_numpy(width)).all()
    assert float(to_numpy(width)) > 0


def test_split_cp(regression_data):
    (X_fit, y_fit), (X_calib, y_calib), (X_test, y_test) = regression_data

    cp = SplitConformalRegression(model=SklearnWrapper(LinearRegression()))
    cp.fit(X_fit, y_fit)
    cp.calibrate(X_calib, y_calib)

    expected_scores = np.abs(to_numpy(cp.model(X_calib)) - to_numpy(y_calib))
    np.testing.assert_allclose(
        to_numpy(cp.nc_scores),
        expected_scores,
        rtol=1e-5,
    )

    alpha = 0.1
    prediction, intervals = assert_valid_intervals(
        cp.predict(X_test, alpha=alpha),
        n_samples=len(X_test),
    )

    # Independent empirical-quantile oracle.
    sorted_scores = np.sort(np.concatenate([expected_scores, [np.inf]]))
    quantile_index = int(np.ceil((len(expected_scores) + 1) * (1 - alpha)) - 1)
    quantile = sorted_scores[quantile_index]

    expected_intervals = np.stack(
        [prediction - quantile, prediction + quantile],
        axis=-1,
    )
    np.testing.assert_allclose(
        intervals,
        expected_intervals,
        rtol=1e-5,
    )

    # Smaller alpha must produce intervals at least as wide.
    _, wider_intervals = assert_valid_intervals(
        cp.predict(X_test, alpha=0.05),
        n_samples=len(X_test),
    )

    assert np.all(wider_intervals[:, 0] <= intervals[:, 0])
    assert np.all(wider_intervals[:, 1] >= intervals[:, 1])

    assert_reasonable_coverage(y_test, intervals)


class WeightedSplitConformalRegression(
    WeightedQuantileMixin,
    SplitConformalRegression,
):
    pass


def first_column(X):
    return ops.squeeze(ops.convert_to_tensor(X), axis=-1)


def test_weighted_split_cp():
    X_calib = np.array(
        [[0.0], [1.0], [2.0], [3.0]],
        dtype=np.float32,
    )
    X_calib = tensor(X_calib, "float32")
    y_calib = np.array(
        [0.0, 1.0, 2.0, 5.0],
        dtype=np.float32,
    )
    y_calib = tensor(y_calib, "float32")
    X_test = np.array(
        [[4.0], [5.0]],
        dtype=np.float32,
    )
    X_test = tensor(X_test, "float32")

    cp = WeightedSplitConformalRegression(
        model=first_column,
        weight_function=lambda X: first_column(X) + 1.0,
    )
    cp.calibrate(X_calib, y_calib)

    np.testing.assert_allclose(
        to_numpy(cp.calibration_context.calibration_weights),
        [1.0, 2.0, 3.0, 4.0],
    )
    np.testing.assert_allclose(
        to_numpy(cp.nc_scores),
        [0.0, 0.0, 0.0, 2.0],
    )

    # Weights [1, 2, 3, 4, 5] and [1, 2, 3, 4, 6],
    # including the extra +inf conformity score.
    # Both weighted median thresholds equal 2.
    result = cp.predict(X_test, alpha=0.5)

    np.testing.assert_allclose(
        to_numpy(result.prediction_set),
        [[2.0, 6.0], [3.0, 7.0]],
    )


def test_locally_adaptive_cp(regression_data):
    (X_fit, y_fit), (X_calib, y_calib), (X_test, y_test) = regression_data

    cp = LocallyAdaptiveCP(
        model=SklearnWrapper(LinearRegression()),
        dispertion_estimator=SklearnWrapper(
            RandomForestRegressor(
                n_estimators=30,
                random_state=42,
            )
        ),
    )

    cp.fit(X_fit, y_fit)
    cp.calibrate(X_calib, y_calib)

    assert cp.len_calibr == len(X_calib)
    assert np.all(np.isfinite(to_numpy(cp.nc_scores)))

    _, intervals = assert_valid_intervals(
        cp.predict(X_test, alpha=0.1),
        n_samples=len(X_test),
    )

    # Local dispersion estimates should produce varying interval widths.
    widths = intervals[:, 1] - intervals[:, 0]
    assert np.ptp(widths) > 0

    assert_reasonable_coverage(y_test, intervals)


def test_leverage_weighted_cp(regression_data):
    (X_fit, y_fit), (X_calib, y_calib), (X_test, y_test) = regression_data

    cp = LeverageWeightedCP(model=SklearnWrapper(LinearRegression()))

    cp.fit(X_fit, y_fit)
    cp.calibrate(X_calib, y_calib)

    assert cp.len_calibr == len(X_calib)
    assert np.all(to_numpy(cp.nc_scores) >= 0)

    _, intervals = assert_valid_intervals(
        cp.predict(X_test, alpha=0.1),
        n_samples=len(X_test),
    )

    # Leverage depends on X, so the interval width should vary.
    assert np.ptp(intervals[:, 1] - intervals[:, 0]) > 0

    assert_reasonable_coverage(y_test, intervals)


def test_leverage_weighted_cp_requires_leverage_fit(regression_data):
    (X_fit, y_fit), (X_calib, y_calib), _ = regression_data

    pretrained_model = SklearnWrapper(LinearRegression().fit(to_numpy(X_fit), to_numpy(y_fit)))
    cp = LeverageWeightedCP(model=pretrained_model)

    with pytest.raises(
        RuntimeError,
        match=r"fit\(\) or fit_leverage\(\)",
    ):
        cp.calibrate(X_calib, y_calib)


def test_cqr(regression_data):
    (X_fit, y_fit), (X_calib, y_calib), (X_test, y_test) = regression_data

    lower_model = GradientBoostingRegressor(
        loss="quantile",
        alpha=0.1,
        n_estimators=60,
        random_state=42,
    )
    upper_model = GradientBoostingRegressor(
        loss="quantile",
        alpha=0.9,
        n_estimators=60,
        random_state=42,
    )

    cp = CQR(
        model=MultiPredictorStack(
            SklearnWrapper(lower_model),
            SklearnWrapper(upper_model),
        )
    )

    cp.fit(X_fit, y_fit)
    cp.calibrate(X_calib, y_calib)

    prediction, intervals = assert_valid_intervals(
        cp.predict(X_test, alpha=0.1),
        n_samples=len(X_test),
    )

    assert prediction.shape == (len(X_test), 2)
    assert len(cp.nc_scores) == len(X_calib)

    assert_reasonable_coverage(y_test, intervals)


def test_cv_plus(regression_data):
    (X_fit, y_fit), (X_calib, y_calib), (X_test, y_test) = regression_data

    X_train = ops.concatenate([X_fit, X_calib], axis=0)
    y_train = ops.concatenate([y_fit, y_calib], axis=0)

    cp = CVPlusRegressor(
        model=SklearnWrapper(LinearRegression()),
        K=5,
        random_state=42,
    )

    cp.fit(X_train, y_train)

    assert cp.len_calibr == len(X_train)

    prediction, intervals = assert_valid_intervals(
        cp.predict(X_test, alpha=0.1),
        n_samples=len(X_test),
    )

    assert prediction.shape == (len(X_test),)
    assert_reasonable_coverage(y_test, intervals)


def test_enbpi_conformalize_uses_signed_residuals():
    # Known signed residuals, including asymmetric extremes.
    residuals = np.array(
        [-10.0, -5.0, -4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 100.0, 101.0],
        dtype=np.float32,
    )
    residuals = tensor(residuals, "float32")

    cp = EnbPIRegressor(
        model=SklearnWrapper(LinearRegression()),
        beta_grid_size=4,
    )

    context = cp.compute_calibration_state(
        CalibrationContext(
            y_calib=residuals,
            y_pred=tensor(np.zeros_like(to_numpy(residuals)), "float32"),
        )
    )

    np.testing.assert_allclose(
        to_numpy(context.residuals),
        to_numpy(residuals),
    )

    prediction = np.array([10.0, 20.0], dtype=np.float32)
    prediction = tensor(prediction, "float32")

    result = cp.conformalize(
        prediction,
        alpha=0.3,
        calibration_context=context,
    )

    # beta grid: [0.0, 0.1, 0.2, 0.3].
    #
    # The inverse empirical CDF gives candidate offsets:
    # beta=0.0: [-10,   4] (width 14)
    # beta=0.1: [ -5,   6] (width 11) <- narrowest
    # beta=0.2: [ -4, 100] (width 104)
    # beta=0.3: [  0, 101] (width 101)
    #
    # The selected interval is prediction + [-5, 6].
    np.testing.assert_allclose(
        to_numpy(result.prediction_set),
        [[5.0, 16.0], [15.0, 26.0]],
    )

    # Conformalization must use the explicitly provided context,
    # not calibration state stored inside the predictor.
    shifted_context = cp.compute_calibration_state(
        CalibrationContext(
            y_calib=residuals + 2.0,
            y_pred=tensor(np.zeros_like(to_numpy(residuals)), "float32"),
        )
    )

    shifted_result = cp.conformalize(
        prediction,
        alpha=0.3,
        calibration_context=shifted_context,
    )

    np.testing.assert_allclose(
        to_numpy(shifted_result.prediction_set),
        [[7.0, 18.0], [17.0, 28.0]],
    )


def test_enbpi_fit_predict_update(regression_data):
    (X_fit, y_fit), (X_calib, y_calib), (X_test, y_test) = regression_data

    cp = EnbPIRegressor(
        model=SklearnWrapper(LinearRegression()),
        B=40,
        sampler=IIDBootstrapSampler(random_state=42),
        beta_grid_size=21,
    )

    # Calibration is initialized by fit(), not calibrate().
    with pytest.raises(NotCalibratedError):
        _ = cp.residuals

    with pytest.raises(
        RuntimeError,
        match="does not use a separate calibration set",
    ):
        cp.calibrate(X_calib, y_calib)

    assert cp.fit(X_fit, y_fit) is cp

    assert cp.window_size_ == len(X_fit)
    assert cp.len_calibr == len(X_fit)

    # The initial residuals are signed OOB residuals, not
    # absolute nonconformity scores.
    initial_residuals = to_numpy(cp.residuals).copy()
    oob_predictions = to_numpy(cp.calibration_context.y_pred)

    np.testing.assert_allclose(
        initial_residuals,
        to_numpy(y_fit) - oob_predictions,
        rtol=1e-5,
        atol=1e-5,
    )

    ensemble = cp.model
    fitted_models = tuple(ensemble.models_)

    # Prediction must not modify the calibration window.
    prediction, intervals = assert_valid_intervals(
        cp.predict(X_test, alpha=0.1),
        n_samples=len(X_test),
    )

    assert prediction.shape == (len(X_test),)
    assert intervals.shape == (len(X_test), 2)

    np.testing.assert_allclose(
        to_numpy(cp.residuals),
        initial_residuals,
    )

    # Feedback is supplied explicitly, after predictions are issued.
    batch_size = 5
    new_targets = y_test[:batch_size]
    issued_predictions = tensor(prediction[:batch_size], "float32")

    assert (
        cp.update(
            new_targets,
            prediction=issued_predictions,
        )
        is cp
    )

    # The oldest residuals are discarded so the window keeps
    # its original length.
    new_residuals = to_numpy(new_targets) - to_numpy(issued_predictions)

    expected_residuals = np.concatenate(
        [initial_residuals, new_residuals],
    )[-len(X_fit) :]

    np.testing.assert_allclose(
        to_numpy(cp.residuals),
        expected_residuals,
        rtol=1e-5,
        atol=1e-5,
    )

    assert cp.len_calibr == len(X_fit)
    assert cp.window_size_ == len(X_fit)

    # update() changes calibration data, not the trained ensemble.
    assert cp.model is ensemble
    assert len(cp.model.models_) == len(fitted_models)
    assert all(
        current is original
        for current, original in zip(cp.model.models_, fitted_models)
    )

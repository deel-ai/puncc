import pytest
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from puncc.utils import average_coverage, sharpness

from puncc.common.conformalizers import (
    SplitCP,
    WeightedSplitCP,
    LocallyAdaptiveCP,
    CQR,
    CvPlus,
    EnbPI,
    AdaptiveEnbPI,
)


RESULTS = {
    "scp": {"cov": 0.95, "width": 218.98},
    "wscp": {"cov": 0.97, "width": 241.99},
    "lacp": {"cov": 0.96, "width": 347.87},
    "cqr": {"cov": 0.9, "width": 237.8},
    "cv+": {"cov": 0.9, "width": 231.04},
    "enbpi": {"cov": 0.9, "width": 222.05},
    "aenbpi": {"cov": 0.91, "width": 286.64},
}


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
    # Create linear regression object
    regr_model = linear_model.LinearRegression()
    # CP method initialization
    split_cp = SplitCP(regr_model)
    # The fit method trains the model and computes the residuals on the
    # calibration set
    split_cp.fit(X_fit, y_fit, X_calib, y_calib)
    # The predict method infers prediction intervals with respect to
    # the risk alpha
    y_pred, y_pred_lower, y_pred_upper = split_cp.predict(X_test, alpha=alpha)
    assert y_pred is not None
    # Compute marginal coverage
    coverage = average_coverage(y_test, y_pred_lower, y_pred_upper)
    width = sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)
    res = {"cov": np.round(coverage, 2), "width": np.round(width, 2)}
    assert RESULTS["scp"] == res


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
    # Classifier for weight approximation
    calib_test_classifier = RandomForestClassifier(
        n_estimators=100, random_state=random_state
    )
    X_calib_test = np.concatenate((X_calib, X_test))
    y_calib_test = np.concatenate(
        (np.zeros_like(y_calib), np.ones_like(y_test))
    )
    calib_test_classifier.fit(
        X=X_calib_test,
        y=y_calib_test,
    )

    def w_estimator(X):
        return calib_test_classifier.predict_proba(X)[:, 1]

    # Create linear regression object
    regr_model = linear_model.LinearRegression()
    # CP method initialization
    w_split_cp = WeightedSplitCP(regr_model, w_estimator)
    # The fit method trains the model and computes the residuals on the
    # calibration set
    w_split_cp.fit(X_fit, y_fit, X_calib, y_calib)
    # The predict method infers prediction intervals with respect to
    # the risk alpha
    y_pred, y_pred_lower, y_pred_upper = w_split_cp.predict(
        X_test, alpha=alpha
    )
    assert y_pred is not None
    # Compute marginal coverage
    coverage = average_coverage(y_test, y_pred_lower, y_pred_upper)
    width = sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)
    res = {"cov": np.round(coverage, 2), "width": np.round(width, 2)}
    assert RESULTS["wscp"] == res


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
    # Create linear regression object
    regr_model = linear_model.LinearRegression()
    # Create RF regression object
    var_model = RandomForestRegressor(
        n_estimators=100, random_state=random_state
    )
    # CP method initialization
    la_cp = LocallyAdaptiveCP(regr_model, var_model)
    # Fit and conformalize
    la_cp.fit(X_fit, y_fit, X_calib, y_calib)
    y_pred, y_pred_lower, y_pred_upper, var_pred = la_cp.predict(
        X_test, alpha=alpha
    )
    assert (y_pred is not None) and (var_pred is not None)
    # Compute marginal coverage
    coverage = average_coverage(y_test, y_pred_lower, y_pred_upper)
    width = sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)
    res = {"cov": np.round(coverage, 2), "width": np.round(width, 2)}
    assert RESULTS["lacp"] == res


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
    # Create RF regression models for the upper and lower quantiles
    q_hi_model = RandomForestRegressor(
        n_estimators=100, random_state=random_state
    )
    q_lo_model = RandomForestRegressor(
        n_estimators=100, random_state=random_state
    )
    # CP method initialization
    crq = CQR(q_hi_model=q_hi_model, q_lo_model=q_lo_model)
    # Fit and conformalize
    crq.fit(X_fit, y_fit, X_calib, y_calib)
    y_pred_lower, y_pred_upper = crq.predict(X_test, alpha=alpha)
    # Compute marginal coverage
    coverage = average_coverage(y_test, y_pred_lower, y_pred_upper)
    width = sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)
    res = {"cov": np.round(coverage, 2), "width": np.round(width, 2)}
    assert RESULTS["cqr"] == res


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_cv_plus(diabetes_data, alpha, random_state):
    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data
    # Create RF regression object
    rf_model = RandomForestRegressor(
        n_estimators=100, random_state=random_state
    )
    # CP method initialization
    cv_cp = CvPlus(rf_model, K=20, random_state=random_state)
    # Fit and conformalize
    cv_cp.fit(X_train, y_train)
    _, y_pred_lower, y_pred_upper = cv_cp.predict(X_test, alpha=alpha)
    # Compute marginal coverage
    coverage = average_coverage(y_test, y_pred_lower, y_pred_upper)
    width = sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)
    res = {"cov": np.round(coverage, 2), "width": np.round(width, 2)}
    assert RESULTS["cv+"] == res


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_enbpi(diabetes_data, alpha, random_state):
    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data
    # Create RF regression object
    rf_model = RandomForestRegressor(
        n_estimators=100, random_state=random_state
    )
    # Fit and conformalize
    enbpi = EnbPI(
        rf_model, B=30, agg_func_loo=np.mean, random_state=random_state
    )
    enbpi.fit(X_train, y_train)
    y_pred, y_pred_lower, y_pred_upper = enbpi.predict(
        X_test, alpha=alpha, y_true=y_test, s=None
    )
    assert y_pred is not None
    # Compute marginal coverage
    coverage = average_coverage(y_test, y_pred_lower, y_pred_upper)
    width = sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)
    res = {"cov": np.round(coverage, 2), "width": np.round(width, 2)}
    assert RESULTS["enbpi"] == res


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_adaptive_enbpi(diabetes_data, alpha, random_state):
    # Get data
    (X_train, X_test, y_train, y_test) = diabetes_data
    # Create RF regression object
    rf_model = RandomForestRegressor(
        n_estimators=100, random_state=random_state
    )
    var_model = RandomForestRegressor(
        n_estimators=100, random_state=random_state
    )
    # Fit and conformalize
    aenbpi = AdaptiveEnbPI(
        rf_model,
        var_model,
        B=30,
        agg_func_loo=np.mean,
        random_state=random_state,
    )
    aenbpi.fit(X_train, y_train)
    y_pred, y_pred_lower, y_pred_upper = aenbpi.predict(
        X_test, alpha=alpha, y_true=y_test, s=None
    )
    assert y_pred is not None
    # Compute marginal coverage
    coverage = average_coverage(y_test, y_pred_lower, y_pred_upper)
    width = sharpness(y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper)
    res = {"cov": np.round(coverage, 2), "width": np.round(width, 2)}
    assert RESULTS["aenbpi"] == res

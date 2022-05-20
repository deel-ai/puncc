from random import random
import pytest
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from puncc.utils import average_coverage
from puncc.common.confomalizers import (
    SplitCP,
    LocallyAdaptiveCP,
    CQR,
    CvPlus,
    EnbPI,
    AdaptiveEnbPI,
)


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
    # Compute marginal coverage
    coverage = average_coverage(y_test, y_pred_lower, y_pred_upper)
    assert (y_pred is not None) and (coverage >= 1 - alpha)


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
    # Compute marginal coverage
    coverage = average_coverage(y_test, y_pred_lower, y_pred_upper)
    assert (
        (y_pred is not None)
        and (var_pred is not None)
        and (coverage >= 1 - alpha)
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
    assert coverage >= 1 - alpha


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
    cv_cp = CvPlus(rf_model, K=20)
    # Fit and conformalize
    cv_cp.fit(X_train, y_train)
    y_pred, y_pred_lower, y_pred_upper = cv_cp.predict(X_test, alpha=alpha)
    # Compute marginal coverage
    coverage = average_coverage(y_test, y_pred_lower, y_pred_upper)
    assert (y_pred is not None) and (coverage >= 1 - 2 * alpha)


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_enbpi(diabetes_data, alpha, random_state):
    assert True


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_adaptive_enbpi(diabetes_data, alpha, random_state):
    assert True

from random import random
import pytest
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from deel.puncc.api.utils import average_coverage

from deel.puncc.api import conformalization, calibration, prediction, splitting


def coverage_condition(empirical_cov, alpha):
    return empirical_cov >= 1 - alpha - 0.05  # 5% margin


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
    # Classifier for weight approximation
    calib_test_classifier = RandomForestClassifier(
        n_estimators=100, random_state=random_state
    )
    X_calib_test = np.concatenate((X_calib, X_test))
    y_calib_test = np.concatenate((np.zeros_like(y_calib), np.ones_like(y_test)))
    calib_test_classifier.fit(
        X=X_calib_test,
        y=y_calib_test,
    )

    def w_estimator(X):
        return calib_test_classifier.predict_proba(X)[:, 1]  # type: ignore

    # Create linear regression object
    regr_model = linear_model.LinearRegression()
    ## Prediction
    predictor = prediction.MeanPredictor(regr_model, is_trained=False)
    ## Calibration
    calibrator = calibration.MeanCalibrator(weight_func=w_estimator)
    ## Splitter
    splitter = splitting.IdSplitter(X_fit, y_fit, X_calib, y_calib)
    ## Conformal prediction init
    w_split_cp = conformalization.ConformalPredictor(
        predictor=predictor, calibrator=calibrator, splitter=splitter
    )
    # The fit method trains the model and computes the residuals on the
    # calibration set
    w_split_cp.fit(X_calib, y_calib)
    # The predict method infers prediction intervals with respect to
    # the risk alpha
    y_pred, y_pred_lower, y_pred_upper, _ = w_split_cp.predict(X_test, alpha=alpha)
    # Compute marginal coverage
    coverage = average_coverage(y_test, y_pred_lower, y_pred_upper)
    assert (y_pred is not None) and coverage_condition(coverage, alpha)

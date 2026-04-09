import numpy as np
import pytest
from sklearn import linear_model
from sklearn.model_selection import train_test_split

from deel.puncc.api.prediction import BasePredictor
from deel.puncc.metrics import regression_mean_coverage
from deel.puncc.metrics import regression_sharpness
from deel.puncc.regression import SplitCP

from deel.puncc.api.conformalization import GroupConformalPredictor
from deel.puncc.api.calibration import GroupCalibrator
from deel.puncc.api.nonconformity_scores import absolute_difference
from deel.puncc.api.prediction_sets import constant_interval
from deel.puncc.api.splitting import RandomSplitter


@pytest.mark.parametrize(
    "alpha, random_state",
    [(0.1, 42)],
)
def test_group_conformal_predictor(alpha, random_state):
    # 1. Generate biased synthetic data
    np.random.seed(random_state)
    n_samples = 2000
    X = np.random.uniform(-10, 10, size=(n_samples, 1))
    groups = np.random.choice([0, 1], size=n_samples)
    
    # Group 0 has low noise, Group 1 has high noise
    noise = np.where(
        groups == 0, 
        np.random.normal(0, 1, n_samples), 
        np.random.normal(0, 5, n_samples)
    )
    y = 2 * X[:, 0] + noise

    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=random_state
    )

    # 2. Test Standard CP (Baseline)
    predictor_std = BasePredictor(linear_model.LinearRegression())
    split_cp = SplitCP(predictor_std)
    split_cp.fit(X=X_train, y=y_train)
    _, y_pred_lower_std, y_pred_upper_std = split_cp.predict(X_test, alpha=alpha)

    cov_std_g0 = regression_mean_coverage(
        y_test[groups_test == 0], 
        y_pred_lower_std[groups_test == 0], 
        y_pred_upper_std[groups_test == 0]
    )
    cov_std_g1 = regression_mean_coverage(
        y_test[groups_test == 1], 
        y_pred_lower_std[groups_test == 1], 
        y_pred_upper_std[groups_test == 1]
    )

    # Standard CP should under-cover the high-noise group (Group 1)
    assert cov_std_g1 < 1 - alpha - 0.02

    # 3. Test Group-Balanced CP
    predictor_grp = BasePredictor(linear_model.LinearRegression())
    calibrator_grp = GroupCalibrator(
        nonconf_score_func=absolute_difference,
        pred_set_func=constant_interval
    )
    splitter = RandomSplitter(ratio=0.2, random_state=random_state)
    
    group_cp = GroupConformalPredictor(
        predictor=predictor_grp,
        calibrator=calibrator_grp,
        splitter=splitter,
        train=True
    )
    
    group_cp.fit(X_train, y_train, groups=groups_train)
    _, (y_pred_lower_grp, y_pred_upper_grp) = group_cp.predict(X_test, groups=groups_test, alpha=alpha)

    cov_grp_g0 = regression_mean_coverage(
        y_test[groups_test == 0], 
        y_pred_lower_grp[groups_test == 0], 
        y_pred_upper_grp[groups_test == 0]
    )
    cov_grp_g1 = regression_mean_coverage(
        y_test[groups_test == 1], 
        y_pred_lower_grp[groups_test == 1], 
        y_pred_upper_grp[groups_test == 1]
    )

    width_grp_g0 = regression_sharpness(
        y_pred_lower_grp[groups_test == 0], 
        y_pred_upper_grp[groups_test == 0]
    )
    width_grp_g1 = regression_sharpness(
        y_pred_lower_grp[groups_test == 1], 
        y_pred_upper_grp[groups_test == 1]
    )

    # Group CP should cover both groups at ~1-alpha
    assert cov_grp_g0 >= 1 - alpha - 0.05
    assert cov_grp_g1 >= 1 - alpha - 0.05

    # Group 1 should have wider intervals due to higher noise
    assert width_grp_g1 > width_grp_g0 * 2

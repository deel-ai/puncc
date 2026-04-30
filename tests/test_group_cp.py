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
    n_samples = 4000
    X = np.random.uniform(-10, 10, size=(n_samples, 1))
    groups = np.random.choice([0, 1], size=n_samples)
    
    # Group 0: low noise (std=1) | Group 1: high noise (std=5)
    noise = np.where(
        groups == 0, 
        np.random.normal(0, 1, n_samples), 
        np.random.normal(0, 5, n_samples)
    )
    y = 2 * X[:, 0] + noise

    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=random_state
    )

    # 2. Test Standard CP (Baseline) - PROVE THE PROBLEM
    predictor_std = BasePredictor(linear_model.LinearRegression())
    split_cp = SplitCP(predictor_std)
    split_cp.fit(X=X_train, y=y_train)
    _, y_low_std, y_hi_std = split_cp.predict(X_test, alpha=alpha)

    cov_std_g1 = regression_mean_coverage(
        y_test[groups_test == 1], y_low_std[groups_test == 1], y_hi_std[groups_test == 1]
    )
    # Standard CP should fail high-noise group coverage
    assert cov_std_g1 < 1 - alpha - 0.02 

    # 3. Test Group-Balanced CP - PROVE THE SOLUTION
    predictor_grp = BasePredictor(linear_model.LinearRegression())
    calibrator_grp = GroupCalibrator(
        nonconf_score_func=absolute_difference,
        pred_set_func=constant_interval
    )
    splitter = RandomSplitter(ratio=0.8, random_state=random_state)
    
    group_cp = GroupConformalPredictor(
        predictor=predictor_grp,
        calibrator=calibrator_grp,
        splitter=splitter
    )
    
    group_cp.fit(X_train, y_train, groups=groups_train, alpha=alpha)
    y_pred, (y_low, y_hi) = group_cp.predict(X_test, groups=groups_test)
    
    # --- ASSERTION A: Equal Coverage ---
    cov_grp_g0 = regression_mean_coverage(
        y_test[groups_test == 0], y_low[groups_test == 0], y_hi[groups_test == 0]
    )
    cov_grp_g1 = regression_mean_coverage(
        y_test[groups_test == 1], y_low[groups_test == 1], y_hi[groups_test == 1]
    )

    # Both groups should be near 1-alpha (0.90)
    assert cov_grp_g0 >= 1 - alpha - 0.03
    assert cov_grp_g1 >= 1 - alpha - 0.03

    # --- ASSERTION B: Efficiency/Sharpness Adaptation ---
    # This is the "missing link" in your test.
    # Group 1 has 5x the noise, so its intervals SHOULD be much wider.
    width_g0 = regression_sharpness(y_low[groups_test == 0], y_hi[groups_test == 0])
    width_g1 = regression_sharpness(y_low[groups_test == 1], y_hi[groups_test == 1])

    # Assert that the calibrator actually adapted to the group noise
    assert width_g1 > width_g0 * 3  # Should be ~5x wider, but 3x is a safe test margin

    # --- ASSERTION C: Global Coverage ---
    # Ensure it didn't break overall marginal coverage
    global_cov = regression_mean_coverage(y_test, y_low, y_hi)
    assert global_cov >= 1 - alpha - 0.02
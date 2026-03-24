from deel.puncc.api.conformal_predictor import ConformalPredictor
from deel.puncc.typing import Predictor, PredictorLike, TensorLike
from typing import Callable, Any, Literal
from collections.abc import Iterable
from deel.puncc.api.nonconformity_scores import difference, scaled_bbox_difference
from deel.puncc.api.prediction_sets import constant_bbox, scaled_bbox
from deel.puncc.api.corrections import bonferroni

class SplitBoxWise(ConformalPredictor):
    """Implementation of box-wise conformal object detection. For more info,
    we refer the user to the :ref:`theory overview page <theory splitboxwise>`

    :param BasePredictor | Any predictor: a predictive model.
    :param bool train: if False, prediction model(s) will not be (re)trained.
        Defaults to False.
    :param callable weight_func: function that takes as argument an array of
        features X and returns associated "conformality" weights, defaults to
        None.
    :param str method: chose between "additive" or "multiplicative" box-wise
        conformalization.
    :param int random_state: random seed used when the user does not
        provide a custom fit/calibration split in `fit` method.

    :raises ValueError: if method is not 'additive' or 'multiplicative'.

    .. _example splitboxwise:

    Example::

        from deel.puncc.object_detection import SplitBoxWise
        import numpy as np

        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor

        from deel.puncc.metrics import object_detection_mean_coverage
        from deel.puncc.metrics import object_detection_mean_area

        # Generate a random regression problem
        X, y = make_regression(
            n_samples=1000,
            n_features=4,
            n_informative=2,
            n_targets=4,
            random_state=0,
            shuffle=False,
        )

        # Create dummy object localization data formated as (x1, y1, x2, y2)
        y = np.abs(y)
        x1 = np.min(y[:, :2], axis=1)
        y1 = np.min(y[:, 2:], axis=1)
        x2 = np.max(y[:, :2], axis=1)
        y2 = np.max(y[:, 2:], axis=1)
        y = np.array([x1, y1, x2, y2]).T


        # Split data into train and test
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Split train data into fit and calibration
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Create a random forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=0)

        # CP method initialization
        od_cp = SplitBoxWise(rf_model, method="multiplicative", train=True)

        # The call to `fit` trains the model and computes the nonconformity
        # scores on the calibration set
        od_cp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)

        # The predict method infers prediction intervals with respect to
        # the significance level alpha = 20%
        y_pred, box_inner, box_outer = od_cp.predict(X_test, alpha=0.2)

        # Compute marginal coverage and average width of the prediction intervals
        coverage = object_detection_mean_coverage(box_outer, y_test)
        average_area = object_detection_mean_area(box_outer)
        print(f"Marginal coverage: {np.round(coverage, 2)}")
    """
    def __init__(self,
                 model:Predictor|PredictorLike,
                 method:Literal["additive", "multiplicative"]="additive",
                 weight_function:Callable[[Iterable[Any]], Iterable[float]]|None = None,
                 fit_function:Callable[[Predictor, Iterable[Any], TensorLike], Predictor]|None = None):
        if method == "additive":
            nc_score_function = difference()
            pred_set_function = constant_bbox()
        elif method == "multiplicative":
            nc_score_function = scaled_bbox_difference()
            pred_set_function = scaled_bbox()
        else:
            raise ValueError(f"Unknown method '{method} for SplitBoxWise'. Supported methods are 'additive' and 'multiplicative'.")
        super().__init__(model=model,
                            nc_score_function=nc_score_function,
                            pred_set_function=pred_set_function,
                            weight_function=weight_function,
                            fit_function=fit_function)
        
    def predict(self,
                X_test:Iterable[Any],
                alpha:float|TensorLike,
                correction:Callable|None = bonferroni(4))->tuple[TensorLike, Any]:
        return super().predict(X_test, alpha, correction)
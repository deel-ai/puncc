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
"""
This module implements conformal object detection procedures.
"""
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from deel.puncc.api import nonconformity_scores
from deel.puncc.api import prediction_sets
from deel.puncc.api.conformalization import SplitConformalPredictor
from deel.puncc.api.corrections import bonferroni
from deel.puncc.api.prediction import BasePredictor


class SplitBoxWise(SplitConformalPredictor):
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

    def __init__(
        self,
        predictor: Union[BasePredictor, Any],
        *,
        train: bool = False,
        weight_func: Optional[Callable] = None,
        method: str = "additive",
        random_state: int = 0,
    ):
        self.predictor = predictor
        if method == "additive":
            nonconf_score_func = nonconformity_scores.difference
            pred_set_func = prediction_sets.constant_bbox
        elif method == "multiplicative":
            nonconf_score_func = nonconformity_scores.scaled_bbox_difference
            pred_set_func = prediction_sets.scaled_bbox
        else:
            raise ValueError("method must be 'additive' or 'multiplicative'.")

        super().__init__(predictor=predictor,
                         nonconf_score_func=nonconf_score_func,
                         pred_set_func=pred_set_func,
                         train=train,
                         random_state=random_state,
                         weight_func=weight_func)

    def predict(
        self,
        X_test: Iterable,
        alpha: float,
        correction_func: Callable = lambda x: bonferroni(x, 4),
    ) -> Tuple[np.ndarray]:
        """Conformal object detection (w.r.t target miscoverage alpha) for
        new samples.

        :param Iterable X_test: features of new samples.
        :param float alpha: target maximum miscoverage.
        :param Callable correction_func: correction for multiple hypothesis
            testing in the case of multivariate regression. Defaults to
            Bonferroni correction.

        :returns: y_pred, y_lower, y_higher
        :rtype: Tuple[ndarray]

        """

        # Return format: y_pred, y_lower, y_higher
        return self.conformal_predictor.predict(X_test, alpha, correction_func)

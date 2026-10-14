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
This module implements conformal object detection methods.
"""

from collections.abc import Iterable
from typing import Any
from deel.puncc.core.conformal import ConformalPrediction
from deel.puncc.core.split import SplitConformalPredictor
from deel.puncc.od.base import BoxExtensionMode
from deel.puncc.typing import FitFunction, Predictor, PredictorLike, TensorLike
from deel.puncc.nonconformity_scores import (
    boxwise_difference,
    difference,
    scaled_bbox_difference,
)
from deel.puncc.prediction_sets import constant_bbox, scaled_bbox
from deel.puncc.corrections import AlphaCorrection, bonferroni


# TODO : passer d'une conformalization de regression 4D à une conformalization d'OD localization
class SplitBoxWise(SplitConformalPredictor):
    """
    Implementation of box-wise conformal object detection.

    This method constructs conformal bounds around predicted bounding boxes
    by calibrating the four box coordinates. Box extensions can be additive
    (constant offsets) or multiplicative (offsets proportional to box size).

    Args:
        model: Predictive model returning bounding boxes with shape
            (n_samples, 4), in (x_min, y_min, x_max, y_max) format.
        box_extension_mode: Extension mode, either "additive" or
            "multiplicative". Defaults to "additive".
        fit_function: Optional custom function used to train the model.
            Defaults to None.

    Raises:
        ValueError: If box_extension_mode is invalid.

    Notes:
        By default, Bonferroni correction is applied across the four
        bounding-box coordinates.

        The returned prediction set has shape (n_samples, 4, 2).
        Its last dimension contains the inner and outer conformal bounds.

    .. _example splitboxwise:

    Example::

        import numpy as np

        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor

        from deel.puncc.od.split import SplitBoxWise
        from deel.puncc.metrics import (
            object_detection_mean_coverage,
            object_detection_mean_area,
        )

        # Generate a random regression problem
        X, y = make_regression(
            n_samples=1000,
            n_features=4,
            n_informative=2,
            n_targets=4,
            random_state=0,
            shuffle=False,
        )

        # Create dummy object localization data in (x1, y1, x2, y2) format
        y = np.abs(y)
        x1 = np.min(y[:, :2], axis=1)
        y1 = np.min(y[:, 2:], axis=1)
        x2 = np.max(y[:, :2], axis=1)
        y2 = np.max(y[:, 2:], axis=1)
        y = np.array([x1, y1, x2, y2]).T

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Split training data into fit and calibration sets
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0
        )

        # Create a random forest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=0,
        )

        # Initialize the conformal predictor
        od_cp = SplitBoxWise(
            model=rf_model,
            box_extension_mode="multiplicative",
        )

        # Fit the model and calibrate separately
        od_cp.fit(X_fit, y_fit)
        od_cp.calibrate(X_calib, y_calib)

        # Construct conformal bounding boxes for alpha = 20%
        y_pred, box_bounds = od_cp.predict(X_test, alpha=0.2)

        # Extract inner and outer bounding boxes
        box_inner = box_bounds[..., 0]
        box_outer = box_bounds[..., 1]

        # Compute marginal coverage and average outer bounding-box area
        coverage = object_detection_mean_coverage(box_outer, y_test)
        average_area = object_detection_mean_area(box_outer)

        print(f"Marginal coverage: {np.round(coverage, 2)}")
        print(f"Average bounding-box area: {np.round(average_area, 2)}")
    """

    def __init__(
        self,
        model: Predictor | PredictorLike,
        *,
        box_extension_mode: BoxExtensionMode | str = BoxExtensionMode.ADDITIVE,
        fit_function: FitFunction | None = None,
    ):
        box_extension_mode = BoxExtensionMode(box_extension_mode)
        if box_extension_mode == BoxExtensionMode.ADDITIVE:
            nc_score_function = boxwise_difference()
            pred_set_function = constant_bbox()
        elif box_extension_mode == BoxExtensionMode.MULTIPLICATIVE:
            nc_score_function = scaled_bbox_difference()
            pred_set_function = scaled_bbox()
        else:
            raise ValueError(
                f"Unknown method '{box_extension_mode}' for SplitBoxWise. Supported methods are 'additive' and 'multiplicative'."
            )
        super().__init__(
            model=model,
            nc_score_function=nc_score_function,
            pred_set_function=pred_set_function,
            fit_function=fit_function,
        )

    def predict(
        self,
        X_test: Iterable[Any],
        alpha: float | TensorLike,
        *,
        alpha_correction: AlphaCorrection | None = bonferroni(4),
    ) -> ConformalPrediction[
        TensorLike,
        TensorLike,
    ]:
        return super().predict(
            X_test,
            alpha,
            alpha_correction=alpha_correction,
        )

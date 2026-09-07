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
This module implements conformal anomaly detection procedures.
"""
from collections.abc import Callable
from typing import Any, Iterable, Self


from deel.puncc.core.calibration import CalibrationContext
from deel.puncc.core.conformal import CacheKey
from deel.puncc.core.predictors import make_predictor
from deel.puncc.typing import Predictor, PredictorLike, TensorLike
from deel.puncc.backend.keras import ops



# TODO : should inherit from ConformalPredictor in a way or another, but need rethinking to overload correctly basic methods
class SplitCAD:
    """Split conformal anomaly detection method based on Laxhammar's algorithm.
    The anomaly detection is based on the calibrated threshold (through
    conformal prediction) of underlying anomaly detection (model's) scores.
    For more details, we refer the user to the :ref:`theory overview
    page <theory_overview>`.

    :param BasePredictor predictor: a predictor implementing fit and predict.
    :param bool train: if False, prediction model(s) will not be (re)trained.
        Defaults to True.
    :param float random_state: random seed used when the user does not
        provide a custom fit/calibration split in `fit` method.

    Example::

        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.datasets import make_moons
        import matplotlib.pyplot as plt

        from deel.puncc.anomaly_detection import SplitCAD
        from deel.puncc.api.prediction import BasePredictor

        # We generate the two moons dataset
        dataset = 4 * make_moons(n_samples=1000, noise=0.05, random_state=0)[
            0
        ] - np.array([0.5, 0.25])

        # We generate uniformly new (test) data points
        rng = np.random.RandomState(42)
        z_test = rng.uniform(low=-6, high=6, size=(150, 2))


        # The nonconformity scores are defined as the IF scores (anomaly score).
        # By default, score_samples return the opposite of IF scores.
        # We need to redefine the predict to output the nonconformity scores.
        class ADPredictor(BasePredictor):
            def predict(self, X):
                return -self.model.score_samples(X)

        # Instantiate the Isolation Forest (IF) anomaly detection model
        # and wrap it in a predictor
        if_predictor = ADPredictor(IsolationForest(random_state=42))

        # Instantiate CAD on top of IF predictor
        if_cad = SplitCAD(if_predictor, train=True, random_state=0)

        # Fit the IF on the proper fitting dataset and
        # calibrate it using calibration dataset.
        # The two datasets are sampled randomly with a ration of 7:3,
        # respectively.
        if_cad.fit(z=dataset, fit_ratio=0.7)

        # We set the maximum false detection rate to 1%
        alpha = 0.01

        # The method `predict` is called on the new data points
        # to test which are anomalous and which are not
        results = if_cad.predict(z_test, alpha=alpha)

        anomalies = z_test[results]
        not_anomalies = z_test[np.invert(results)]

        # Plot results
        plt.scatter(dataset[:, 0], dataset[:, 1], s=10, label="Inliers")
        plt.scatter(
            anomalies[:, 0],
            anomalies[:, 1],
            marker="x",
            color="red",
            s=40,
            label="Anomalies",
        )
        plt.scatter(
            not_anomalies[:, 0],
            not_anomalies[:, 1],
            marker="x",
            color="blue",
            s=40,
            label="Normal",
        )
        plt.xticks(())
        plt.yticks(())
        plt.legend()
    """
    __slots__ = ("model", "fit_function", "calibration_context", "conformalization_cache")

    # Any conformal method should have a model attribute.
    def __init__(self, model:Predictor|PredictorLike,
                 fit_function:Callable[..., Predictor] | None|None = None):
        self.model = make_predictor(model)
        self.fit_function = fit_function
        self.calibration_context = CalibrationContext()
        # TODO : cache is not used here. uniformize with other methods
        self.conformalization_cache:dict[CacheKey, Any] = {}

    def fit(self,
            z:Iterable[Any],
            *args:Any, 
            **kwargs:Any
            )->Self:
        """
        Fit the underlying predictive model.

        The custom fit_function is used when provided. Otherwise the predictor's own fit method is called.

        Args:
            X:
                Training inputs.

            y:
                Training targets.

            *args:
                Additional positional arguments forwarded to the fitting function.

            **kwargs:
                Additional keyword arguments forwarded to the fitting function.

        Returns:
            The predictor with its underlying model fitted.
        """
        if self.fit_function is not None:
            self.model = self.fit_function(self.model, z, *args, **kwargs)
            return self
        
        fit_method = getattr(
            self.model,
            "fit",
            None,
        )
        if callable(fit_method):
            fit_method(
                z,
                *args,
                **kwargs,
            )
            return self
        raise NotImplementedError("The model does not have a fit method and no fit_function was provided. Please provide a pretrained model or a fit_function.")

    def calibrate(self,
                  z_calib:Iterable[Any])->Self:
        self.conformalization_cache.clear()
        self.calibration_context.clear()
        self.calibration_context.update(
            z_calib=z_calib,
            nc_scores=self.model(z_calib)
        )
        return self

    def predict(self, z_test: Iterable, alpha) -> TensorLike:
        """Predict whether each example is an anomaly or not. The decision is
        taken based on the calibrated threshold (through conformal prediction)
        of underlying anomaly detection scores.

        :param Iterable z_test: new data points.
        :param float alpha: target maximum FDR.

        :returns: outlier tag. True if outlier, False otherwise.
        :rtype: Iterables[bool]

        """
        n_calib = self.calibration_context.size
        # TODO : separate the quantile computation to allow usage of weighting mixin
        quantile = ops.weighted_quantile(self.calibration_context.nc_scores, (1 - alpha) * (n_calib + 1) / n_calib, axis=0) 
        test_nonconf_scores = self.model(z_test)
        anomaly_pred = ops.logical_not(test_nonconf_scores <= quantile)
        # TODO : maybe uniformize the output type with others methods
        return anomaly_pred

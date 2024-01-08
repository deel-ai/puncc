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
import logging
from typing import Iterable
from typing import Optional
from typing import Tuple

import numpy as np

from deel.puncc.api.calibration import ScoreCalibrator
from deel.puncc.api.splitting import IdSplitter
from deel.puncc.api.splitting import RandomSplitter

logger = logging.getLogger(__name__)


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

    def __init__(self, predictor, *, train=True, random_state: float = None):
        self.predictor = predictor
        self.calibrator = ScoreCalibrator(nonconf_score_func=predictor.predict)

        self.train = train

        self.random_state = random_state

        self.__is_fit = False

    def fit(
        self,
        *,
        z: Optional[Iterable] = None,
        fit_ratio: float = 0.8,
        z_fit: Optional[Iterable] = None,
        z_calib: Optional[Iterable] = None,
        **kwargs: Optional[dict],
    ):
        """This method fits the models on the fit data
        and computes nonconformity scores on calibration data.
        If z are provided, randomly split data into
        fit and calib subsets w.r.t to the fit_ratio.
        In case z_fit and z_calib are provided,
        the conformalization is performed on the given user defined
        fit and calibration sets.

        .. NOTE::

            If z is provided, `fit` ignores
            any user-defined fit/calib split.


        :param Iterable z: data points from the training dataset.
        :param float fit_ratio: the proportion of samples assigned to the
            fit subset.
        :param Iterable z_fit: data points from the fit dataset.
        :param Iterable z_calib: data points from the calibration dataset.
        :param dict kwargs: predict configuration to be passed to the model's
            fit method.

        :raises RuntimeError: no dataset provided.

        """

        if z is not None:
            splitter = RandomSplitter(
                ratio=fit_ratio, random_state=self.random_state
            )

        elif z_fit is not None and z_calib is not None:
            splitter = IdSplitter(z_fit, z_fit, z_calib, z_calib)

        elif (
            self.predictor.is_trained and z_fit is None and z_calib is not None
        ):
            splitter = IdSplitter(
                np.empty_like(z_calib), np.empty_like(z_calib), z_calib, z_calib
            )

        else:
            raise RuntimeError("No dataset provided.")

        # Apply splitter
        z_fit, _, z_calib, _ = splitter(z, z)[0]

        # Fit underlying model and calibrator
        if self.train:
            logger.info("Fitting model")
            self.predictor.fit(z_fit, **kwargs)

        # Make sure that predictor is already trained if train arg is False
        elif self.train is False and self.predictor.is_trained is False:
            raise RuntimeError(
                "'train' argument is set to 'False' but model is not pre-trained"
            )

        else:  # Skipping training
            logger.info("Skipping training.")

        # Fitting calibrator
        self.calibrator.fit(z_calib)

        self.__is_fit = True

    def predict(self, z_test: Iterable, alpha) -> Tuple[np.ndarray]:
        """Predict whether each example is an anomaly or not. The decision is
        taken based on the calibrated threshold (through conformal prediction)
        of underlying anomaly detection scores.

        :param Iterable z_test: new data points.
        :param float alpha: target maximum FDR.

        :returns: outlier tag. True if outlier, False otherwise.
        :rtype: Iterables[bool]

        """

        if self.__is_fit is None:
            raise RuntimeError("Fit method should be called before predict.")

        anomaly_pred = np.invert(
            self.calibrator.is_conformal(z_test, alpha=alpha)
        )

        return anomaly_pred

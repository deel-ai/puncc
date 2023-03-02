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
This module implements usual conformal classification wrappers.
"""
from typing import Iterable
from typing import Optional
from typing import Tuple

from deel.puncc.api import nonconformity_scores
from deel.puncc.api import prediction_sets
from deel.puncc.api.calibration import BaseCalibrator
from deel.puncc.api.conformalization import ConformalPredictor
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.api.splitting import IdSplitter


class RAPS:
    """Implementation of Regularized Adaptive Prediction Sets (RAPS). The hyperparameters :math:`\\lambda` and :math:`k_{reg}` are used to encourage small prediction sets.
    For more details, we refer the user to the :ref:`theory overview page <theory raps>`.

    :param BasePredictor predictor: a predictor implementing fit and predict.
    :param bool train: if False, prediction model(s) will not be trained and will be used as is. Defaults to True.
    :param float lambd: positive weight associated to the regularization term that encourages small set sizes. If :math:`\\lambda = 0`, there is no regularization and the implementation identifies with **APS**.
    :param float k_reg: class rank (ordered by descending probability) starting from which the regularization is applied. For example, if :math:`k_{reg} = 3`, then the fourth most likely estimated class has an extra penalty of size :math:`\\lambda`.

    .. note::

        If :math:`\\lambda = 0`, there is no regularization and the implementation identifies with **APS**.

    .. _example raps:

    Example::

        from deel.puncc.classification import RAPS
        from deel.puncc.api.prediction import BasePredictor

        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        from deel.puncc.metrics import classification_mean_coverage
        from deel.puncc.metrics import classification_mean_size

        from tensorflow.keras.utils import to_categorical


        # Generate a random regression problem
        X, y = make_classification(n_samples=1000, n_features=4, n_informative=2,
                    n_classes = 2,random_state=0, shuffle=False)

        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=0
        )

        # Split train data into fit and calibration
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train, y_train, test_size=.2, random_state=0
        )

        # One hot encoding of classes
        y_fit_cat = to_categorical(y_fit)
        y_calib_cat = to_categorical(y_calib)
        y_test_cat = to_categorical(y_test)

        # Create rf classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=0)

        # Create a wrapper of the random forrest model to redefine its predict method
        # into logits predictions. Make sure to subclass BasePredictor.
        # Note that we needed to build a new wrapper (over BasePredictor) only because
        # the predict(.) method of RandomForestClassifier does not predict logits.
        # Otherwise, it is enough to use BasePredictor (e.g., neural network with softmax).
        class RFPredictor(BasePredictor):
            def predict(self, X, **kwargs):
                return self.model.predict_proba(X, **kwargs)

        # Wrap model in the newly created RFPredictor
        rf_predictor = RFPredictor(rf_model)

        # CP method initialization
        raps_cp = RAPS(rf_predictor, k_reg=1, lambd=0)

        # The call to `fit` trains the model and computes the nonconformity
        # scores on the calibration set
        raps_cp.fit(X_fit, y_fit, X_calib, y_calib)

        # The predict method infers prediction intervals with respect to
        # the significance level alpha = 20%
        y_pred, set_pred = raps_cp.predict(X_test, alpha=.2)

        # Compute marginal coverage
        coverage = classification_mean_coverage(y_test, set_pred)
        width = classification_mean_size(set_pred)

        print(f"Marginal coverage: {np.round(coverage, 2)}")
        print(f"Average width: {np.round(width, 2)}")

    """

    def __init__(self, predictor, train=True, lambd=0, k_reg=1):
        self.predictor = predictor
        self.calibrator = BaseCalibrator(
            nonconf_score_func=nonconformity_scores.raps_score(
                lambd=lambd, k_reg=k_reg
            ),
            pred_set_func=prediction_sets.raps_set(lambd=lambd, k_reg=k_reg),
            weight_func=None,
        )
        self.train = train

    def fit(
        self,
        X_fit: Iterable,
        y_fit: Iterable,
        X_calib: Iterable,
        y_calib: Iterable,
        **kwargs: Optional[dict],
    ):
        """This method fits the models to the fit data (X_fit, y_fit)
        and computes residuals on (X_calib, y_calib).

        :param ndarray|DataFrame|Tensor X_fit: features from the fit dataset.
        :param ndarray|DataFrame|Tensor y_fit: labels from the fit dataset.
        :param ndarray|DataFrame|Tensor X_calib: features from the calibration dataset.
        :param ndarray|DataFrame|Tensor y_calib: labels from the calibration dataset.
        :param dict kwargs: predict configuration to be passed to the model's predict method.
        """
        self.conformal_predictor = ConformalPredictor(
            predictor=self.predictor,
            calibrator=self.calibrator,
            splitter=IdSplitter(X_fit, y_fit, X_calib, y_calib),
        )
        self.conformal_predictor.fit(X=None, y=None, **kwargs)  # type: ignore

    def predict(
        self,
        X_test: Iterable,
        alpha: float,
        lambd: float = 0,
        k_reg: float = 2,
    ) -> Tuple[Iterable, Iterable, Optional[Iterable]]:
        """Conformal interval predictions (w.r.t target miscoverage alpha)
        for new samples.

        :param ndarray|DataFrame|Tensor X_test: features of new samples.
        :param float alpha: target maximum miscoverage.
        :param float lambd: positive weight associated to the regularization term. If :math:`\\lambda = 0`, there is no regularization and the implementation identifies with **APS**.
        :param float k_reg: class rank (ordered by descending probability) starting from which the regularization is applied. For example, if :math:`k_{reg} = 3`, then the fourth most likely estimated class has an extra penalty of size :math:`\\lambda`.

        :returns: y_pred, y_lower, y_higher
        :rtype: Tuple[ndarray]
        """

        if not hasattr(self, "conformal_predictor"):
            raise RuntimeError("Fit method should be called before predict.")

        (y_pred, set_pred) = self.conformal_predictor.predict(X_test, alpha=alpha)

        return y_pred, set_pred

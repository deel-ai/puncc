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
This module implements conformal classification procedures.
"""
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from deel.puncc.api import nonconformity_scores
from deel.puncc.api import prediction_sets
from deel.puncc.api.calibration import BaseCalibrator
from deel.puncc.api.conformalization import ConformalPredictor
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.api.splitting import IdSplitter
from deel.puncc.api.splitting import RandomSplitter


class RAPS:
    """Implementation of Regularized Adaptive Prediction Sets (RAPS).
    The hyperparameters :math:`\\lambda` and :math:`k_{reg}` are used to
    encourage small prediction sets. For more details, we refer the user to the
    :ref:`theory overview page <theory raps>`.

    :param BasePredictor predictor: a predictor implementing fit and predict.
    :param bool train: if False, prediction model(s) will not be trained and
        will be used as is. Defaults to True.
    :param float random_state: random seed used when the user does not
        provide a custom fit/calibration split in `fit` method.
    :param float lambd: positive weight associated to the regularization term
        that encourages small set sizes. If :math:`\\lambda = 0`, there is no
        regularization and the implementation identifies with **APS**.
    :param float k_reg: class rank (ordered by descending probability) starting
        from which the regularization is applied. For example,
        if :math:`k_{reg} = 3`, then the fourth most likely estimated class has
        an extra penalty of size :math:`\\lambda`.
    :param bool rand: turn on or off randomization used in raps algorithm.
        One consequence of turning off randomization is avoiding empty
        prediction sets.

    .. note::

        If :math:`\\lambda = 0`, there is no regularization and the
        implementation identifies with **APS**.

    .. _example raps:

    Example::

        from deel.puncc.classification import RAPS
        from deel.puncc.api.prediction import BasePredictor

        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        from deel.puncc.metrics import classification_mean_coverage
        from deel.puncc.metrics import classification_mean_size

        import numpy as np

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

        # Create a wrapper of the random forest model to redefine its predict method
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
        raps_cp = RAPS(rf_predictor, k_reg=2, lambd=1)

        # The call to `fit` trains the model and computes the nonconformity
        # scores on the calibration set
        raps_cp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)


        # The predict method infers prediction intervals with respect to
        # the significance level alpha = 20%
        y_pred, set_pred = raps_cp.predict(X_test, alpha=.2)

        # Compute marginal coverage
        coverage = classification_mean_coverage(y_test, set_pred)
        size = classification_mean_size(set_pred)

        print(f"Marginal coverage: {np.round(coverage, 2)}")
        print(f"Average prediction set size: {np.round(size, 2)}")

    """

    def __init__(
        self,
        predictor: Union[BasePredictor, Any],
        train: bool = True,
        random_state: float = None,
        lambd: float = 0,
        k_reg: int = 1,
        rand: bool = True,
    ):
        self.predictor = predictor
        self.calibrator = BaseCalibrator(
            nonconf_score_func=nonconformity_scores.raps_score_builder(
                lambd=lambd, k_reg=k_reg, rand=rand
            ),
            pred_set_func=prediction_sets.raps_set_builder(
                lambd=lambd, k_reg=k_reg, rand=rand
            ),
            weight_func=None,
        )

        self.train = train

        self.random_state = random_state

        self.conformal_predictor = ConformalPredictor(
            predictor=self.predictor,
            calibrator=self.calibrator,
            splitter=object(),
            train=self.train,
        )

    def fit(
        self,
        *,
        X: Optional[Iterable] = None,
        y: Optional[Iterable] = None,
        fit_ratio: float = 0.8,
        X_fit: Optional[Iterable] = None,
        y_fit: Optional[Iterable] = None,
        X_calib: Optional[Iterable] = None,
        y_calib: Optional[Iterable] = None,
        **kwargs: Optional[dict],
    ):
        """This method fits the models on the fit data
        and computes nonconformity scores on calibration data.
        If (X, y) are provided, randomly split data into
        fit and calib subsets w.r.t to the fit_ratio.
        In case (X_fit, y_fit) and (X_calib, y_calib) are provided,
        the conformalization is performed on the given user defined
        fit and calibration sets.

        .. NOTE::

            If X and y are provided, `fit` ignores
            any user-defined fit/calib split.


        :param Iterable X: features from the training dataset.
        :param Iterable y: labels from the training dataset.
        :param float fit_ratio: the proportion of samples assigned to the
            fit subset.
        :param Iterable X_fit: features from the fit dataset.
        :param Iterable y_fit: labels from the fit dataset.
        :param Iterable X_calib: features from the calibration dataset.
        :param Iterable y_calib: labels from the calibration dataset.
        :param dict kwargs: predict configuration to be passed to the model's
            fit method.

        :raises RuntimeError: no dataset provided.

        """

        # Check if predictor is trained. Suppose that it is trained if the
        # predictor has not "is_trained" attribute
        is_trained = not hasattr(self.predictor, "is_trained") or (
            hasattr(self.predictor, "is_trained") and self.predictor.is_trained
        )

        if X is not None and y is not None:
            splitter = RandomSplitter(
                ratio=fit_ratio, random_state=self.random_state
            )

        elif (
            X_fit is not None
            and y_fit is not None
            and X_calib is not None
            and y_calib is not None
        ):
            splitter = IdSplitter(X_fit, y_fit, X_calib, y_calib)

        elif (
            is_trained
            and X_fit is None
            and y_fit is None
            and X_calib is not None
            and y_calib is not None
        ):
            splitter = IdSplitter(
                np.empty_like(X_calib), np.empty_like(y_calib), X_calib, y_calib
            )

        else:
            raise RuntimeError("No dataset provided.")

        # Update splitter
        self.conformal_predictor.splitter = splitter

        self.conformal_predictor.fit(X=X, y=y, **kwargs)

    def predict(self, X_test: Iterable, alpha: float) -> Tuple:
        """Conformal interval predictions (w.r.t target miscoverage alpha)
        for new samples.

        :param Iterable X_test: features of new samples.
        :param float alpha: target maximum miscoverage.

        :returns: Tuple composed of the model estimate y_pred and the
            prediction set set_pred
        :rtype: Tuple
        """

        if self.conformal_predictor is None:
            raise RuntimeError("Fit method should be called before predict.")

        (y_pred, set_pred) = self.conformal_predictor.predict(
            X_test, alpha=alpha
        )

        return y_pred, set_pred


class APS(RAPS):
    """Implementation of Adaptive Prediction Sets (APS).
    For more details, we refer the user to the
    :ref:`theory overview page <theory aps>`.

    :param BasePredictor predictor: a predictor implementing fit and predict.
    :param bool train: if False, prediction model(s) will not be trained and
        will be used as is. Defaults to True.
    :param bool rand: turn on or off randomization used in aps algorithm.
        One consequence of turning off randomization is avoiding empty
        prediction sets.

    .. _example aps:

    Example::

        from deel.puncc.classification import APS
        from deel.puncc.api.prediction import BasePredictor

        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        from deel.puncc.metrics import classification_mean_coverage
        from deel.puncc.metrics import classification_mean_size

        import numpy as np

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

        # Create a wrapper of the random forest model to redefine its predict method
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
        aps_cp = APS(rf_predictor)

        # The call to `fit` trains the model and computes the nonconformity
        # scores on the calibration set
        aps_cp.(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)

        # The predict method infers prediction intervals with respect to
        # the significance level alpha = 20%
        y_pred, set_pred = aps_cp.predict(X_test, alpha=.2)

        # Compute marginal coverage
        coverage = classification_mean_coverage(y_test, set_pred)
        size = classification_mean_size(set_pred)

        print(f"Marginal coverage: {np.round(coverage, 2)}")
        print(f"Average prediction set size: {np.round(size, 2)}")

    """

    def __init__(self, predictor, train=True, rand=True):
        super().__init__(predictor=predictor, train=train, lambd=0, rand=rand)

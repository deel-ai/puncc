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
from typing import Optional
from typing import Union


from deel.puncc.api import nonconformity_scores
from deel.puncc.api import prediction_sets
from deel.puncc.api.calibration import ClasswiseCalibrator
from deel.puncc.api.conformalization import SplitConformalPredictor
from deel.puncc.api.prediction import BasePredictor


class LAC(SplitConformalPredictor):
    """Implementation of the Least Ambiguous Set-Valued Classifier (LAC).
    For more details, we refer the user to the
    :ref:`theory overview page <theory lac>`.

    :param BasePredictor predictor: a predictor implementing fit and predict.
    :param bool train: if False, prediction model(s) will not be trained and
        will be used as is. Defaults to True.

    .. _example lac:

    Example::

        from deel.puncc.classification import LAC
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
        lac_cp = LAC(rf_predictor)

        # The call to `fit` trains the model and computes the nonconformity
        # scores on the calibration set
        lac_cp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)


        # The predict method infers prediction sets with respect to
        # the significance level alpha = 20%
        y_pred, set_pred = lac_cp.predict(X_test, alpha=.2)

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
        random_state: Optional[int] = None,
    ):
        super().__init__(
            predictor=predictor,
            nonconf_score_func=nonconformity_scores.lac_score,
            pred_set_func=prediction_sets.lac_set,
            train=train,
            random_state=random_state,
        )


class ClasswiseLAC(SplitConformalPredictor):
    """Implementation of the Classwise Least Ambiguous Set-Valued Classifier.

    Unlike standard LAC which computes a single quantile across all classes,
    ClasswiseLAC computes one quantile per class, providing class-conditional
    coverage guarantees.

    For more details on classwise conformal prediction, see:
    Ding et al. "Class-Conditional Conformal Prediction with Many Classes"
    https://arxiv.org/abs/2306.09335

    :param BasePredictor predictor: a predictor implementing fit and predict.
    :param bool train: if False, prediction model(s) will not be trained and
        will be used as is. Defaults to True.
    :param float random_state: random seed used when the user does not
        provide a custom fit/calibration split in `fit` method.

    Example::

        from deel.puncc.classification import ClasswiseLAC
        from deel.puncc.api.prediction import BasePredictor

        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        from deel.puncc.metrics import classification_mean_coverage
        from deel.puncc.metrics import classification_mean_size

        import numpy as np

        from tensorflow.keras.utils import to_categorical

        # Generate a random classification problem
        X, y = make_classification(n_samples=1000, n_features=4, n_informative=2,
                    n_classes=3, n_clusters_per_class=1, random_state=0, shuffle=False)

        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=0
        )

        # Split train data into fit and calibration
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train, y_train, test_size=.2, random_state=0
        )

        # Create rf classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=0)

        # Create a wrapper of the random forest model to redefine its predict method
        # into logits predictions. Make sure to subclass BasePredictor.
        class RFPredictor(BasePredictor):
            def predict(self, X, **kwargs):
                return self.model.predict_proba(X, **kwargs)

        # Wrap model in the newly created RFPredictor
        rf_predictor = RFPredictor(rf_model)

        # CP method initialization
        classwise_lac_cp = ClasswiseLAC(rf_predictor)

        # The call to `fit` trains the model and computes the nonconformity
        # scores on the calibration set
        classwise_lac_cp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)

        # The predict method infers prediction sets with respect to
        # the significance level alpha = 20%
        y_pred, set_pred = classwise_lac_cp.predict(X_test, alpha=.2)

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
        random_state: Optional[int] = None,
    ):
        super().__init__(
            predictor=predictor,
            nonconf_score_func=nonconformity_scores.classwise_lac_score,
            pred_set_func=prediction_sets.classwise_lac_set,
            train=train,
            random_state=random_state,
            CalibratorClass=ClasswiseCalibrator,
        )


class RAPS(SplitConformalPredictor):
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


        # The predict method infers prediction sets with respect to
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

        super().__init__(
            predictor=predictor,
            nonconf_score_func=nonconformity_scores.raps_score_builder(
                lambd=lambd, k_reg=k_reg, rand=rand
            ),
            pred_set_func=prediction_sets.raps_set_builder(
                lambd=lambd, k_reg=k_reg, rand=rand
            ),
            train=train,
            random_state=random_state,
        )


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
        aps_cp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_calib, y_calib=y_calib)

        # The predict method infers prediction sets with respect to
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

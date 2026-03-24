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
from deel.puncc.api.nonconformity_scores import lac_score, aps_score, raps_score
from deel.puncc.api.prediction_sets import lac_set, aps_set, raps_set
from deel.puncc.api.conformal_predictor import ClasswiseConformalPredictorMixin, ClassificationConformalPredictor, ConformalPredictor

class LAC(ClassificationConformalPredictor):
    nc_score_function=lac_score()
    pred_set_function=lac_set()

class ClasswiseLAC(ClasswiseConformalPredictorMixin, LAC):
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

        # One hot encoding of classes
        y_fit_cat = to_categorical(y_fit)
        y_calib_cat = to_categorical(y_calib)
        y_test_cat = to_categorical(y_test)

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

class APS(ClassificationConformalPredictor):
    nc_score_function = aps_score()
    pred_set_function = aps_set(rand=True)

class RAPS(ConformalPredictor):
    # TODO : add random state propagation to control randomized tie breaking
    def __init__(self, model, lambd:float=0, k_reg:int=1, rand:bool=False, weight_function=None, fit_function=None):
        nc_score_function = raps_score(lambd=lambd, k_reg=k_reg)
        pred_set_function = raps_set(lambd=lambd, k_reg=k_reg, rand=rand)
        super().__init__(model, nc_score_function, pred_set_function, weight_function=weight_function, fit_function=fit_function)

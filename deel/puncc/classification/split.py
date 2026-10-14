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
Basic components for split conformal classification
"""

from __future__ import annotations
from collections.abc import Iterable
from numbers import Real
from typing import Any, Self

from deel.puncc.core.calibration import CalibrationContext
from deel.puncc.core.conformal import ConformalPrediction
from deel.puncc.core.splitters import ClasswiseSplitter
from deel.puncc.typing import TensorLike
from deel.puncc.backend.keras import ops
from deel.puncc.nonconformity_scores import lac_score, aps_score, raps_score
from deel.puncc.prediction_sets import lac_set, aps_set, raps_set
from deel.puncc.core.split import (
    PresetSplitConformalPredictor,
    SplitConformalPredictor,
)

# class ClassificationSplitConformalPredictor(SplitConformalPredictor):
#     nc_score_function: ClassVar[NCScoreFunction]

#     def __init__(
#         self,
#         model: Predictor | PredictorLike,
#         *,
#         fit_function: FitFunction | None = None,
#     ) -> None:
#         super().__init__(
#             model=model,
#             nc_score_function=(type(self).nc_score_function),
#             pred_set_function=(self.pred_set_function),
#             fit_function=fit_function,
#         )

#     def pred_set_function(self, y_pred:TensorLike, quantile:float|TensorLike):
#         n_samples = len(y_pred)
#         n_classes = int(
#             ops.shape(y_pred)[1]
#         )
#         y_pred_tiled = ops.repeat(y_pred, repeats=n_classes, axis=0)
#         y_true_flat  = ops.tile(ops.arange(n_classes), (n_samples,))
#         scores_flat = self.nc_score_function(y_pred_tiled, y_true_flat)
#         scores = ops.reshape(scores_flat, (n_samples, n_classes))
#         mask = scores <= quantile
#         return [ops.where_1d(mask[i]) for i in range(n_samples)]


class ClassConditionalSplitConformalMixin(SplitConformalPredictor):
    """
    Mixin providing class-conditional calibration for split conformal classification methods.

    Calibration samples are partitioned according to their true class labels.
    A separate conformal quantile is computed for each class using only the corresponding calibration samples.

    During prediction, each candidate class is evaluated against its own calibration quantile rather than
    a global threshold.

    Under the usual exchangeability assumptions, this approach provides class-conditional coverage for
    classes represented in the calibration dataset.

    Notes:
        The underlying model must return class probabilities with shape (n_samples, n_classes), and calibration
        labels must be integer class indices corresponding to the probability columns.

        A scalar alpha applies the same miscoverage level to all classes.
        Alternatively, alpha may specify one miscoverage level per class.

        Classes absent from the calibration dataset are assigned an infinite quantile, so they are always included in prediction sets.

        The class-specific calibration contexts are rebuilt whenever calibrate() is called and cleared by reset_calibration().

    Example:
        ClassConditionalLAC combines this mixin with LAC to provide class-conditional prediction sets.
    """

    splitter = ClasswiseSplitter()

    def calibrate(
        self, X_calib: Iterable[Any], y_calib: Iterable[Any] | None = None
    ) -> Self:
        super().calibrate(X_calib, y_calib)
        self.classwise_calibration_contexts = self.splitter.split_context_by_group(
            self.calibration_context
        )
        return self

    def reset_calibration(self) -> Self:
        super().reset_calibration()
        self.classwise_calibration_contexts = {}
        return self

    def conformalize(
        self,
        prediction: Any,
        alpha: float | TensorLike,
        calibration_context: CalibrationContext,
        *,
        X: Any | None = None,
    ) -> ConformalPrediction[Any, Any]:
        if isinstance(alpha, Real):
            alpha_is_scalar = True
        else:
            alpha_array = ops.array(alpha)
            alpha_is_scalar = ops.ndim(alpha_array) == 0

        if calibration_context is self.calibration_context:
            classwise_contexts = self.classwise_calibration_contexts
        else:
            classwise_contexts = self.splitter.split_context_by_group(
                calibration_context
            )

        n_classes = int(ops.shape(prediction)[-1])
        quantiles = []

        for k in range(n_classes):
            alpha_k = alpha if alpha_is_scalar else alpha[k]
            class_context = classwise_contexts.get(k)

            if class_context is not None and class_context.size > 0:
                quantile_k = self._get_quantile(alpha_k, class_context, X=X)
            else:
                quantile_k = ops.array(float("inf"))
            quantiles.append(quantile_k)

        # For weighted class conditional with missing classes, we need to ensure that all quantiles have the same shape.
        non_scalar_quantile = next((q for q in quantiles if ops.ndim(q) > 0), None)
        if non_scalar_quantile is not None:
            quantiles = [
                (ops.ones_like(non_scalar_quantile) * q if ops.ndim(q) == 0 else q)
                for q in quantiles
            ]

        quantiles = ops.stack(quantiles, axis=-1)
        if ops.ndim(quantiles) == 1:
            quantiles = ops.expand_dims(quantiles, axis=0)

        y_set = self.pred_set_function(prediction, quantiles)
        return ConformalPrediction(prediction, y_set)


class LAC(PresetSplitConformalPredictor):
    """
    Implementation of the Least Ambiguous Set-Valued Classifier (LAC).

    This method constructs prediction sets using the nonconformity score 1 - p(y | x),
    where p(y | x) is the predicted probability of the true class.

    A single conformal quantile is computed from the calibration scores and used to
    select the classes included in each prediction set.

    Args:
        model: Predictive model returning class probabilities with shape (n_samples, n_classes).
        fit_function: Optional custom function used to train the model.
            Defaults to None.

    Notes:
        The true class labels must be integer indices corresponding to the columns of the predicted probability matrix.

        The predict() method returns a ConformalPrediction containing the class probabilities and a list of prediction
        sets, each containing the indices of the selected classes.

    Example::

        import numpy as np

        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        from deel.puncc.classification import LAC
        from deel.puncc.metrics import (
            classification_mean_coverage,
            classification_mean_size,
        )

        # Generate a random classification problem
        X, y = make_classification(
            n_samples=1000,
            n_features=4,
            n_informative=2,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=0,
            shuffle=False,
        )

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Split training data into fit and calibration sets
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0
        )

        # Create a random forest classifier
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=0,
        )

        # Adapt the classifier to return class probabilities
        class ProbabilityPredictor:
            def __init__(self, model):
                self.model = model

            def fit(self, X, y):
                self.model.fit(X, y)
                return self

            def __call__(self, X):
                return self.model.predict_proba(X)

        predictor = ProbabilityPredictor(rf_model)

        # Initialize the conformal predictor
        cp = LAC(model=predictor)

        # Fit the model and calibrate separately
        cp.fit(X_fit, y_fit)
        cp.calibrate(X_calib, y_calib)

        # Construct prediction sets for alpha = 20%
        y_pred, set_pred = cp.predict(X_test, alpha=0.2)

        # Compute marginal coverage and average prediction set size
        coverage = classification_mean_coverage(y_test, set_pred)
        size = classification_mean_size(set_pred)

        print(f"Marginal coverage: {np.round(coverage, 2)}")
        print(f"Average prediction set size: {np.round(size, 2)}")
    """

    nc_score_function = lac_score()
    pred_set_function = lac_set()


class ClassConditionalLAC(ClassConditionalSplitConformalMixin, LAC):
    """
    Implementation of the Class-Conditional Least Ambiguous Set-Valued Classifier.

    Unlike standard LAC, which computes a single quantile across all classes,
    this method computes one quantile per class to provide class-conditional
    coverage under the usual conformal prediction assumptions.

    Classes absent from the calibration set receive an infinite quantile
    and are therefore included in every prediction set.

    Args:
        model: A predictive model returning class probabilities of shape
            (n_samples, n_classes).
        fit_function: Optional function used to train the predictive model.
            Defaults to None.

    For more details on class-conditional conformal prediction, see:
    Ding et al. "Class-Conditional Conformal Prediction with Many Classes"
    https://arxiv.org/abs/2306.09335

    Example::

        import numpy as np

        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        from deel.puncc.classification import ClassConditionalLAC
        from deel.puncc.metrics import (
            classification_mean_coverage,
            classification_mean_size,
        )

        # Generate a random classification problem
        X, y = make_classification(
            n_samples=1000,
            n_features=4,
            n_informative=2,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=0,
            shuffle=False,
        )

        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Split training data into fit and calibration sets
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0
        )

        # Create a random forest classifier
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=0,
        )

        # Use class probabilities instead of predicted class labels
        class ProbabilityPredictor:
            def __init__(self, model):
                self.model = model

            def fit(self, X, y):
                self.model.fit(X, y)
                return self

            def __call__(self, X):
                return self.model.predict_proba(X)

        predictor = ProbabilityPredictor(rf_model)

        # Initialize the conformal predictor
        cp = ClassConditionalLAC(model=predictor)

        # Fit the predictive model and calibrate separately
        cp.fit(X_fit, y_fit)
        cp.calibrate(X_calib, y_calib)

        # Construct prediction sets for alpha = 20%
        y_pred, set_pred = cp.predict(X_test, alpha=0.2)

        # Compute marginal coverage and average prediction set size
        coverage = classification_mean_coverage(y_test, set_pred)
        size = classification_mean_size(set_pred)

        print(f"Marginal coverage: {np.round(coverage, 2)}")
        print(f"Average prediction set size: {np.round(size, 2)}")
    """


class APS(PresetSplitConformalPredictor):
    """
    Implementation of Adaptive Prediction Sets (APS).

    This method constructs prediction sets by sorting class probabilities in descending order and accumulating
    them until a conformal threshold is reached.

    Unlike LAC, which evaluates each class probability independently, APS uses cumulative class probabilities
    to adapt prediction set sizes to the uncertainty of the classification model.

    Randomization is enabled by default to determine whether the last class should be included in the prediction set.

    Args:
        model: Predictive model returning class probabilities with shape (n_samples, n_classes).
        fit_function: Optional custom function used to train the model.
            Defaults to None.

    Notes:
        True class labels must be integer indices corresponding to the columns of the predicted probability matrix.

        The predict() method returns a ConformalPrediction containing the class probabilities and a list of prediction sets, each
        containing the indices of the selected classes.

        Randomized tie-breaking is used when several classes have identical predicted probabilities.

    Example::
        import numpy as np

        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        from deel.puncc.classification import APS
        from deel.puncc.metrics import (
            classification_mean_coverage,
            classification_mean_size,
        )

        # Generate a random classification problem
        X, y = make_classification(
            n_samples=1000,
            n_features=4,
            n_informative=2,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=0,
            shuffle=False,
        )

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Split training data into fit and calibration sets
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0
        )

        # Create a random forest classifier
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=0,
        )

        # Adapt the classifier to return class probabilities
        class ProbabilityPredictor:
            def __init__(self, model):
                self.model = model

            def fit(self, X, y):
                self.model.fit(X, y)
                return self

            def __call__(self, X):
                return self.model.predict_proba(X)

        predictor = ProbabilityPredictor(rf_model)

        # Initialize the conformal predictor
        cp = APS(model=predictor)

        # Fit the model and calibrate separately
        cp.fit(X_fit, y_fit)
        cp.calibrate(X_calib, y_calib)

        # Construct adaptive prediction sets for alpha = 20%
        y_pred, set_pred = cp.predict(X_test, alpha=0.2)

        # Compute marginal coverage and average prediction set size
        coverage = classification_mean_coverage(y_test, set_pred)
        size = classification_mean_size(set_pred)

        print(f"Marginal coverage: {np.round(coverage, 2)}")
        print(f"Average prediction set size: {np.round(size, 2)}")
    """

    nc_score_function = aps_score()
    pred_set_function = aps_set(rand=True)


class RAPS(SplitConformalPredictor):
    """
    Implementation of Regularized Adaptive Prediction Sets (RAPS).

    RAPS extends Adaptive Prediction Sets (APS) by introducing a regularization penalty for classes whose rank exceeds k_reg.

    Classes are sorted by decreasing predicted probability, and their cumulative probabilities are adjusted using the penalty
    lambd * max(rank - k_reg, 0).

    Regularization can reduce prediction set sizes by discouraging the inclusion of low-ranked classes.

    Args:
        model: Predictive model returning class probabilities with shape (n_samples, n_classes).
        lambd: Non-negative regularization strength. Defaults to 0, corresponding to APS without regularization.
        k_reg: Non-negative rank threshold beyond which regularization is applied. Defaults to 1.
        rand: Whether to randomize the inclusion of the last class.
            Defaults to True.
        fit_function: Optional custom function used to train the model.
            Defaults to None.

    Raises:
        ValueError: If lambd or k_reg is negative.

    Notes:
        True class labels must be integer indices corresponding to the columns of the predicted probability matrix.

        The predict() method returns a ConformalPrediction containing the class probabilities and a list of prediction sets, each
        containing the indices of the selected classes.

        Randomized tie-breaking is performed when classes have identical predicted probabilities, independently of rand.

    Example::

        import numpy as np

        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        from deel.puncc.classification import RAPS
        from deel.puncc.metrics import (
            classification_mean_coverage,
            classification_mean_size,
        )

        # Generate a random classification problem
        X, y = make_classification(
            n_samples=1000,
            n_features=4,
            n_informative=2,
            n_classes=5,
            n_clusters_per_class=1,
            random_state=0,
            shuffle=False,
        )

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Split training data into fit and calibration sets
        X_fit, X_calib, y_fit, y_calib = train_test_split(
            X_train, y_train, test_size=0.2, random_state=0
        )

        # Create a random forest classifier
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=0,
        )

        # Adapt the classifier to return class probabilities
        class ProbabilityPredictor:
            def __init__(self, model):
                self.model = model

            def fit(self, X, y):
                self.model.fit(X, y)
                return self

            def __call__(self, X):
                return self.model.predict_proba(X)

        predictor = ProbabilityPredictor(rf_model)

        # Initialize the conformal predictor
        cp = RAPS(
            model=predictor,
            lambd=0.01,
            k_reg=2,
            rand=True,
        )

        # Fit the model and calibrate separately
        cp.fit(X_fit, y_fit)
        cp.calibrate(X_calib, y_calib)

        # Construct regularized prediction sets for alpha = 20%
        y_pred, set_pred = cp.predict(X_test, alpha=0.2)

        # Compute marginal coverage and average prediction set size
        coverage = classification_mean_coverage(y_test, set_pred)
        size = classification_mean_size(set_pred)

        print(f"Marginal coverage: {np.round(coverage, 2)}")
        print(f"Average prediction set size: {np.round(size, 2)}")
    """

    # TODO : add random state propagation to control randomized tie breaking
    def __init__(
        self,
        model,
        lambd: float = 0,
        k_reg: int = 1,
        rand: bool = True,
        fit_function=None,
    ):
        nc_score_function = raps_score(lambd=lambd, k_reg=k_reg, rand=rand)
        pred_set_function = raps_set(lambd=lambd, k_reg=k_reg, rand=rand)
        super().__init__(
            model,
            nc_score_function=nc_score_function,
            pred_set_function=pred_set_function,
            fit_function=fit_function,
        )

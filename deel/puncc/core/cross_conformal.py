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
Cross-conformal prediction methods.

This module provides the common machinery for fitting conformal predictors across multiple data splits and implements CV+ for scalar regression.
"""

from __future__ import annotations
from abc import abstractmethod, ABC
from collections.abc import Iterable
from typing import Any, Callable, Never
from typing_extensions import Self

from deel.puncc.core.predictors import make_predictor
from deel.puncc.core.split import SplitConformalPredictor
from deel.puncc.core.conformal import ConformalPrediction
from deel.puncc.core.splitters import KFoldSplitter, BaseSplitter
from deel.puncc.corrections import AlphaCorrection
from deel.puncc.typing import Predictor, PredictorLike, TensorLike
from deel.puncc.cloning import clone_model
from deel.puncc.regression.split import SplitConformalRegression
from deel.puncc import ops
from deel.puncc.typing import (
    FitFunction,
    Predictor,
    PredictorLike,
    TensorLike,
)

class CrossConformalPredictor(ABC):
    """
    Base class for cross-conformal prediction methods.

    Cross-conformal predictors train several conformal predictors on different data splits and aggregate their predictions at inference time.
    Unlike split conformal predictors, calibration is performed internally during fitting and no separate calibration step is required.

    Args:
        model: Predictive model used as a template for each data split.
        conformal_predictor_class: Split conformal predictor class instantiated independently on each split.
        splitter: Data splitter defining the fitting and calibration subsets.
        fit_function: Optional custom function used to fit each cloned model.
    """
    def __init__(self,
                 model:Predictor|PredictorLike,
                 conformal_predictor_class:type[SplitConformalPredictor],
                 splitter:BaseSplitter,
                 fit_function:FitFunction|None = None):
        # TODO : implement WCV+
        self.model = make_predictor(model)
        self.fit_function = fit_function
        self.splitter = splitter

        self._conformal_predictors: list[SplitConformalPredictor] = []
        self.conformal_predictor_class = conformal_predictor_class

    @property
    def len_calibr(self)->int:
        """
        Total number of calibration samples across all fitted splits.
        """
        return sum(cp.len_calibr for cp in self._conformal_predictors)

    def calibrate(self, X_calib:Iterable[Any], y_calib:TensorLike)->Never:
        """
        Raises:
            RuntimeError: the calibration step is not required for cross-conformal predictors, only the `fit` method should be used to train and calibrate the model.
        """
        raise RuntimeError("Cross-conformal predictors do not require a separate calibration step. Please use the `fit` method to train and calibrate the model.")

    def fit(self, X:Iterable[Any], y:TensorLike)->Self:
        """
        Fit and calibrate conformal predictors across all data splits.

        For each split, the base predictive model is cloned, fitted on the training subset, and calibrated on the corresponding calibration subset.

        Args:
            X: Input samples.
            y: Target values.

        Returns:
            The fitted cross-conformal predictor.
        """
        self._conformal_predictors.clear()

        for ((X_fit, y_fit),(X_calib, y_calib)) in self.splitter(X=X, y=y):
            cp = self.conformal_predictor_class(clone_model(self.model), fit_function=self.fit_function)
            cp.fit(X_fit, y_fit)
            cp.calibrate(X_calib, y_calib)
            self._conformal_predictors.append(cp)
        return self
    
    @abstractmethod
    def predict(self,
                X_test:Iterable[Any],
                alpha:float|TensorLike,
                *,
                alpha_correction: AlphaCorrection | None = None,)->ConformalPrediction[Any, Any]:
        """
        Produce an aggregated cross-conformal prediction.

        Args:
            X_test: Input samples on which predictions are produced.
            alpha: Requested scalar miscoverage level. May be provided as a Python float or a scalar tensor.
            alpha_correction: Optional correction applied to the miscoverage level before conformalization.

        Returns:
            Aggregated point predictions and conformal prediction sets.
        """
        ...

class CVPlusRegressor(CrossConformalPredictor):
    """
    CV+ conformal predictor for scalar regression.

    The dataset is partitioned into K folds.
    For each fold, a predictive model is trained on the remaining folds and calibrated on the held-out fold.
    Prediction intervals are then obtained from the pooled lower and upper CV+ candidates across all calibration samples.

    Args:
        model: Underlying regression model.
        K: Number of folds.
        random_state: Random seed controlling fold generation.
        fit_function: Optional custom function used to fit each cloned model.
    """
    def __init__(self,
                 model:Predictor|PredictorLike,
                 K:int=5,
                 random_state:int|None=None,
                 fit_function:Callable[[Predictor, Iterable[Any], TensorLike], Predictor]|None = None):
        super().__init__(model,
                         splitter = KFoldSplitter(K=K, shuffle=True, random_state=random_state),
                         conformal_predictor_class=SplitConformalRegression,
                         fit_function = fit_function
                         )

    # TODO : see what can be moved to the parent class Here
    def predict(self, X_test:Iterable[Any], alpha:float|TensorLike, *, alpha_correction:AlphaCorrection|None = None)->ConformalPrediction[Any, Any]:
        """
        Compute CV+ prediction intervals.

        Predictions from each fold-specific model are combined with their calibration nonconformity scores to form lower and upper CV+ candidates.
        The corresponding empirical order statistics define the prediction interval.

        The point prediction returned alongside the interval is the mean prediction across the fold-specific models.

        Args:
            X_test: Input samples on which predictions are produced.
            alpha: Requested scalar miscoverage level. May be provided as a Python float or a scalar tensor.
            alpha_correction: Optional correction applied to the miscoverage level before interval construction.

        Returns:
            Mean point predictions and their associated CV+ prediction intervals.

        Raises:
            RuntimeError: If the predictor has not been fitted.
        """
        if not self._conformal_predictors:
            raise RuntimeError("CVPlusRegressor must be fitted before prediction.")

        n = self.len_calibr

        if alpha_correction is not None:
            alpha = alpha_correction(alpha)

        predictions: list[TensorLike] = []
        lower_candidates: list[TensorLike] = []
        upper_candidates: list[TensorLike] = []

        for cp in self._conformal_predictors:
            prediction = cp.model(X_test)
            predictions.append(prediction)
            scores = ops.reshape(cp.nc_scores, (-1,))
            prediction = ops.expand_dims(prediction,axis=0)
            scores = ops.expand_dims(scores,axis=1)
            lower_candidates.append(prediction - scores)
            upper_candidates .append(prediction + scores)

        lower_candidates = ops.sort(ops.concatenate(lower_candidates, axis=0), axis=0)
        upper_candidates = ops.sort(ops.concatenate(upper_candidates, axis=0), axis=0)

        # TODO : revoir les formules des indices ici
        l_alpha = lower_candidates[ops.cast(ops.floor(alpha * (n+1)) - 1, int)]
        u_alpha = upper_candidates[ops.cast(ops.ceil((1 - alpha) * (n+1)) - 1, int)]

        # TODO : See if mean is the best aggregation here
        point_prediction = ops.mean(
            ops.stack(predictions, axis=0),
            axis=0,
        )
        return ConformalPrediction(point_prediction, ops.stack([l_alpha, u_alpha], axis=-1))

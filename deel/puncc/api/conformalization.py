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
Abstract base classes for conformal prediction methods
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Generic, Iterator, TypeVar, overload
from typing_extensions import Self
from abc import ABC, abstractmethod
from collections.abc import Iterable
from deel.puncc.typing import Predictor, PredictorLike, make_predictor, TensorLike


# To be defined more precisely in the future.
TSet = TypeVar("TSet")

@dataclass(frozen=True, slots=True)
class ConformalPrediction(Generic[TSet]):
    """
    Container for a conformal prediction result.

    Attributes:
        prediction (TensorLike):
            The base (non-conformal) prediction of the underlying model.

        prediction_set (TSet):
            The conformal prediction set (interval, label set, bounding boxes, etc.).
    """

    prediction: TensorLike
    prediction_set: TSet

    def __iter__(self) -> Iterator[TensorLike | TSet]:
        yield self.prediction
        yield self.prediction_set

    def __len__(self) -> int:
        return 2

    @overload
    def __getitem__(self, index: int) -> TensorLike | TSet: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[TensorLike | TSet, ...]: ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return (self.prediction, self.prediction_set)[index]
        if index == 0:
            return self.prediction
        if index == 1:
            return self.prediction_set
        raise IndexError("ConformalPrediction only contains two elements.")

class ConformalMethod(ABC):
    """
    Asbtract base class for conformal prediction methods.
    Any conformal prediction method should inherit from this class and implement the `calibrate` and `predict` methods.
    """

    # Any conformal method should have a model attribute.
    __slots__ = ("model",)
    def __init__(self, model:Predictor|PredictorLike):
        self.model = make_predictor(model)

    @abstractmethod
    def calibrate(self, X_calib:Iterable[Any], y_calib:TensorLike)->Self:
        """
        Calibration step of the conformal method.
       
        Args:
            X_calib (Iterable[Any]): Features of calibration dataset
            y_calib (TensorLike): Labels of calibration dataset
        """
        ...

    @abstractmethod
    def predict(self, X_test:Iterable[Any], alpha:float|TensorLike)->ConformalPrediction:
        """
        Perform a conformal prediction using the calibrated model.

        Args:
            X_test (Iterable[Any]): Features to perform the conformal prediction on
            alpha (float | TensorLike): Miscoverage level(s) for the conformal prediction. Can be a single float or a tensor of floats of the same length as X_test.

        Returns:
            ConformalPrediction: A container for the conformal prediction result, containing the base (non-conformal) prediction and the conformal prediction set.
        """
        ...


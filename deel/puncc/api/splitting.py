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
Data splitting schemes.

This module defines splitters that partition a dataset into:
- a training subset used to fit the base model
- a calibration subset used for calibration phase.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias

from deel.puncc import ops
from deel.puncc._keras import random

from deel.puncc.api.utils import features_len_check
from deel.puncc.api.utils import sample_len_check
from deel.puncc.typing import TensorLike


### TODO : deal with pandas series
# if importlib.util.find_spec("pandas") is not None:
#     import pandas as pd

Split: TypeAlias = list[tuple[TensorLike, TensorLike, TensorLike, TensorLike]]
IndexTensor: TypeAlias = TensorLike

def _take(X:TensorLike, y:TensorLike, fit_idxs:IndexTensor, cal_idxs:IndexTensor)->tuple[TensorLike, TensorLike, TensorLike, TensorLike]:
    """
    Materialize one split from index tensors.

    Args:
        X (TensorLike), : Input features
        y (TensorLike): Labels
        fit_idxs (IndexTensor): 1D integer index tensors specifying rows/items to pick for train dataset
        cal_idxs (IndexTensor): 1D integer index tensors specifying rows/items to pick for calibration dataset

    Returns:
        tuple[TensorLike, TensorLike, TensorLike, TensorLike]: subsets for train and calibration : X_train, y_train, X_calib, y_calib
    """
    return (ops.take(X, fit_idxs, axis=0),
                ops.take(y, fit_idxs, axis=0),
                ops.take(X, cal_idxs, axis=0),
                ops.take(y, cal_idxs, axis=0))


class BaseSplitter(ABC):
    """
    Base class for data splitters.

    A splitter partitions a dataset (X, y) into one or more folds.
    Each fold is represented as a tuple:
        (X_train, y_train, X_calib, y_calib)

    Args:
        random_state (int | None, optional): Optional seed controlling random operations.
            Defaults to None.
    """
    def __init__(self, random_state:int|None=None) -> None:
        self.random_state = random_state

    @abstractmethod
    def split(self, X:TensorLike, y:TensorLike)->Split:
        ...

    def __call__(self, *args, **kwargs) -> Split:
        return self.split(*args, **kwargs)


class IdSplitter(BaseSplitter):
    """
    Identity splitter.

    This splitter does not compute a split. It simply wraps already-defined
    training and calibration subsets.

    Args:
        X_fit (TensorLike): Training features.
        y_fit (TensorLike): Training labels.
        X_calib (TensorLike): Calibration features.
        y_calib (TensorLike): Calibration labels.
    """
    def __init__(
        self,
        X_fit: TensorLike,
        y_fit: TensorLike,
        X_calib: TensorLike,
        y_calib: TensorLike,
    ):
        super().__init__(random_state=None)

        # Checks
        sample_len_check(X_fit, y_fit)
        sample_len_check(X_calib, y_calib)
        features_len_check(X_fit, X_calib)

        self._split = [(X_fit, y_fit, X_calib, y_calib)]

    def split(self, X:TensorLike|None=None, y:TensorLike|None=None) -> Split:
        """
        Return the stored training and calibration subsets.

        Args:
            X (TensorLike | None): Unused. Present for API compatibility.
            y (TensorLike | None): Unused. Present for API compatibility.

        Returns:
            Split: List of one tuple of deterministic subsets
                (X_train, y_train, X_calib, y_calib).
        """
        return self._split


class RandomSplitter(BaseSplitter):
    """
    Random train/calibration splitter.

    Each sample is independently assigned to:
    - the training set with probability ratio,
    - the calibration set with probability 1 - ratio.

    Args:
        ratio (float): Fraction of samples assigned to the training set.
            Must be strictly between 0 and 1.
        random_state (int | None): Optional seed controlling random operations.
    """

    def __init__(self, ratio: float, random_state: int | None = None):
        if (ratio <= 0) or (ratio >= 1):
            raise ValueError(f"Ratio must be in ]0,1[. Provided value: {ratio}")
        super().__init__(random_state=random_state)
        self.ratio = ratio

    def split(
        self,
        X: TensorLike,
        y: TensorLike,
    ) -> Split:
        """
        Split the dataset randomly into training and calibration subsets.

        Args:
            X (TensorLike): Input features.
            y (TensorLike): Labels.

        Returns:
            Split: A single-element list containing
                (X_train, y_train, X_calib, y_calib).
        """
        # Checks
        sample_len_check(X, y)


        u = random.uniform((len(X),), seed=self.random_state)

        fit_mask = ops.less(u, self.ratio)
        cal_mask = ops.logical_not(fit_mask)

        fit_idxs = ops.where_1d(fit_mask)
        cal_idxs = ops.where_1d(cal_mask)

        return [_take(X, y, fit_idxs, cal_idxs)]


class KFoldSplitter(BaseSplitter):
    """
    K-fold splitter.

    The dataset is partitioned into K folds. For each fold:
        - the calibration subset is one fold,
        - the training subset is the union of the remaining K-1 folds.

    Args:
        K (int): Number of folds (must be >= 2).
        shuffle (bool): Whether to shuffle samples before creating folds.
        random_state (int | None): Optional seed controlling random operations.
    """

    def __init__(self, K: int,
                 shuffle:bool=True,
                 random_state:int|None=None) -> None:
        if K < 2:
            raise ValueError(f"K must be >= 2. Provided value: {K}.")
        super().__init__(random_state=random_state)
        self.K = K
        self.shuffle = shuffle

    def split(
        self,
        X: TensorLike,
        y: TensorLike,
    ) -> Split:
        """
        Split the dataset into K training/calibration folds.

        Args:
            X (TensorLike): Input features.
            y (TensorLike): Labels.

        Returns:
            Split: A list of K tuples, each tuple being
                (X_train, y_train, X_calib, y_calib).
        """
        # Checks
        sample_len_check(X, y)

        n_samples = len(X)
        idxs = ops.arange(n_samples)

        if self.shuffle:
            idxs = random.shuffle(idxs, axis=0, seed=self.random_state)

        n_min = n_samples // self.K
        r = n_samples % self.K
        fold_sizes = [n_min + 1] * r + [n_min] * (self.K - r)

        folds:Split = []
        start = 0
        for size in fold_sizes:
            calib_idx = idxs[start : start + size]
            fit_idx = ops.concatenate([idxs[:start], idxs[start + size :]], axis=0)
            folds.append(_take(X, y, fit_idx, calib_idx))
            start += size
        return folds

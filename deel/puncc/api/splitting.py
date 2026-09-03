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
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

from deel.puncc import ops
from deel.puncc.api.calibration_context import CalibrationContext
from deel.puncc.keras import random

from deel.puncc.typing import TensorLike


### TODO : deal with pandas series
# if importlib.util.find_spec("pandas") is not None:
#     import pandas as pd

DatasetGroup: TypeAlias = tuple[Any, ...]
Split: TypeAlias = tuple[DatasetGroup, ...]
Splits: TypeAlias = list[Split]
IndexTensor: TypeAlias = TensorLike
GroupFunction = Callable[..., TensorLike]


FitCalSplit: TypeAlias = tuple[
    DatasetGroup,
    DatasetGroup,
]

FitCalSplits: TypeAlias = list[
    FitCalSplit
]

def _take(
    datasets: dict[str, Any],
    *group_idxs: IndexTensor,
) -> Split:
    return tuple(
        tuple(
            dataset[idxs]
            for dataset in datasets.values()
        )
        for idxs in group_idxs
    )

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
    def split(self, **datasets:Any)->Splits:
        ...

    def __call__(self, **datasets: Any) -> Splits:
        return self.split(**datasets)


    def split_context(
        self,
        context: CalibrationContext,
    ) -> list[tuple[CalibrationContext, ...]]:
        return [
            tuple(
                context.from_values(group)
                for group in split
            )
            for split in self.split(
                **dict(context.items())
            )
        ]
    
class FunctionalSplitter(BaseSplitter):
    def __init__(
        self,
        group_function: GroupFunction,
        groups:Sequence[Any]|None = None,
    ) -> None:
        super().__init__(random_state=None)
        self.group_function = group_function
        self.groups = groups

    def group_indices(
        self,
        **datasets: Any,
    ) -> list[tuple[Any, IndexTensor]]:
        group_ids = ops.reshape(
            self.group_function(**datasets),
            (-1,),
        )

        groups = self.groups

        if groups is None:
            groups = dict.fromkeys(
                ops.convert_to_numpy(
                    group_ids
                ).tolist()
            )

        return [
            (
                group,
                ops.where_1d(
                    group_ids == group
                ),
            )
            for group in groups
        ]

    def split(
        self,
        **datasets: Any,
    ) -> Splits:
        grouped_indices = self.group_indices(
            **datasets
        )
        return [
            _take(
                datasets,
                *(
                    indices
                    for _, indices
                    in grouped_indices
                ),
            )
        ]
    
    def split_context_by_group(
        self,
        context: CalibrationContext,
    ) -> dict[Any, CalibrationContext]:
        grouped_indices = self.group_indices(
            **dict(context.items())
        )

        return {
            group: context[indices]
            for group, indices
            in grouped_indices
        }

class ClasswiseSplitter(FunctionalSplitter):
    def __init__(
        self,
        classes: Sequence[int]|None=None,
    ) -> None:
        super().__init__(
            group_function=lambda y_calib, **_: y_calib,
            groups=classes,
        )

class FitCalSplitter(BaseSplitter):
    @abstractmethod
    def split(
        self,
        **datasets: Any,
    ) -> FitCalSplits:
        ...

#TODO : refaire le IDSplitter
class IdSplitter(FitCalSplitter):
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
        X_fit:Sequence[Any],
        y_fit:Sequence[Any],
        X_calib:Sequence[Any],
        y_calib:Sequence[Any],
    ):
        super().__init__(random_state=None)

        # TODO : Check again
        #sample_len_check(X_fit, y_fit)
        #sample_len_check(X_calib, y_calib)
        #features_len_check(X_fit, X_calib)

        self._split = [
            (
                (X_fit, y_fit),
                (X_calib, y_calib),
            )
        ]

    def split(self, **datasets:Any) -> FitCalSplits:
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


class RandomSplitter(FitCalSplitter):
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
        **datasets: Any,
    ) -> FitCalSplits:
        """
        Split the dataset randomly into training and calibration subsets.

        Args:
            X (TensorLike): Input features.
            y (TensorLike): Labels.

        Returns:
            Split: A single-element list containing
                (X_train, y_train, X_calib, y_calib).
        """
        # TODO : checks length of datasets

        n_samples = len(
            next(iter(datasets.values()))
        )

        if n_samples < 2:
            raise ValueError(
                "RandomSplitter requires at least 2 samples."
            )


        idxs = ops.arange(n_samples)
        idxs = random.shuffle(
            idxs,
            axis=0,
            seed=self.random_state,
        )

        n_fit = int(self.ratio * n_samples)
        n_fit = max(1, min(n_fit, n_samples - 1))

        fit_idxs = idxs[:n_fit]
        cal_idxs = idxs[n_fit:]

        return [
            _take(
                datasets,
                fit_idxs,
                cal_idxs,
            )
        ]

class KFoldSplitter(FitCalSplitter):
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
        **datasets:Any
    ) -> FitCalSplits:
        """
        Split the dataset into K training/calibration folds.

        Args:
            X (TensorLike): Input features.
            y (TensorLike): Labels.

        Returns:
            Split: A list of K tuples, each tuple being
                (X_train, y_train, X_calib, y_calib).
        """
        # TODO : checks
        # sample_len_check(X, y)

        n_samples = len(
            next(iter(datasets.values()))
        )

        if self.K > n_samples:
            raise ValueError(f"K must be <= number of samples. Provided K: {self.K}, number of samples: {n_samples}.")

        idxs = ops.arange(n_samples)

        if self.shuffle:
            idxs = random.shuffle(idxs, axis=0, seed=self.random_state)

        n_min = n_samples // self.K
        r = n_samples % self.K
        fold_sizes = [n_min + 1] * r + [n_min] * (self.K - r)

        folds: Splits = []

        start = 0
        for size in fold_sizes:
            calib_idx = idxs[start : start + size]
            fit_idx = ops.concatenate([idxs[:start], idxs[start + size :]], axis=0)
            folds.append(_take(datasets, fit_idx, calib_idx))
            start += size
        return folds

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
Data splitting utilities for conformal prediction.

This module provides splitters for partitioning sample-aligned datasets into subsets.
Supported strategies include predefined splits, random train/calibration splits, K-fold
cross-validation, and group-based  partitioning.

Splitters preserve the ordering of dataset fields within each subset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, cast

from deel.puncc import ops
from deel.puncc.core.calibration import CalibrationContext
from deel.puncc.backend.keras import random

from deel.puncc.typing import TensorLike

DatasetGroup: TypeAlias = tuple[Any, ...]
Split: TypeAlias = tuple[DatasetGroup, ...]
Splits: TypeAlias = list[Split]
IndexTensor: TypeAlias = TensorLike
GroupFunction = Callable[..., TensorLike]


FitCalSplit: TypeAlias = tuple[
    DatasetGroup,
    DatasetGroup,
]

FitCalSplits: TypeAlias = list[FitCalSplit]


def tensor_indexing(item: Any, indices: IndexTensor) -> Any:
    """
    Select samples from a dataset using integer indices.

    Tensor inputs are indexed using the active numerical backend.
    Python lists and tuples preserve their container type.
    Other indexable objects use their native indexing implementation.

    Args:
        item:
            Dataset from which samples are selected.
        indices:
            One-dimensional tensor of sample indices.

    Returns:
        The selected samples in the dataset's corresponding representation.
    """

    if isinstance(item, ops.tensor_type):
        return ops.take(item, indices, axis=0)
    idx = ops.tolist(indices)
    if isinstance(item, list):
        return [item[i] for i in idx]
    elif isinstance(item, tuple):
        return tuple(item[i] for i in idx)
    return item[indices]


def _take(
    datasets: dict[str, Any],
    *group_idxs: IndexTensor,
) -> Split:
    """
    Extract several aligned dataset groups from sample indices.

    Args:
        datasets:
            Dictionary of sample-aligned datasets.
        *group_idxs:
            Index tensors defining the requested groups.

    Returns:
        A tuple containing one dataset group per index tensor.
        Each group preserves the order of the input datasets.
    """
    return tuple(
        tuple(tensor_indexing(dataset, idxs) for dataset in datasets.values())
        for idxs in group_idxs
    )


def _sample_count(datasets: dict[str, Any]) -> int:
    """Validate sample counts and return the common length."""
    if not datasets:
        raise ValueError("At least one dataset must be provided.")

    lengths = {name: len(data) for name, data in datasets.items()}

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"All datasets must have the same number of samples: {lengths}"
        )

    return next(iter(lengths.values()))


class BaseSplitter(ABC):
    """
    Abstract base class for data splitters.

    A splitter accepts one or more sample-aligned datasets through
    keyword arguments and returns a list of splits.

    Each split contains one or more groups of datasets. Within each
    group, datasets appear in the same order as the input arguments.

    For train/calibration splitters, each split has the structure::

        ((X_fit, y_fit), (X_calib, y_calib))

    Args:
        random_state (int | None, optional): Optional seed controlling random operations.
            Defaults to None.
    """

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state

    @abstractmethod
    def split(self, **datasets: Any) -> Splits:
        """
        Partition sample-aligned datasets into groups.
        """

    def __call__(self, **datasets: Any) -> Splits:
        return self.split(**datasets)

    def split_context(
        self,
        context: CalibrationContext,
    ) -> list[tuple[CalibrationContext, ...]]:
        """
        Split a calibration context into sample-aligned subcontexts.

        The stored fields are passed to the splitter using their names.
        Each resulting dataset group is converted into a new CalibrationContext.

        Args:
            context:
                Calibration context containing sample-aligned fields.

        Returns:
            A list of splits, each represented as a tuple of CalibrationContext instances.
        """
        return [
            tuple(context.from_values(group) for group in split)
            for split in self.split(**dict(context.items()))
        ]


class FunctionalSplitter(BaseSplitter):
    """
    Split datasets according to sample group identifiers.

    A user-provided function assigns a group identifier to each sample.
    Samples sharing the same identifier are collected into the same subset.

    If explicit groups are provided, they appear first in the specified
    order, followed by any additional groups observed in the data.
    Otherwise, groups follow their order of first occurrence.

    Args:
        group_function:
            Function accepting the datasets as keyword arguments and
            returning one group identifier per sample.
        groups:
            Optional sequence of group identifiers to include.
            Groups without observations are represented by empty subsets.
            Defaults to None.
    """

    def __init__(
        self,
        group_function: GroupFunction,
        groups: Sequence[Any] | None = None,
    ) -> None:
        super().__init__(random_state=None)
        self.group_function = group_function
        self.groups = groups

    def group_indices(self, **datasets: Any) -> dict[Any, IndexTensor]:
        """
        Compute the sample indices associated with each group.

        Args:
            **datasets:
                Sample-aligned datasets forwarded to the group function.

        Returns:
            A dictionary mapping group identifiers to one-dimensional tensors containing their sample indices.

        Notes:
            Explicitly configured groups are included even when they have no observations.
            Additional observed groups are also included.
        """
        group_ids = ops.reshape(self.group_function(**datasets), (-1,))
        observed_groups = list(dict.fromkeys(ops.tolist(group_ids)))
        if self.groups is None:
            groups = observed_groups
        else:
            groups = list(dict.fromkeys([*self.groups, *observed_groups]))
        return {group: ops.where_1d(group_ids == group) for group in groups}

    def split(self, **datasets: Any) -> Splits:
        """
        Partition datasets into groups.

        Args:
            **datasets:
                Sample-aligned datasets to partition.

        Returns:
            A single-element list containing one tuple of dataset groups.
            Each group contains the subsets corresponding to its sample indices, preserving the
            order of the input datasets.
        """
        grouped_indices = self.group_indices(**datasets)
        return [_take(datasets, *grouped_indices.values())]

    def split_context_by_group(
        self, context: CalibrationContext
    ) -> dict[Any, CalibrationContext]:
        """
        Partition a calibration context by group identifier.

        Args:
            context:
                Calibration context containing sample-aligned fields.

        Returns:
            A dictionary mapping each group identifier to its corresponding
            CalibrationContext.

        Notes:
            Groups without observations are associated with empty
            calibration contexts.
        """
        grouped_indices = self.group_indices(**dict(context.items()))

        return {group: context[indices] for group, indices in grouped_indices.items()}


class ClasswiseSplitter(FunctionalSplitter):
    """
    Split calibration data according to their true class labels.

    Group identifiers are obtained from the ``y_calib`` field.
    This splitter is intended for class-conditional conformal calibration.

    Args:
        classes:
            Optional sequence of class labels to include.
            Unobserved classes are represented by empty subsets.
            Defaults to None.
    """

    def __init__(
        self,
        classes: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            group_function=lambda y_calib, **_: y_calib,
            groups=classes,
        )


class FitCalSplitter(BaseSplitter):
    """
    Abstract base class for train/calibration splitting strategies.

    Each split contains exactly two dataset groups: a fitting subset and a calibration subset.

    The returned structure is::

        [
            ((X_fit, y_fit), (X_calib, y_calib)),
            ...
        ]

    Subclasses determine how samples are assigned to each subset.
    """

    @abstractmethod
    def split(
        self,
        **datasets: Any,
    ) -> FitCalSplits: ...


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
        X_fit: Sequence[Any],
        y_fit: Sequence[Any],
        X_calib: Sequence[Any],
        y_calib: Sequence[Any],
    ):
        super().__init__(random_state=None)
        _sample_count({"X_fit": X_fit, "y_fit": y_fit})
        _sample_count({"X_calib": X_calib, "y_calib": y_calib})

        self._split = [
            (
                (X_fit, y_fit),
                (X_calib, y_calib),
            )
        ]

    def split(self, **datasets: Any) -> FitCalSplits:
        """
        Return the predefined fitting and calibration subsets.

        Args:
            **datasets:
                Unused keyword arguments, accepted for compatibility with the splitter interface.

        Returns:
            A single-element list containing the stored fitting and calibration dataset groups.

        Notes:
            No new split is computed.
            The predefined datasets are returned by reference.
        """
        return self._split


class RandomSplitter(FitCalSplitter):
    """
    Random train/calibration splitter.

    Samples are randomly permuted, then the first ratio fraction is assigned to the fitting subset and the
    remainder to calibration.

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
            **datasets:
                Sample-aligned datasets to split.

        Returns:
            Split: A single-element list containing
                (X_train, y_train, X_calib, y_calib).
        """
        n_samples = _sample_count(datasets)

        if n_samples < 2:
            raise ValueError("RandomSplitter requires at least 2 samples.")

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
            cast(
                FitCalSplit,
                _take(
                    datasets,
                    fit_idxs,
                    cal_idxs,
                ),
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

    def __init__(
        self, K: int, shuffle: bool = True, random_state: int | None = None
    ) -> None:
        if K < 2:
            raise ValueError(f"K must be >= 2. Provided value: {K}.")
        super().__init__(random_state=random_state)
        self.K = K
        self.shuffle = shuffle

    def split(self, **datasets: Any) -> FitCalSplits:
        """
        Partition datasets into K fitting/calibration folds.

        Each sample belongs to exactly one calibration fold and to the fitting subsets of all remaining folds.

        Args:
            **datasets:
                Sample-aligned datasets to split.

        Returns:
            A list of K splits, each containing a fitting dataset group and a calibration dataset group.

        Notes:
            Folds are as evenly sized as possible.
            When the number of samples is not divisible by K, the first folds contain one additional calibration sample.

            If shuffle is True, a single permutation is applied before constructing the folds.
        """
        n_samples = _sample_count(datasets)

        if self.K > n_samples:
            raise ValueError(
                f"K must be <= number of samples. Provided K: {self.K}, number of samples: {n_samples}."
            )

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
        return cast(FitCalSplits, folds)

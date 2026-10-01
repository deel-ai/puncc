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
Bootstrap sampling schemes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Integral
from typing import TypeAlias

from deel.puncc import ops
from deel.puncc.backend.keras import random
from deel.puncc.typing import TensorLike

IndexTensor: TypeAlias = TensorLike


@dataclass(frozen=True, slots=True)
class BootstrapSample:
    """
    Indices associated with one bootstrap sample.

    Attributes:
        train_indices (IndexTensor): One-dimensional integer tensor of sampled
            indices. Repetitions and sampling order are preserved.
        oob_indices (IndexTensor): Sorted, unique indices absent from the
            training sample. May be empty.

    Note:
        Tensors are stored by reference. Freezing the container does not make
        the tensors themselves immutable.
    """

    train_indices: IndexTensor
    oob_indices: IndexTensor


class BootstrapSampler(ABC):
    """
    Base class for bootstrap index samplers.

    Each sample contains as many training indices as the original dataset.
    Samples are generated lazily, without selecting or copying data.

    Args:
        random_state (int | None): Optional seed controlling random operations.

    Note:
        A fixed seed reproduces the sequence on each call to ``sample`` within
        the same backend. Different backends may produce different sequences.
        Select the PUNCC backend before sampling, since indices alone do not
        allow backend inference. Sampling runs eagerly.
    """

    def __init__(self, random_state:int|None=None) -> None:
        self.random_state = random_state

    @abstractmethod
    def sample(
        self,
        n_samples:int,
        n_resamples:int,
    ) -> Iterator[BootstrapSample]:
        """
        Generate bootstrap samples of the indices ``0, ..., n_samples - 1``.

        Args:
            n_samples (int): Number of observations. Must be positive.
            n_resamples (int): Number of bootstrap samples. Must be positive.

        Yields:
            BootstrapSample: Training and out-of-bag indices.
        """
        ...

    def __call__(
        self,
        n_samples:int,
        n_resamples:int,
    ) -> Iterator[BootstrapSample]:
        return self.sample(n_samples, n_resamples)

    @staticmethod
    def _check_positive_integer(value:int, name:str) -> None:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer. Provided value: {value}.")

    @staticmethod
    def _make_sample(
        train_indices:IndexTensor,
        n_samples:int,
    ) -> BootstrapSample:
        # Counts avoid allocating a pairwise membership comparison matrix.
        counts = ops.bincount(train_indices, minlength=n_samples)
        oob_indices = ops.where_1d(counts == 0)
        return BootstrapSample(train_indices, oob_indices)


class IIDBootstrapSampler(BootstrapSampler):
    """
    Independent bootstrap sampler.

    Each replicate draws ``n_samples`` indices independently and uniformly
    with replacement from the original dataset.

    Args:
        random_state (int | None): Optional seed controlling random operations.
    """

    def sample(
        self,
        n_samples:int,
        n_resamples:int,
    ) -> Iterator[BootstrapSample]:
        """
        Generate independent bootstrap samples.

        Args:
            n_samples (int): Number of observations. Must be positive.
            n_resamples (int): Number of bootstrap samples. Must be positive.

        Yields:
            BootstrapSample: Training indices of length ``n_samples`` and
                their out-of-bag complement.

        Raises:
            ValueError: If either size is not a positive integer.
        """
        self._check_positive_integer(n_samples, "n_samples")
        self._check_positive_integer(n_resamples, "n_resamples")
        n_samples = int(n_samples)
        n_resamples = int(n_resamples)

        seed = random.SeedGenerator(self.random_state)

        for _ in range(n_resamples):
            train_indices = random.randint(
                shape=(n_samples,),
                minval=0,
                maxval=n_samples,
                dtype="int32",
                seed=seed,
            )
            yield self._make_sample(train_indices, n_samples)


class BlockBootstrapSampler(BootstrapSampler):
    """
    Non-overlapping block bootstrap sampler.

    The original indices are partitioned into consecutive blocks of
    ``block_length`` observations. The last block may be shorter.
    Blocks are drawn uniformly with replacement until at least ``n_samples``
    indices have been collected. The concatenation is truncated to exactly
    ``n_samples`` indices, preserving order within each sampled block.

    Args:
        block_length (int): Number of observations per block. Must be positive
            and no greater than the dataset size at sampling time.
        random_state (int | None): Optional seed controlling random operations.

    Note:
        Blocks neither overlap nor wrap around the end of the dataset.
        A block spanning the entire dataset produces an empty out-of-bag set.
        Ensemble-wide out-of-bag coverage must be checked by the consumer.
    """

    def __init__(
        self,
        block_length:int,
        random_state:int|None=None,
    ) -> None:
        self._check_positive_integer(block_length, "block_length")
        super().__init__(random_state=random_state)
        self.block_length = int(block_length)

    def sample(
        self,
        n_samples:int,
        n_resamples:int,
    ) -> Iterator[BootstrapSample]:
        """
        Generate bootstrap samples by drawing consecutive blocks.

        Args:
            n_samples (int): Number of observations. Must be positive.
            n_resamples (int): Number of bootstrap samples. Must be positive.

        Yields:
            BootstrapSample: Training indices of length ``n_samples`` and
                their out-of-bag complement after truncation.

        Raises:
            ValueError: If either size is not a positive integer, or if
                ``block_length`` exceeds ``n_samples``.
        """
        self._check_positive_integer(n_samples, "n_samples")
        self._check_positive_integer(n_resamples, "n_resamples")
        n_samples = int(n_samples)
        n_resamples = int(n_resamples)

        if self.block_length > n_samples:
            raise ValueError(
                f"block_length must be <= number of samples. Provided "
                f"block_length: {self.block_length}, number of samples: {n_samples}."
            )

        n_blocks = (n_samples + self.block_length - 1) // self.block_length
        idxs = ops.arange(n_samples, dtype="int32")
        seed = random.SeedGenerator(self.random_state)

        for _ in range(n_resamples):
            blocks: list[IndexTensor] = []
            n_drawn = 0

            while n_drawn < n_samples:
                block_id = random.randint(
                    shape=(),
                    minval=0,
                    maxval=n_blocks,
                    dtype="int32",
                    seed=seed,
                )
                start = int(ops.item(block_id)) * self.block_length
                block = idxs[start : start + self.block_length]
                blocks.append(block)
                n_drawn += len(block)

            train_indices = ops.concatenate(blocks, axis=0)[:n_samples]
            yield self._make_sample(train_indices, n_samples)

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
This module provides data splitting schemes.
"""
import pkgutil
from abc import ABC
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np
from sklearn import model_selection

from deel.puncc.api.utils import features_len_check
from deel.puncc.api.utils import sample_len_check
from deel.puncc.api.utils import supported_types_check

if pkgutil.find_loader("pandas") is not None:
    import pandas as pd


class BaseSplitter(ABC):
    """Abstract structure of a splitter. A splitter provides a function
    that assignes data points to fit and calibration sets.

    :param int random_state: seed to control random generation.
    """

    def __init__(self, random_state=None) -> None:
        self.random_state = random_state

    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray]:
        raise NotImplementedError("Use a subclass of Splitter.")


class IdSplitter(BaseSplitter):
    """Identity splitter that wraps an already existing data assignment.

    :param Iterable X_fit: Fit features.
    :param Iterable y_fit: Fit labels.
    :param Iterable X_calib: calibration features.
    :param Iterable y_calib: calibration labels.
    """

    def __init__(
        self,
        X_fit: Iterable,
        y_fit: Iterable,
        X_calib: Iterable,
        y_calib: Iterable,
    ):
        super().__init__(random_state=None)

        # Checks
        supported_types_check(X_fit, y_fit)
        supported_types_check(X_calib, y_calib)
        sample_len_check(X_fit, y_fit)
        sample_len_check(X_calib, y_calib)
        features_len_check(X_fit, X_calib)

        self._split = [(X_fit, y_fit, X_calib, y_calib)]

    def __call__(self, X=None, y=None) -> Tuple[Iterable]:
        """Wraps into a splitter the provided fit and calibration subsets.

        :param Iterable X: features array. Not needed here, just a placeholder
            for interoperability.
        :param Iterable y: labels array. Not needed here, just a placeholder
            for interoperability.

        :returns: List of one tuple of deterministic subsets
            (X_fit, y_fit, X_calib, y_calib).
        :rtype: List[Tuple[Iterable]]
        """
        return self._split


class RandomSplitter(BaseSplitter):
    """Random splitter that assign samples given a ratio.

    :param float ratio: ratio of data assigned to the training
        (1-ratio to calibration).
    :param int random_state: seed to control random generation.
    """

    def __init__(self, ratio, random_state=None):
        if (ratio <= 0) or (ratio >= 1):
            raise ValueError(f"Ratio must be in (0,1). Provided value: {ratio}")
        self.ratio = ratio
        super().__init__(random_state=random_state)

    def __call__(
        self,
        X: Iterable,
        y: Iterable,
    ) -> Tuple[Iterable]:
        """Implements a random split strategy.

        :param Iterable X: features array.
        :param Iterable y: labels array.

        :returns: List of one tuple of random subsets
            (X_fit, y_fit, X_calib, y_calib).
        :rtype: List[Tuple[Iterable]]
        """
        # Checks
        supported_types_check(X, y)
        sample_len_check(X, y)

        rng = np.random.RandomState(seed=self.random_state)
        fit_idxs = rng.rand(len(X)) > self.ratio
        cal_idxs = np.invert(fit_idxs)
        return [(X[fit_idxs], y[fit_idxs], X[cal_idxs], y[cal_idxs])]


class KFoldSplitter(BaseSplitter):
    """KFold data splitter.

    :param int K: number of folds to generate.
    :param int random_state: seed to control random generation.
    """

    def __init__(self, K: int, random_state=None) -> None:
        if K < 2:
            raise ValueError(f"K must be >= 2. Provided value: {K}.")
        self.K = K
        super().__init__(random_state=random_state)

    def __call__(
        self,
        X: Iterable,
        y: Iterable,
    ) -> List[Tuple[Iterable]]:
        """Implements a K-fold split strategy.

        :param Iterabler X: features array.
        :param Iterable y: labels array.

        :returns: list of K split folds. Each fold is a tuple
            (X_fit, y_fit, X_calib, y_calib).
        :rtype: List[Tuple[Iterable]]
        """
        # Checks
        supported_types_check(X, y)
        sample_len_check(X, y)

        kfold = model_selection.KFold(
            self.K, shuffle=True, random_state=self.random_state
        )
        folds = []

        for fit, calib in kfold.split(X):
            if pkgutil.find_loader("pandas") is not None and isinstance(
                X, pd.DataFrame
            ):
                if isinstance(y, pd.DataFrame):
                    folds.append(
                        (
                            X.iloc[fit],
                            y.iloc[fit],
                            X.iloc[calib],
                            y.iloc[calib],
                        )
                    )
                else:
                    folds.append((X.iloc[fit], y[fit], X.iloc[calib], y[calib]))

            else:
                bool_fit_idx = np.array([i in fit for i in range(len(X))])
                folds.append(
                    (
                        X[bool_fit_idx],
                        y[bool_fit_idx],
                        X[~bool_fit_idx],
                        y[~bool_fit_idx],
                    )
                )

        return folds

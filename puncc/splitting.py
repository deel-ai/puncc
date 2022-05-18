"""
This module provides data splitting schemes.
"""

from abc import ABC
import sklearn
import numpy as np


class Splitter(ABC):
    """Abstract structure of a splitter. A splitter provides a function
    that assignes data points to fit and calibration sets.

    Attributes:
        random_state: seed to control random generation
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Use a subclass of Splitter.")


class IdSplitter(Splitter):
    """Identity splitter that wraps an already existing data assignment."""

    def __init__(self, X_fit, y_fit, X_calib, y_calib):
        super().__init__(random_state=None)
        self._split = [(X_fit, y_fit, X_calib, y_calib)]

    def __call__(self, *args, **kwargs):
        return self._split


class RandomSplitter(Splitter):
    """Random splitter that assign samples given a ratio.

    Attributes:
        ratio: ratio of data assigned to the training (1-ratio to calibration)
        random_state: seed to control random generation
    """

    def __init__(self, ratio, random_state=None):
        self.ratio = ratio
        super().__init__(random_state=random_state)

    def __call__(self, X, y):
        rng = np.random.RandomState(seed=self.random_state)
        fit_idxs = rng.rand(len(X)) > self.ratio
        cal_idxs = np.invert(fit_idxs)
        return [(X[fit_idxs], y[fit_idxs], X[cal_idxs], y[cal_idxs])]


class KFoldSplitter(Splitter):
    """KFold data splitter.

    Attributes:
        K: number of folds to generate
        random_state: seed to control random generation
    """

    def __init__(self, K, random_state=None):
        self.K = K
        super().__init__(random_state=random_state)

    def __call__(self, X, y):
        kfold = sklearn.model_selection.KFold(
            self.K, shuffle=True, random_state=self.random_state
        )
        splits = list()
        for fit, calib in kfold.split(X):
            splits.append((X[fit], y[fit], X[calib], y[calib]))
        return splits

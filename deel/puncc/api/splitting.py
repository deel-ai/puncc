"""
This module provides data splitting schemes.
"""

from abc import ABC
from sklearn import model_selection
import numpy as np


class BaseSplitter(ABC):
    """Abstract structure of a splitter. A splitter provides a function
    that assignes data points to fit and calibration sets.

    :param int random_state: seed to control random generation.
    """

    def __init__(self, random_state=None) -> None:
        self.random_state = random_state

    def __call__(self, *args, **kwargs) -> tuple[np.ndarray]:
        raise NotImplementedError("Use a subclass of Splitter.")


class IdSplitter(BaseSplitter):
    """Identity splitter that wraps an already existing data assignment.

    :param tuple[ndarray, ndarray, ndarray, ndarray] _split: provided split.
    """

    def __init__(self, X_fit, y_fit, X_calib, y_calib):
        super().__init__(random_state=None)
        self._split = [(X_fit, y_fit, X_calib, y_calib)]

    def __call__(self, X=None, y=None):
        """Wraps into a splitter the provided fit and calibration subsets.

        :param ndarray X: features array. Not needed here, just a placeholder for interoperability.
        :param ndarray y: labels array. Not needed here, just a placeholder for interoperability.

        :returns: Tuple of deterministic subsets (X_fit, y_fit, X_calib, y_calib).
        :rtype: tuple[ndarray, ndarray, ndarray, ndarray]
        """
        return self._split


class RandomSplitter(BaseSplitter):
    """Random splitter that assign samples given a ratio.

    :param float ratio: ratio of data assigned to the training (1-ratio to calibration).
    :param int random_state: seed to control random generation.
    """

    def __init__(self, ratio, random_state=None):
        self.ratio = ratio
        super().__init__(random_state=random_state)

    def __call__(self, X, y):
        """Implements a random split strategy.

        :param ndarray X: features array.
        :param ndarray y: labels array.

        :returns: Tuple of random subsets (X_fit, y_fit, X_calib, y_calib).
        :rtype: tuple[ndarray, ndarray, ndarray, ndarray]
        """
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
        self.K = K
        super().__init__(random_state=random_state)

    def __call__(self, X, y):
        """Implements a K-fold split strategy.

        :param ndarray X: features array.
        :param ndarray y: labels array.

        :returns: list of K split folds. Each fold is a tuple (X_fit, y_fit, X_calib, y_calib).
        :rtype: list[tuple[ndarray, ndarray, ndarray, ndarray]]
        """
        kfold = model_selection.KFold(
            self.K, shuffle=True, random_state=self.random_state
        )
        folds = list()
        for fit, calib in kfold.split(X):
            folds.append((X[fit], y[fit], X[calib], y[calib]))
        return folds

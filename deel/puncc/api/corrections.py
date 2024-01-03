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
This module provides correction functions for multiple hypothesis testing. To be used
when building a :ref:`calibrator <calibration>` for multivariate regression or object detection.
"""
from typing import Union

import numpy as np


def bonferroni(alpha: float, nvars: int = 1) -> Union[float, np.ndarray]:
    """Bonferroni correction for multiple comparisons.

    :param float alpha: nominal coverage level.
    :param int nvars: number of output features.


    :returns: corrected coverage level.
    :rtype: float or ndarray.
    """
    # Sanity checks
    if np.any(alpha <= 0) or np.any(alpha >= 1):
        raise ValueError("alpha must be in (0,1)")

    if nvars <= 0:
        raise ValueError("nvars must be a positive integer")

    if nvars == 1:
        return alpha

    return np.ones(nvars) * alpha / nvars


def weighted_bonferroni(alpha: float, weights: np.ndarray) -> np.ndarray:
    """Weighted Bonferroni correction for multiple comparisons.

    :param float alpha: nominal coverage level.
    :param np.ndarray weights: weights associated to each output feature.


    :returns: array of corrected featurewise coverage levels.
    :rtype: np.ndarray.
    """
    # Sanity checks
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0,1)")

    # Positivity check
    positiveness_condition = np.all(weights > 0)
    if not positiveness_condition:
        raise RuntimeError("weights must be positive")

    # Normalization check
    norm_condition = np.isclose(np.sum(weights) - 1, 0, atol=1e-14)
    if not norm_condition:
        error = (
            "weights are not normalized. Sum of weights is"
            + f"{np.sum(weights)}"
        )
        raise RuntimeError(error)

    return alpha * weights

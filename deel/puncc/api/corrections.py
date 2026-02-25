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
This module provides correction functions for multiple hypothesis testing.
To be used when building a conformal predictor for multivariate regression or object detection.
"""
from deel.puncc.typing import TensorLike
from deel.puncc import ops
from typing import TypeAlias, Callable

CorrectionFunction:TypeAlias = Callable[[float|TensorLike], float|TensorLike]

def bonferroni(nvars:int=1)->CorrectionFunction:
    """
    Bonferroni correction for multiple comparisons.

    Args:
        nvars (int, optional): Number of output features.. Defaults to 1.

    Returns:
        CorrectionFunction: Bonferonni correction function
    """
    def _bonferroni(alpha: float | TensorLike) -> float | TensorLike:
        """
        Bonferonni Correction function

        Args:
            alpha (float | TensorLike): nominal coverage level.

        Returns:
            float | TensorLike: corrected coverage level.
        """
        if nvars == 1:
            return alpha
        return ops.ones(nvars) * alpha / nvars
    return _bonferroni

def weighted_bonferroni(weights: TensorLike) -> CorrectionFunction:
    """
    Weighted Bonferroni correction for multiple comparisons.

    Args:
        weights (TensorLike): weights associated to each output feature.

    Returns:
        CorrectionFunction: Weighted Bonferonni correction function
    """
    def _weighted_bonferroni(alpha: float | TensorLike) -> float | TensorLike:
        """
        Weighted Bonferonni correction function.

        Args:
            alpha (float | TensorLike): Nominal coverage level.

        Returns:
            float | TensorLike: Corrected featurewise coverage levels.
        """
        # normalization of weights
        w = weights / ops.sum(weights)
        return alpha * w
    return _weighted_bonferroni

def sidak(nvars:int=1)->CorrectionFunction:
    """
    Sidak correction for multiple comparisons.

    Args:
        nvars (int, optional): Number of output features.. Defaults to 1.

    Returns:
        CorrectionFunction: Correction function implementing the Sidak correction.
    """
    def _sidak(alpha: float | TensorLike) -> float | TensorLike:
        """
        Sidak correction function.

        Args:
            alpha (float | TensorLike): Nominal coverage level.

        Returns:
            float | TensorLike: Corrected coverage level.
        """
        if nvars == 1:
            return alpha
        return ops.ones(nvars) * (1 - (1 - alpha) ** (1 / nvars))
    return _sidak

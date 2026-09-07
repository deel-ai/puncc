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

AlphaCorrection:TypeAlias = Callable[[float|TensorLike], float|TensorLike]

def bonferroni(nvars:int=1)->AlphaCorrection:
    """
    Bonferroni correction for multiple comparisons.

    Args:
        nvars (int, optional): Number of output features.. Defaults to 1.

    Returns:
        AlphaCorrection: Bonferroni correction function
    """
    if nvars < 1:
        raise ValueError("nvars must be a positive integer.")
    def _bonferroni(alpha: float | TensorLike) -> float | TensorLike:
        """
        Bonferroni Correction function

        Args:
            alpha (float | TensorLike): nominal miscoverage level.

        Returns:
            float | TensorLike: corrected miscoverage level.
        """
        if nvars == 1:
            return alpha
        return ops.ones((nvars,)) * alpha / nvars
    return _bonferroni

def weighted_bonferroni(weights: TensorLike) -> AlphaCorrection:
    """
    Weighted Bonferroni correction for multiple comparisons.

    Args:
        weights (TensorLike): weights associated to each output feature.

    Returns:
        AlphaCorrection: Weighted Bonferroni correction function
    """
    def _weighted_bonferroni(alpha: float | TensorLike) -> float | TensorLike:
        """
        Weighted Bonferroni correction function.

        Args:
            alpha (float | TensorLike): Nominal miscoverage level.

        Returns:
            float | TensorLike: Corrected featurewise miscoverage levels.
        """
        # normalization of weights
        w = weights / ops.sum(weights)
        return alpha * w
    return _weighted_bonferroni

def sidak(nvars:int=1)->AlphaCorrection:
    """
    Sidak correction for multiple comparisons.

    Args:
        nvars (int, optional): Number of output features.. Defaults to 1.

    Returns:
        AlphaCorrection: Correction function implementing the Sidak correction.
    """
    if nvars < 1:
        raise ValueError("nvars must be a positive integer.")
    def _sidak(alpha: float | TensorLike) -> float | TensorLike:
        """
        Sidak correction function.

        Args:
            alpha (float | TensorLike): Nominal miscoverage level.

        Returns:
            float | TensorLike: Corrected miscoverage level.
        """
        if nvars == 1:
            return alpha
        return ops.ones((nvars,)) * (1 - (1 - alpha) ** (1 / nvars))
    return _sidak

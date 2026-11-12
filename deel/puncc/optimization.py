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
Optimization tools for CRC.
"""


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable
from scipy.optimize import root_scalar
import math

ScalarFunction = Callable[[float], float]


class ScalarOptimizer(ABC):
    """
    Abstract class for optimizers used by CRC conformal predictors.
    """

    def __init__(
        self,
        *,
        xtol: float = 1e-6,
        rtol: float = 1e-6,
        maxiter: int = 100,
    ):
        self.xtol = xtol
        self.rtol = rtol
        self.maxiter = maxiter

    @abstractmethod
    def __call__(
        self,
        function: ScalarFunction,
        a: float,
        b: float,
    ) -> float: ...


class BinarySearchOptimizer(ScalarOptimizer):
    """
    Find the smallest feasible parameter using binary search.

    The objective function is assumed to be non-increasing over the search interval.
    A parameter is feasible when the objective function is non-positive.

    Unlike root-finding algorithms, binary search does not require the objective function to be continuous, making
    it suitable for monotone step functions such as empirical risk functions used in Conformal Risk Control.

    Args:
        xtol:
            Absolute convergence tolerance.
        rtol:
            Relative convergence tolerance.
        maxiter:
            Maximum number of iterations.

    Notes:
        If the lower bound is already feasible, it is returned.
        If the upper bound is infeasible, a ValueError is raised.

        Otherwise, the algorithm maintains a feasible upper bound and returns it when convergence is reached or
        the maximum number of iterations is exceeded.
    """

    def __call__(
        self,
        function: ScalarFunction,
        a: float,
        b: float,
    ) -> float:
        if a >= b:
            raise ValueError(
                "lower bound must be strictly smaller than upper bound in optimization process."
            )

        f_a = function(a)
        f_b = function(b)

        if f_a <= 0:
            return a

        if f_b > 0:
            raise ValueError("No feasible solution was found in given bounds.")

        low = a
        high = b

        for _ in range(self.maxiter):
            mid = (low + high) / 2
            f_mid = function(mid)
            if not math.isfinite(f_mid):
                raise ValueError("Objective function returned a non-finite value.")
            if f_mid <= 0:
                high = mid
            else:
                low = mid

            if high - low <= self.xtol + self.rtol * abs(high):
                break

        return high


class ScipyOptimizer(ScalarOptimizer):
    """
    Base class for scalar root-finding optimizers using SciPy.

    The optimizer calls scipy.optimize.root_scalar with the root-finding method specified by the subclass.

    Args:
        xtol:
            Absolute convergence tolerance.
        rtol:
            Relative convergence tolerance.
        maxiter:
            Maximum number of iterations.

    Notes:
        Bracketing root-finding methods generally require a continuous objective function with a suitable sign
        change over the search interval.

        They may be unsuitable for discontinuous empirical risk functions commonly encountered in Conformal
        Risk Control. BinarySearchOptimizer is generally preferable for monotone step functions.
    """

    method: str

    def __call__(
        self,
        function: ScalarFunction,
        a: float,
        b: float,
    ) -> float:
        result = root_scalar(
            function,
            bracket=(a, b),
            method=self.method,
            xtol=self.xtol,
            rtol=self.rtol,
            maxiter=self.maxiter,
        )
        if not result.converged:
            raise RuntimeError(f"SciPy optimizer {self.method!r} did not converge.")

        root = float(result.root)
        if function(root) <= 0:
            return root
        candidate = min(root + self.xtol + self.rtol * abs(root), b)

        if function(candidate) <= 0:
            return candidate
        raise RuntimeError(
            "Optimizer converged to an infeasible point and no feasible point was found within its numerical tolerance."
        )


__Scipy_subclass_doc = """
    A root-finding optimizer using SciPy's {method} method.

    Args:
        xtol:
            Absolute convergence tolerance.
        rtol:
            Relative convergence tolerance.
        maxiter:
            Maximum number of iterations.
"""


class BrentqOptimizer(ScipyOptimizer):
    __doc__ = __Scipy_subclass_doc.format(method="brentq")
    method: str = "brentq"


class BrenthOptimizer(ScipyOptimizer):
    __doc__ = __Scipy_subclass_doc.format(method="brenth")
    method: str = "brenth"


class RidderOptimizer(ScipyOptimizer):
    __doc__ = __Scipy_subclass_doc.format(method="ridder")
    method: str = "ridder"


class TomsOptimizer(ScipyOptimizer):
    __doc__ = __Scipy_subclass_doc.format(method="toms748")
    method: str = "toms748"

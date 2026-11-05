from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable
from scipy.optimize import root_scalar
import math

ScalarFunction = Callable[[float], float]


class ScalarOptimizer(ABC):
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
                raise ValueError(
                    "Objective function returned a non-finite value."
                )
            if f_mid <= 0:
                high = mid
            else:
                low = mid

            if high - low <= self.xtol + self.rtol * abs(high):
                break

        return high


class ScipyOptimizer(ScalarOptimizer):
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
            raise RuntimeError(
                f"SciPy optimizer {self.method!r} did not converge."
            )

        root = float(result.root)
        if function(root) <= 0:
            return root
        candidate = min(root + self.xtol + self.rtol * abs(root), b)

        if function(candidate) <= 0:
            return candidate
        raise RuntimeError(
            "Optimizer converged to an infeasible point and no feasible point was found within its numerical tolerance."
        )


class BrentqOptimizer(ScipyOptimizer):
    method: str = "brentq"


class BrenthOptimizer(ScipyOptimizer):
    method: str = "brenth"


class RidderOptimizer(ScipyOptimizer):
    method: str = "ridder"


class TomsOptimizer(ScipyOptimizer):
    method: str = "toms748"

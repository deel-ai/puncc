from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable
from scipy.optimize import root_scalar

ScalarFunction = Callable[[float], float]


@runtime_checkable
class ScalarOptimizer(Protocol):
    def __call__(
        self,
        function: ScalarFunction,
        a: float,
        b: float,
        *,
        xtol: float | None = None,
        rtol: float | None = None,
        maxiter: int | None = None,
    ) -> float:
        ...

class BinarySearchOptimizer:
    def __call__(
        self,
        function: ScalarFunction,
        a: float,
        b: float,
        *,
        xtol: float | None = 1e-4,
        rtol: float | None = None,
        maxiter: int | None = 50,
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
            raise ValueError(
                "No feasible solution was found in given bounds."
            )

        xtol = 0.0 if xtol is None else xtol
        rtol = 0.0 if rtol is None else rtol
        maxiter = 50 if maxiter is None else maxiter

        low = a
        high = b

        for _ in range(maxiter):
            mid = (low + high) / 2
            f_mid = function(mid)

            if f_mid <= 0:
                high = mid
            else:
                low = mid

            tolerance = xtol + rtol * abs(high)

            if high - low <= tolerance:
                break

        return high


class ScipyRootFinder:
    method:str
    def __init__(
        self,
        method: str = "brentq",
    ):
        self.method = method or self.__class__.method

    def __call__(
        self,
        function: ScalarFunction,
        a: float,
        b: float,
        *,
        xtol: float | None = None,
        rtol: float | None = None,
        maxiter: int | None = None,
    ) -> float:
        result = root_scalar(
            function,
            bracket=(a, b),
            method=self.method,
            xtol=xtol,
            rtol=rtol,
            maxiter=maxiter,
        )

        if not result.converged:
            raise RuntimeError(
                f"SciPy optimizer {self.method!r} did not converge."
            )

        return float(result.root)
    
class BrentqOptimizer(ScipyRootFinder):
    method:str="brentq"

class BrenthOptimizer(ScipyRootFinder):
    method:str="brenth"

class RidderOptimizer(ScipyRootFinder):
    method:str="ridder"

class TomsOptimizer(ScipyRootFinder):
    method:str="toms748"
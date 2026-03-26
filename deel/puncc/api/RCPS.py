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
This module implements Risk Controlling Prediction Sets method as described in [paper]
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm, binom
from typing import Optional
from typing import Callable, Iterable, Any
from functools import lru_cache
from deel.puncc.api.conformalization import ConformalMethod
from deel.puncc.typing import TensorLike, LambdaPredictor


class RCPS(ConformalMethod):
    def __init__(self, model:LambdaPredictor,
                 loss_function:Callable[[Iterable, Iterable], Iterable[float]], 
                 loss_function_upper_bound:int=1):
        super().__init__(model)
        self.loss_function_upper_bound = loss_function_upper_bound
        self.loss_function = lambda lam: lambda X, y: loss_function(model.predict(X, lam), y) / loss_function_upper_bound

        self.losses = []
        self.ucb = None


    def is_calibrated(self):
        return self.ucb is not None


    def calibrate(self, X_calib:Iterable[Any], y_calib:Iterable[Any], delta:float, method:str="wsr"):
        self.losses = [lambda lam, i=i: self.loss_function(lam)(X_calib[i:i+1], y_calib[i:i+1])[0] for i in range(len(y_calib))]
        self.compute_ucb(self.losses, delta, method)


    def compute_ucb(self, losses:Iterable[Callable], delta:float, method:str="wsr"):
        n_calib = len(losses)
        if n_calib < 2:
            raise ValueError("At least 2 calibration samples are required to compute UCB.")
        @lru_cache(maxsize=1)
        def _eval(lam: float):
            vals = np.array([f(lam) for f in losses])
            return vals.mean(), vals.std(ddof=1)

        risk     = lambda lam: _eval(lam)[0]
        risk_std = lambda lam: _eval(lam)[1]

        if method == "clt":
            self.ucb = clt_ucb(risk, risk_std, delta, n_calib)
        elif method == "simplified_hoeffding":
            self.ucb = simplified_hoeffding_ucb(risk, delta, n_calib)
        elif method == "tighter_hoeffding":
            self.ucb = tighter_hoeffding_ucb(risk, delta, n_calib)
        elif method == "bentkus":
            self.ucb = bentkus_ucb(risk, delta, n_calib)
        elif method == "hoeffding_bentkus":
            self.ucb = hoeffding_bentkus_ucb(risk, delta, n_calib)
        elif method == "bernstein":
            self.ucb = bernstein_ucb(risk, risk_std, delta, n_calib)
        elif method == "wsr":
            self.ucb = wsr_ucb(losses, delta)
        else:
            raise ValueError(f"Unknown UCB method: {method}")


    def __get_lambda_from_alpha(self, alpha:float, lambda_grid:Iterable[float])->float:
        lambda_grid = np.sort(lambda_grid)
        ucb_values = np.vectorize(self.ucb)(lambda_grid)
        violation_indices = np.where(ucb_values >= alpha)[0]
        if len(violation_indices) == 0:
            # ucb < alpha for all lambda, the smallest lambda controls the risk
            return float(lambda_grid[0])

        last_violation = violation_indices[-1]
        if last_violation == len(lambda_grid) - 1:
            # ucb >= alpha for all lambda, no lambda controls the risk
            raise ValueError("No lambda in the grid controls the risk at level alpha.")

        return float(lambda_grid[last_violation + 1])


    def predict(self, X_test:Iterable[Any], alpha:float|TensorLike, lambda_grid:Iterable[float]) -> Iterable[Any]:
        if not isinstance(alpha, float):
            raise NotImplementedError("Only scalar alpha is supported for RCPS.")
        if not self.is_calibrated():
            raise ValueError("Model must be calibrated before making predictions.")
        # TODO : check for monotonous loss to find root faster
        lambda_hat = self.__get_lambda_from_alpha(alpha/self.loss_function_upper_bound, lambda_grid)
        c_lambda_pred = self.model.predict(X_test, lambda_hat)
        return c_lambda_pred


def clt_ucb(risk:Callable, risk_std:Callable, delta:float, n_calib:int) -> Callable:
    z = norm.ppf(1 - delta)
    return lambda lam: risk(lam) + z * risk_std(lam) / np.sqrt(n_calib)


def simplified_hoeffding_ucb(risk:Callable, delta:float, n_calib:int) -> Callable:
    return lambda lam:risk(lam) + np.sqrt(np.log(1/delta) / (2 * n_calib))


def tighter_hoeffding_ucb(risk:Callable, delta:float, n_calib:int) -> Callable:
    def tpb(t: float, R: float) -> float:
        return tighter_hoeffding_tpb(t, R, n_calib)
    return tpb_to_ucb(tpb, risk, delta)


def bentkus_ucb(risk:Callable, delta:float, n_calib:int) -> Callable:
    def tpb(t: float, R: float) -> float:
        return bentkus_tpb(t, R, n_calib)
    return tpb_to_ucb(tpb, risk, delta)


def hoeffding_bentkus_ucb(risk:Callable, delta:float, n_calib:int) -> Callable:
    def tpb(t: float, R: float) -> float:
        return min(tighter_hoeffding_tpb(t, R, n_calib), bentkus_tpb(t, R, n_calib))
    return tpb_to_ucb(tpb, risk, delta)


def bernstein_ucb(risk:Callable, risk_std:Callable, delta:float, n_calib:int) -> Callable:
    c1 = np.sqrt(2 * np.log(2/delta) / n_calib)
    c2 = 7 * np.log(2/delta) / (3 * (n_calib - 1))
    return lambda lam: risk(lam) + risk_std(lam) * c1 + c2


def wsr_ucb(losses: Iterable[Callable], delta: float,
            R_grid: Optional[np.ndarray] = None) -> Callable:
    """Waudby-Smith-Ramdas (WSR) upper confidence bound.

    For a fixed lambda, letting L_i = L_i(lambda):

        mu_i     = (1/2 + sum_{j=1}^i L_j) / (1+i),        sigma2_0 = 1/4
        sigma2_i = (1/4 + sum_{j=1}^i (L_j - mu_j)^2) / (1+i)
        nu_i     = min(1, sqrt(2 ln(1/delta)) / (n * sigma2_{i-1}))
        K_i(R)   = prod_{j=1}^i (1 - nu_j (L_j - R))

    :param Iterable[Callable] losses: individual loss functions f_1,...,f_n.
    :param float delta: confidence level in (0, 1).
    :param np.ndarray R_grid: grid of candidate R values.
        Defaults to 1000 evenly spaced points in [0, 1].
    :returns: function lambda -> inf{R >= 0 : max_{i=1,...,n} K_i(R;lambda) > 1/delta}.
    :rtype: callable
    """
    losses = list(losses)
    n = len(losses)
    if R_grid is None:
        R_grid = np.linspace(0.0, 1.0, 1000)

    def ucb(lam: float) -> float:
        L = np.array([f(lam) for f in losses])

        # Running mean with 1/2 prior
        i_arr = np.arange(n)
        mu = (0.5 + np.cumsum(L)) / (i_arr + 2)

        # Running variance with 1/4 prior (sigma2[0] is the prior)
        sigma2 = np.empty(n + 1)
        sigma2[0] = 0.25
        sigma2[1:] = (0.25 + np.cumsum((L - mu) ** 2)) / (i_arr + 2)

        # Betting fractions nu_i use sigma2_{i-1}, i.e. sigma2[:n]
        nu = np.minimum(1.0, np.sqrt(2.0 * np.log(1.0 / delta)) / (n * sigma2[:n]))

        # factors[i, r] = 1 - nu[i] * (L[i] - R_grid[r])
        # Guaranteed in [0, 2] since L, R in [0,1] and nu in [0,1]
        factors = 1.0 - nu[:, None] * (L[:, None] - R_grid[None, :])
        K = np.cumprod(factors, axis=0)  # K[i, r] = K_{i+1}(R_grid[r])

        # UCB = inf{R : max_{i=1,...,n} K_i(R) > 1/delta}
        qualifying = R_grid[np.max(K, axis=0) > 1.0 / delta]
        return float(qualifying.min()) if len(qualifying) > 0 else 1.0

    return ucb


def tpb_to_ucb(tpb: Callable, risk: Callable, delta: float,
               R_grid: Optional[np.ndarray] = None) -> Callable:
    """Convert a tail probability bound into an upper confidence bound function.

    For each lambda, returns sup{R : tpb(risk(lambda), R) >= delta},
    i.e. the largest risk level still deemed plausible at confidence delta.

    The only assumption on tpb is that it is nondecreasing in its first
    argument for every fixed R.  No monotonicity in R is assumed, so the
    supremum is computed by exhaustive search over R_grid.

    :param callable tpb: tail probability bound tpb(r_hat, R) -> [0, 1],
        nondecreasing in its first argument.
    :param callable risk: empirical risk function risk(lambda) -> [0, 1].
    :param float delta: confidence level in (0, 1).
    :param np.ndarray R_grid: grid of candidate R values to search over.
        Defaults to 1000 evenly spaced points in [0, 1].
    :returns: function lambda -> sup{R : tpb(risk(lambda), R) >= delta}.
    :rtype: callable
    """
    if R_grid is None:
        R_grid = np.linspace(0.0, 1.0, 1000)

    def ucb(lam: float) -> float:
        r_hat = risk(lam)
        tpb_values = np.vectorize(lambda R: tpb(r_hat, R))(R_grid)
        qualifying = R_grid[tpb_values >= delta]
        # If no R satisfies the condition, return 1.0 as a conservative upper bound.
        return float(qualifying.max()) if len(qualifying) > 0 else 1.0

    return ucb


def bernoulli_rate_function(t: float, p: float) -> float:
    if p < 0 or p > 1:
        raise ValueError("Bernoulli rate function is only defined for p in [0, 1].")
    if t < 0 or t > 1:
        raise ValueError("Bernoulli rate function is only defined for t in [0, 1].")
    if p == 0:
        return 0.0 if t == 0 else np.inf
    if p == 1:
        return 0.0 if t == 1 else np.inf
    if t == 0:
        return -np.log(1 - p)
    elif t == 1:
        return -np.log(p)
    else:
        return t * np.log(t / p) + (1 - t) * np.log((1 - t) / (1 - p))


def tighter_hoeffding_tpb(t: float, R: float, n: int) -> float:
    return np.exp(-2 * n * bernoulli_rate_function(t, R))


def bentkus_tpb(t: float, R: float, n: int) -> float:
    return min(1.0, np.e * binom.cdf(int(np.ceil(n * t)), n, R))

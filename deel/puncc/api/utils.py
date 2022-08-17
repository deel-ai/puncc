#
# This module implements utility functions.
#

import numpy as np
from typing import Tuple, Optional
import matplotlib
import matplotlib.pyplot as plt
import sys

EPSILON = sys.float_info.min  # small value to avoid underflow


def check_alpha_calib(alpha: float, n: int, complement_check: bool = False):
    """Check if the value of alpha is consistent with the size of calibration set.

    The quantile order is inflated by a factor :math:`(1+1/n)` and has to be in the interval (0,1). From this, we derive the condition:
        * :math:`0 < (1-alpha)\cdot(1+1/n) < 1 \implies 1 > alpha > 1/(n+1)`
    If complement_check is set, we consider an additional condition:
        - :math:`0 < alpha \cdot (1+1/n) < 1 \implies 0 < alpha < n/(n+1)`

    :param float alpha: target quantile order.
    :param int n: size of the calibration dataset.
    :param bool complement_check: complementary check to compute the :math:`alpha \cdot (1+1/n)`-th quantile, required by some methods such as jk+.

    :raises ValueError: if the value of alpha is inconsistent with the size of calibration set.
    """
    if alpha < 1 / (n + 1):
        raise ValueError(
            f"Alpha is too small: (alpha={alpha}, n={n}) {alpha} < {1/(n+1)}. "
            + "Increase alpha or the size of the calibration set."
        )
    if alpha >= 1:
        raise ValueError(
            f"Alpha={alpha} is too large. Decrease alpha such that alpha < 1."
        )
    if complement_check and alpha > n / (n + 1):
        raise ValueError(
            f"Alpha is too large: (alpha={alpha}, n={n}) {alpha} < {n/(n+1)}. "
            + "Decrease alpha or increase the size of the calibration set."
        )

    if complement_check and alpha <= 0:
        raise ValueError(
            f"Alpha={alpha} is too small. Increase alpha such that alpha > 0."
        )


def get_min_max_alpha_calib(
    n: int, two_sided_conformalization: bool = False
) -> Optional[Tuple[float, float]]:
    """Get the greatest alpha in (0,1) that is consistent with the size of calibration
    set. Recall that while true Conformal Prediction (CP, Vovk et al 2005) is defined only on (0,1)
    (bounds not included), there maybe cases where we must handle alpha=0 or 1
    (e.g. Adaptive Conformal Inference, Gibbs & Candès 2021).

    This function computes the admissible range of alpha for split CP (Papadopoulos et al 2002,
    Lei et al 2018) and the Jackknife-plus (Barber et al 2019).
    For more CP method, see the literature and update this function.

    In Split Conformal prediction we take the index := ceiling( (1-alpha)(n+1) )-th element
    from the __sorted__ nonconformity scores as the margin of error.
    We must have 1 <= index <= n ( = len(calibration_set)) which boils down to: 1/(n+1) <= alpha < 1.

    In the case of cv-plus and jackknife-plus (Barber et al 2019), we need:
        1. ceiling( (1-alpha)(n+1) )-th element of scores
        2. floor(     (alpha)(n+1) )-th element of scores

    The argument two_sided_conformalization=True, ensures the indexes are within
    range for this special case: 1/(n+1) <= alpha <= n/(n+1)

    :param int n: size of the calibration dataset.
    :param bool two_sided_conformalization: alpha threshold for two-sided quantile of jackknife+ (Barber et al 2021).

    :returns: lower and upper bounds for alpha (miscoverage probability)
    :rtype: Tuple[float, float]

    :raises ValueError: must have integer n, boolean two_sided_conformalization and n>=1
    """
    if isinstance(n, int) and isinstance(two_sided_conformalization, bool) and n >= 1:
        # Special case: Conformal Prediction with jackknife-plus (Barber et al 2019)
        if two_sided_conformalization:
            return (1 / (n + 1), n / (n + 1))
        # Base case: split conformal prediction (e.g. Lei et al 2019)
        else:
            return (1 / (n + 1), 1)
    else:
        if not isinstance(n, int):
            raise ValueError(
                f"Invalid input: need isinstance(n, int)==True but received {type(n)=}"
            )
        elif not isinstance(two_sided_conformalization, bool):
            raise ValueError(
                f"Invalid input: need isinstance(two_sided_conformalization, bool)==True but received {type(two_sided_conformalization)=}"
            )
        elif n < 1:
            raise ValueError(f"Invalid input: you need n>=1 but received n={n}")


def quantile(a: np.ndarray, q: float, w: np.ndarray = None):  # type: ignore
    """Estimate the q-th empirical weighted quantiles.

    :param ndarray a: vector of n samples
    :param float q: target quantile order. Must be in the open interval (0, 1).
    :param ndarray w: vector of size n. By default, w is None and equal weights = 1/m are associated.

    :returns: weighted empirical quantiles.
    :rtype: ndarray

    :raises NotImplementedError: a must be unidimensional.
    """
    # Sanity checks
    if q <= 0 or q >= 1:
        raise ValueError("q must be in the open interval (0, 1).")
    if a.ndim > 1:
        raise NotImplementedError(f"a dimension {a.ndim} should be 1.")
    if w is not None and w.ndim > 1:
        raise NotImplementedError(f"w dimension {w.ndim} should be 1.")

    # Case of None weights
    if w is None:
        return np.quantile(a, q=q, method="higher")
        ## An equivalent method would be to assign equal values to w
        ## and carry on with the computations.
        ## np.quantile is however more optimized.
        # w = np.ones_like(a) / len(a)

    # Sanity check
    if len(w) != len(a):
        error = "a and w must have the same shape:" + f"{len(a)} != {len(w)}"
        raise RuntimeError(error)

    # Normalization check
    norm_condition = np.isclose(np.sum(w, axis=-1), 1, atol=1e-6)
    if ~np.all(norm_condition):
        error = (
            f"W is not normalized. Sum of weights on" + f"rows is {np.sum(w, axis=-1)}"
        )
        raise RuntimeError(error)

    # Empirical Weighted Quantile
    sorted_idxs = np.argsort(a)  # rows are sorted in ascending order
    sorted_cumsum_w = np.cumsum(w[sorted_idxs])
    weighted_quantile_idxs = sorted_idxs[sorted_cumsum_w >= q][0]
    return a[weighted_quantile_idxs]


#
# ========================= Aggregation functions =========================
#


def agg_list(a: np.ndarray):
    """Ancillary function to aggregate array following the axis 0.

    :param ndarray a: array.

    :returns: the concatenated array or None
    :rtype: ndarray or None
    """
    try:
        return np.concatenate(a, axis=0)
    except ValueError:
        return None


#
# ========================= Visualization =========================
#

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
LARGE_SIZE = 15
HUGE_SIZE = 16


custom_rc_params = {
    "font.family": "Times New Roman",
    "ytick.labelsize": BIGGER_SIZE,
    "xtick.labelsize": BIGGER_SIZE,
    "axes.labelsize": LARGE_SIZE,
    "legend.fontsize": LARGE_SIZE,
    "axes.titlesize": HUGE_SIZE,
    "lines.linewidth": 2,
}


def plot_prediction_interval(
    y_true: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    X: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    sort_X: bool = False,
    **kwargs,
) -> None:
    """Plot prediction intervals whose bounds are given by y_pred_lower and y_pred_upper. True values and point estimates are also plotted if given as argument.

    :param ndarray y_true: label true values.
    :param ndarray y_pred_lower: lower bounds of the prediction intervals.
    :param ndarray y_pred_upper: upper bounds of the prediction intervals.
    :param ndarray, optional X: abscisse vector.
    :param ndarray, optional y_pred: predicted values.
    :param kwargs: plot configuration parameters.
    """

    # Figure configuration
    if "figsize" in kwargs.keys():
        figsize = kwargs["figsize"]
    else:
        figsize = (15, 6)
    if "loc" not in kwargs.keys():
        loc = kwargs["loc"]
    else:
        loc = "upper left"
    plt.figure(figsize=figsize)

    # Custom matplotlib style sheet
    matplotlib.rcParams.update(custom_rc_params)

    if X is None:
        X = np.arange(len(y_true))
    elif sort_X:
        sorted_idx = np.argsort(X)
        X = X[sorted_idx]
        y_true = y_true[sorted_idx]

        if y_pred is not None:
            y_pred = y_pred[sorted_idx]

        y_pred_lower = y_pred_lower[sorted_idx]
        y_pred_upper = y_pred_upper[sorted_idx]

    if y_pred_upper is None or y_pred_lower is None:
        miscoverage = np.array([False for _ in range(len(y_true))])
    else:
        miscoverage = (y_true > y_pred_upper) | (y_true < y_pred_lower)

    if X is not None:
        label = "Observation" if y_pred_upper is None else "Observation (inside PI)"
        plt.plot(
            X[~miscoverage],
            y_true[~miscoverage],
            "darkgreen",
            marker="X",
            markersize=2,
            linewidth=0,
            label=label,
            zorder=20,
        )

    if X is not None:
        label = "Observation" if y_pred_upper is None else "Observation (outside PI)"
        plt.plot(
            X[miscoverage],
            y_true[miscoverage],
            color="red",
            marker="o",
            markersize=2,
            linewidth=0,
            label=label,
            zorder=20,
        )

    if (y_pred_upper is not None) and (y_pred_lower is not None) and (X is not None):
        plt.plot(X, y_pred_upper, "--", color="blue", linewidth=1, alpha=0.7)
        plt.plot(X, y_pred_lower, "--", color="blue", linewidth=1, alpha=0.7)
        plt.fill_between(
            x=X,
            y1=y_pred_upper,
            y2=y_pred_lower,
            alpha=0.2,
            fc="b",
            ec="None",
            label="Prediction Interval",
        )

    if y_pred is not None:
        plt.plot(X, y_pred, color="k", label="Prediction")

    plt.xlabel("X")
    plt.ylabel("Y")

    if "loc" not in kwargs.keys():
        loc = "upper left"
    else:
        loc = kwargs["loc"]

    plt.legend(loc=loc)
    if save_path:
        plt.savefig(f"{save_path}", format="pdf")
    else:
        plt.show()


def plot_sorted_pi(
    y_true: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    X: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    **kwargs,
) -> None:
    """Plot prediction intervals in an ordered fashion (lowest to largest width)
    showing the upper and lower bounds for each prediction.

    :param ndarray y_true: label true values.
    :param ndarray y_pred_lower: lower bounds of the prediction intervals.
    :param ndarray y_pred_upper: upper bounds of the prediction intervals.
    :param ndarray, optional X: abscisse vector.
    :param ndarray, optional y_pred: predicted values.
    :param kwargs: plot configuration parameters.
    """

    if y_pred is None:
        y_pred = (y_pred_upper + y_pred_lower) / 2

    width = np.abs(y_pred_upper - y_pred_lower)  # type: ignore
    sorted_order = np.argsort(width)

    # Figure configuration
    if "figsize" in kwargs.keys():
        figsize = kwargs["figsize"]
    else:
        figsize = (15, 6)
    # if "loc" not in kwargs.keys():
    #     loc = kwargs["loc"]
    # else:
    #     loc = "upper left"
    plt.figure(figsize=figsize)

    if X is None:
        X = np.arange(len(y_pred_lower))

    # True values
    plt.plot(
        X,
        y_pred[sorted_order] - y_pred[sorted_order],
        color="black",
        markersize=2,
        zorder=20,
        label="Prediction",
    )

    miscoverage = (y_true > y_pred_upper) | (y_true < y_pred_lower)
    miscoverage = miscoverage[sorted_order]

    # True values
    plt.plot(
        X[~miscoverage],
        y_true[sorted_order][~miscoverage] - y_pred[sorted_order][~miscoverage],
        color="darkgreen",
        marker="o",
        markersize=2,
        linewidth=0,
        zorder=20,
        label="Observation (inside PI)",
    )

    plt.plot(
        X[miscoverage],
        y_true[sorted_order][miscoverage] - y_pred[sorted_order][miscoverage],
        color="red",
        marker="o",
        markersize=2,
        linewidth=0,
        zorder=20,
        label="Observation (outside PI)",
    )

    # PI Lower bound
    plt.plot(
        X,
        y_pred_lower[sorted_order] - y_pred[sorted_order],
        "--",
        label="Prediction Interval Bounds",
        color="blue",
        linewidth=1,
        alpha=0.7,
    )

    # PI upper bound
    plt.plot(
        X,
        y_pred_upper[sorted_order] - y_pred[sorted_order],
        "--",
        color="blue",
        linewidth=1,
        alpha=0.7,
    )

    plt.legend()

    plt.show()


#
# ========================= Metrics =========================
#

# TODO: add comments, explain their formula, how to interpret.
# TODO: add docstrings


def average_coverage(y_true, y_pred_lower, y_pred_upper):
    """Compute average coverage on several prediction intervals.

    Given a prediction interval i defined by its lower bound y_pred_lower[i] and upper bound y_pred_upper[i], the i-th coverage is:

        * c[i] = 1 if y_pred_lower[i]  <= y_true[i] <= bound y_pred_upper[i]
        * c[i] = 0 otherwise

    With N the number of example, the average coverage is :math:`1/N \sum_{i=1}^{N} c(i)`.

    :param ndarray y_true: label true values.
    :param ndarray y_pred_lower: lower bounds of the prediction intervals.
    :param ndarray y_pred_upper: upper bounds of the prediction intervals.

    :returns: Average coverage
    :rtype: float
    """
    return ((y_true >= y_pred_lower) & (y_true <= y_pred_upper)).mean()


def ace(y_true, y_pred_lower, y_pred_upper, alpha):
    """Compte the Average Coverage Error (ACE).

    :param ndarray y_true: label true values.
    :param ndarray y_pred_lower: lower bounds of the prediction intervals.
    :param ndarray y_pred_upper: upper bounds of the prediction intervals.
    :param float alpha: significance level (max miscoverage target).

    .. NOTE::
        The ACE is the distance between the nominal coverage :math:`1-alpha` and the empirical average coverage :math:`AC` such that :math:`ACE = AC - (1-alpha)`.
        If the ACE is strictly negative, the prediction intervals are marginally undercovering. If the ACE is strictly positive, the prediction intervals are maginally conservative.

    :returns: the average coverage error (ACE).
    :rtype: float
    """
    cov = average_coverage(y_true, y_pred_lower, y_pred_upper)
    return cov - (1 - alpha)


def sharpness(y_pred_lower, y_pred_upper):
    """Compute the average absolute width of the prediction intervals.

    :param ndarray y_pred_lower: lower bounds of the prediction intervals.
    :param ndarray y_pred_upper: upper bounds of the prediction intervals.

    :returns: average absolute width of the prediction intervals.
    :rtype: float
    """
    return (np.abs(y_pred_upper - y_pred_lower)).mean()

"""
This module implements utility functions.
"""
import numpy as np
from typing import Iterable, Tuple, Optional
import matplotlib.pyplot as plt
import sys

EPSILON = sys.float_info.min  # small value to avoid underflow


def check_alpha_calib(alpha: float, n: int, complement_check: bool = False):
    """Check if the value of alpha is consistent with the size of calibration
    set.
    The quantile order are inflated by a factor (1+1/n) and have to be in
    the interval [0,1]. From this, we derive the condition:
        - alpha => 1/(n+1) given that 0 <= (1-alpha)*(1+1/n) <= 1
    If complement_check is set, we consider an additional condition:
        - alpha <= n/(n+1) given that 0 <= alpha*(1+1/n) <= 1

    Args:
        alpha: target quantile order.
        n: size of the calibration dataset.
        complement_check: complementary check to compute the alpha*(1+1/n)-th
                          quantile, required by some methods such as jk+.

    """
    if alpha < 1 / (n + 1):
        raise ValueError(
            f"Alpha is too small: (alpha={alpha}, n={n}) {alpha} < {1/(n+1)}. "
            + "Increase alpha or the size of the calibration set."
        )
    if complement_check and alpha > n / (n + 1):
        raise ValueError(
            f"Alpha is too large: (alpha={alpha}, n={n}) {alpha} < {n/(n+1)}. "
            + "Decrease alpha or increase the size of the calibration set."
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

    Args:
        n [int]: size of the calibration dataset.
        two_sided_conformalization [bool]: alpha threshold for two-sided quantile of jackknife+ (Barber et al 2021).

    Raises:
        ValueError -- must have integer n, boolean two_sided_conformalization and n>=1

    Returns:
        Optional[Tuple[float, float]] -- lower and upper bounds for alpha (miscoverage probability)
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


def quantile(a, q, w=None):
    """Estimate the q-th empirical weighted quantiles.

    Args:
        a: vector of n samples
        q: target quantile order. Must be in the open interval [0, 1].
        w: vector of size n
           By default, w is None and equal weights = 1/m are associated.
    Returns:
        Weighted empirical quantiles
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
        return np.quantile(a, q=q, interpolation="higher")
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


"""
========================= Aggregation functions =========================
"""


def agg_list(a: Iterable):
    try:
        return np.concatenate(a, axis=0)
    except ValueError:
        return None


def agg_func(a: Iterable):
    try:
        return np.mean(a, axis=0)
    except TypeError:
        return None


"""
========================= Visualization =========================
"""


def plot_prediction_interval(
    y_true: np.array,
    y_pred_lower: np.array,
    y_pred_upper: np.array,
    X: np.array = None,
    y_pred: np.array = None,
    save_path: str = None,
    sort_X: bool = False,
    **kwargs,
) -> None:
    """Plot prediction intervals whose bounds are given by y_pred_lower
    and y_pred_upper.
    True values and point estimates are also plotted if given as argument.

    Args:
        y_true: label true values.
        y_pred_lower: lower bounds of the prediction interval.
        y_pred_upper: upper bounds of the prediction interval.
        X <optionnal>: abscisse vector.
        y_pred <optionnal>: predicted values.
        kwargs: plot parameters.
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

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["ytick.labelsize"] = 15
    plt.rcParams["xtick.labelsize"] = 15
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["legend.fontsize"] = 16

    if X is None:
        X = np.arange(len(y_true))
    elif sort_X:
        sorted_idx = np.argsort(X)
        X = X[sorted_idx]
        y_true = y_true[sorted_idx]
        y_pred = y_pred[sorted_idx]
        y_pred_lower = y_pred_lower[sorted_idx]
        y_pred_upper = y_pred_upper[sorted_idx]

    if y_pred_upper is None or y_pred_lower is None:
        miscoverage = np.array([False for _ in range(len(y_true))])
    else:
        miscoverage = (y_true > y_pred_upper) | (y_true < y_pred_lower)

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
    if y_pred_upper is not None and y_pred_lower is not None:
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
    y_true: np.array,
    y_pred_lower: np.array,
    y_pred_upper: np.array,
    X: np.array = None,
    y_pred: np.array = None,
    **kwargs,
) -> None:
    """Plot prediction intervals in an ordered fashion (lowest to largest width)
    showing the upper and lower bounds for each prediction.
    Args:
        y_true: label true values.
        y_pred_lower: lower bounds of the prediction interval.
        y_pred_upper: upper bounds of the prediction interval.
        X <optionnal>: abscisse vector.
        y_pred <optionnal>: predicted values.
        kwargs: plot parameters.
    """

    if y_pred is None:
        y_pred = (y_pred_upper + y_pred_lower) / 2

    width = np.abs(y_pred_upper - y_pred_lower)
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

    misscoverage = (y_true > y_pred_upper) | (y_true < y_pred_lower)
    misscoverage = misscoverage[sorted_order]

    # True values
    plt.plot(
        X[~misscoverage],
        y_true[sorted_order][~misscoverage] - y_pred[sorted_order][~misscoverage],
        color="darkgreen",
        marker="o",
        markersize=2,
        linewidth=0,
        zorder=20,
        label="Observation (inside PI)",
    )

    plt.plot(
        X[misscoverage],
        y_true[sorted_order][misscoverage] - y_pred[sorted_order][misscoverage],
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


"""
========================= Metrics =========================
"""


def average_coverage(y_true, y_pred_lower, y_pred_upper):
    return ((y_true >= y_pred_lower) & (y_true <= y_pred_upper)).mean()


def ace(y_true, y_pred_lower, y_pred_upper, alpha):
    cov = average_coverage(y_true, y_pred_lower, y_pred_upper)
    return cov - (1 - alpha)


def sharpness(y_pred_lower, y_pred_upper):
    return (np.abs(y_pred_upper - y_pred_lower)).mean()

"""
This module implements utility functions.
"""
import numpy as np
from typing import Iterable
import matplotlib.pyplot as plt
import sys

EPSILON = sys.float_info.min  # small value to avoid underflow


def w_quantile(a, alpha, w=None):
    """Weighted empirical quantile.

    Args:
        a: array of samples
        alpha: Quantile to compute, which must be between 0 and 1
        w: matrix of weight.
           If w is None, np.quantile is returned. Otherwise, w columns
           must be of same size as argument lenght of a
    Returns:
        weighted empirical quantile
    """
    if alpha < 0 or alpha > 1:  # Sanity check
        raise ValueError("Alpha must land between 0 and 1.")
    # Regular Empirical Quantile
    if w is None:
        return np.quantile(a, alpha)
    # Empirical Weighted Quantile
    sorted_idxs = np.argsort(a)
    sorted_cumsum_w = np.cumsum(w[:, sorted_idxs], axis=1)
    weighted_quantile_idxs = [
        sorted_idxs[sorted_cumsum_w[i] >= alpha][0] for i in range(len(w))
    ]
    return np.array([a[idx] for idx in weighted_quantile_idxs])


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

plt.rcParams["figure.figsize"] = [8, 8]
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 15


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

    label = (
        "Observation" if y_pred_upper is None else "Observation (inside PI)"
    )
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

    label = (
        "Observation" if y_pred_upper is None else "Observation (outside PI)"
    )
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
        y_true[sorted_order][~misscoverage]
        - y_pred[sorted_order][~misscoverage],
        color="darkgreen",
        marker="o",
        markersize=2,
        linewidth=0,
        zorder=20,
        label="Observation (inside PI)",
    )

    plt.plot(
        X[misscoverage],
        y_true[sorted_order][misscoverage]
        - y_pred[sorted_order][misscoverage],
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

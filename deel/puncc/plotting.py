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
This module provides plotting functions for conformal prediction.
"""
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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


def plot_prediction_intervals(
    y_true: np.ndarray,
    y_pred_lower: np.ndarray,
    y_pred_upper: np.ndarray,
    X: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    ax: matplotlib.axes.Axes = None,
    **fig_kw,
) -> matplotlib.axes.Axes:
    """Plot prediction intervals whose bounds are given by `y_pred_lower` and
    `y_pred_upper`. True and predicted (if provided) point values are also
    displayed.

    :param ndarray y_true: true output values.
    :param ndarray y_pred_lower: lower bounds of the prediction intervals.
    :param ndarray y_pred_upper: upper bounds of the prediction intervals.
    :param ndarray, optional X: abscisse vector.
    :param ndarray, optional y_pred: predicted values.
    :param matplotlib.axes.Axes, optional ax: plot using the provided axis.
        Otherwise, a new figure is created and the corresponding axis is
        returned as output.
    :param fig_kw: all additional keyword arguments are passed to the
        pyplot.figure call.

    :returns: updated axis if `ax` provided. Otherwise a new figure is created
        and the corresponding axis is returned as output
    :rtype: matplotlib.axes.Axes

    Example 1::

        import numpy as np
        from sklearn.datasets import make_regression
        from deel.puncc.plotting import plot_prediction_intervals

        np.random.seed(0)

        # Generate a random regression problem
        X, y = make_regression(n_samples=100, n_features=100, n_informative=20,
                        random_state=0, shuffle=False)

        # Generate dummy point predictions
        y_pred = y + np.random.normal(loc=0.0, scale=100, size=len(y))
        # Generate dummy interval predictions
        y_pred_lower = y_pred - 120 - np.random.normal(loc=0.0, scale=20.0, size=len(y))
        y_pred_upper = y_pred + 120 + np.random.normal(loc=0.0, scale=20.0, size=len(y))

        # Plot the prediction interval. The optional argument "y_pred" is
        # provided and the prediction is displayed.
        # Additionnaly, two keyword arguments "figsize" and "loc" are given to
        # configure the figure.
        ax = plot_prediction_intervals(
        y_true=y,
        y_pred=y_pred,
        y_pred_lower=y_pred_lower,
        y_pred_upper=y_pred_upper,
        figsize=(20,10),
        loc="best")

    Example 2::

        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.datasets import make_regression
        from deel.puncc.plotting import plot_prediction_intervals

        np.random.seed(0)

        # Generate a random regression problem
        X, y = make_regression(n_samples=100, n_features=1, noise=20,
                        random_state=0, shuffle=False)

        # Generate dummy point predictions
        y_pred = y + np.random.normal(loc=0.0, scale=10, size=len(y))

        # Generate two dummy interval predictions
        ## First
        y_pred_lower1 = y_pred - 15
        y_pred_upper1 = y_pred + 15
        # Second
        y_pred_lower2 = y_pred - 10 - np.random.normal(loc=0.0, scale=2.0, size=len(y))
        y_pred_upper2 = y_pred + 10 + np.random.normal(loc=0.0, scale=2.0, size=len(y))

        # Create a grid of figures
        fig, ax = plt.subplots(nrows=2, figsize=(20,10))

        # Plot first figure by providing the axis "ax[0]"
        plot_prediction_intervals(
        X = X[:,0],
        y_true=y,
        y_pred=y_pred,
        y_pred_lower=y_pred_lower1,
        y_pred_upper=y_pred_upper1,
        ax=ax[0])

        # Plot second figure by providing the axis "ax[1]"
        plot_prediction_intervals(
        X = X[:,0],
        y_true=y,
        y_pred=y_pred,
        y_pred_lower=y_pred_lower2,
        y_pred_upper=y_pred_upper2,
        ax=ax[1])

    """

    # Figure size configuration
    if "figsize" in fig_kw.keys():
        figsize = fig_kw["figsize"]
    else:
        figsize = (15, 6)

    # Create new figure and ax if None provided
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        # Before changing rcparams, save old config to restablish is later
        restablish_rcparams = True
        current_rcparams = dict(matplotlib.rcParams)
        # Custom matplotlib style sheet
        matplotlib.rcParams.update(custom_rc_params)
    else:
        restablish_rcparams = False

    # X handler
    if X is None:
        X = np.arange(len(y_true))
    else:
        sorted_idx = np.argsort(X)
        X = X[sorted_idx]
        y_true = y_true[sorted_idx]

        if y_pred is not None:
            y_pred = y_pred[sorted_idx]

        y_pred_lower = y_pred_lower[sorted_idx]
        y_pred_upper = y_pred_upper[sorted_idx]

    # Checks if user is plotting an interval
    plot_interval = False
    if (y_pred_upper is not None) and (y_pred_lower is not None):
        plot_interval = True

    # Coverage count
    if plot_interval:
        miscoverage = (y_true > y_pred_upper) | (y_true < y_pred_lower)
    else:  # No interval given, so no miscoverage
        miscoverage = np.array([False for _ in range(len(y_true))])

    # plot observations inside PI
    label = "Observation (inside PI)" if plot_interval else "Observation"
    ax.plot(
        X[~miscoverage],
        y_true[~miscoverage],
        "darkgreen",
        marker="o",
        markersize=4,
        linewidth=0,
        label=label,
        zorder=20,
    )

    if plot_interval:
        # Plot observations outside PI
        label = "Observation (outside PI)"

        ax.plot(
            X[miscoverage],
            y_true[miscoverage],
            color="red",
            marker="o",
            markersize=4,
            linewidth=0,
            label=label,
            zorder=20,
        )

        # plot interval
        ax.plot(X, y_pred_upper, "--", color="blue", linewidth=1, alpha=0.7)
        ax.plot(X, y_pred_lower, "--", color="blue", linewidth=1, alpha=0.7)
        ax.fill_between(
            x=X,
            y1=y_pred_upper,  # type: ignore
            y2=y_pred_lower,  # type: ignore
            alpha=0.2,
            fc="b",
            ec="None",
            label="Prediction Interval",
        )

    if y_pred is not None:
        ax.plot(X, y_pred, color="k", label="Prediction")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if "loc" not in fig_kw.keys():
        loc = "upper left"
    else:
        loc = fig_kw["loc"]

    ax.legend(loc=loc)

    # Set x limits
    int_size = X[-1] - X[0]
    ax.set_xlim(X[0] - int_size * 0.01, X[-1] + int_size * 0.01)

    # restablish rcparams
    if restablish_rcparams:
        matplotlib.rcParams.update(current_rcparams)

    return ax

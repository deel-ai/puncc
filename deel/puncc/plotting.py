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
            y1=y_pred_upper,  # type: ignore
            y2=y_pred_lower,  # type: ignore
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

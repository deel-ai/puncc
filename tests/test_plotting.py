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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from deel.puncc.plotting import draw_bounding_box
from deel.puncc.plotting import plot_prediction_intervals


def test_plot_prediction_intervals_sorts_x_and_adds_coverage_to_title():
    fig, ax = plt.subplots()
    x = np.array([2, 0, 1])
    y_true = np.array([3.0, 0.0, 1.0])
    y_pred = np.array([2.5, 0.0, 1.0])
    y_pred_lower = np.array([2.0, -0.5, 0.5])
    y_pred_upper = np.array([2.8, 0.5, 1.5])

    returned_ax = plot_prediction_intervals(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_lower=y_pred_lower,
        y_pred_upper=y_pred_upper,
        X=x,
        ax=ax,
        title="Demo",
    )

    np.testing.assert_array_equal(returned_ax.lines[-1].get_xdata(), np.array([0, 1, 2]))
    assert returned_ax is ax
    assert returned_ax.get_title() == "Demo | coverage=0.667"
    plt.close(fig)


def test_plot_prediction_intervals_without_interval_uses_custom_title():
    fig, ax = plt.subplots()

    returned_ax = plot_prediction_intervals(
        y_true=np.array([1.0, 2.0]),
        y_pred=np.array([1.5, 2.5]),
        y_pred_lower=None,
        y_pred_upper=None,
        ax=ax,
        title="Points Only",
    )

    assert returned_ax is ax
    assert returned_ax.get_title() == "Points Only"
    plt.close(fig)


def test_draw_bounding_box_requires_image_or_path():
    with pytest.raises(ValueError, match="Either image or image_path must be provided"):
        draw_bounding_box()


def test_draw_bounding_box_adds_box_and_deduplicates_legend():
    image = Image.new("RGB", (20, 20), color="white")
    image_before = image.tobytes()

    image = draw_bounding_box(
        box=(1, 1, 10, 10),
        label="cat",
        image=image,
        color="red",
        legend="prediction",
    )
    image = draw_bounding_box(
        box=(5, 5, 15, 15),
        label="cat",
        image=image,
        color="red",
        legend="prediction",
    )

    assert image.size == (20, 20)
    assert image.legends == ["prediction"]
    assert len(image.custom_lines) == 1
    assert image.tobytes() != image_before

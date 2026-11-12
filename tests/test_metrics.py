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

import numpy as np
import pytest

from deel.puncc import ops
from deel.puncc.metrics import (
    classification_mean_coverage,
    classification_mean_size,
    iou,
    object_detection_mean_area,
    object_detection_mean_coverage,
    regression_ace,
    regression_mean_coverage,
    regression_sharpness,
)
from tests._utils import to_numpy


def test_classification_metrics():
    y_true = np.array([0, 1, 1])
    prediction_sets = ([0, 2], [1], [])

    assert classification_mean_coverage(y_true, prediction_sets) == pytest.approx(2 / 3)

    assert classification_mean_size(prediction_sets) == pytest.approx(1.0)


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.0, 2.5, 2.0])
    upper = np.array([1.0, 3.0, 4.0])

    assert regression_mean_coverage(y_true, lower, upper) == pytest.approx(2 / 3)

    assert regression_ace(y_true, lower, upper, alpha=0.2) == pytest.approx(2 / 3 - 0.8)

    np.testing.assert_allclose(
        to_numpy(regression_sharpness(lower, upper)),
        7 / 6,
    )


def test_object_detection_metrics():
    predicted_boxes = np.array([[0, 0, 4, 4], [1, 1, 2, 2]])
    true_boxes = np.array([[1, 1, 3, 3], [0, 0, 3, 3]])

    np.testing.assert_allclose(
        to_numpy(object_detection_mean_coverage(predicted_boxes, true_boxes)),
        0.5,
    )

    np.testing.assert_allclose(
        to_numpy(object_detection_mean_area(predicted_boxes)),
        8.5,
    )


def test_iou_pairwise_matrix():
    boxes1 = np.array([[0, 0, 2, 2], [5, 5, 6, 6]])
    boxes2 = np.array([[1, 1, 3, 3], [5, 5, 6, 6]])

    expected = np.array(
        [
            [1 / 7, 0.0],
            [0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(
        to_numpy(iou(boxes1, boxes2)),
        expected,
    )

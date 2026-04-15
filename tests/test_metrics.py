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

from deel.puncc.metrics import classification_classwise_coverage
from deel.puncc.metrics import classification_classwise_size
from deel.puncc.metrics import classification_mean_coverage
from deel.puncc.metrics import classification_mean_size
from deel.puncc.metrics import iou
from deel.puncc.metrics import object_detection_mean_area
from deel.puncc.metrics import object_detection_mean_coverage
from deel.puncc.metrics import regression_ace
from deel.puncc.metrics import regression_mean_coverage
from deel.puncc.metrics import regression_sharpness


def test_classification_metrics_compute_expected_values():
    y_true = np.array([0, 1, 1])
    set_pred = ([0, 2], [1], [])

    assert classification_mean_coverage(y_true, set_pred) == 2 / 3
    assert classification_mean_size(set_pred) == 1.0
    np.testing.assert_allclose(
        classification_classwise_coverage(y_true, set_pred, n_classes=2),
        np.array([1.0, 0.5]),
    )
    np.testing.assert_allclose(
        classification_classwise_size(y_true, set_pred, n_classes=2),
        np.array([2.0, 0.5]),
    )


def test_classification_classwise_metrics_reject_missing_class():
    y_true = np.array([0, 0])
    set_pred = ([0], [0])

    with pytest.raises(ValueError, match="No instances found for class 1"):
        classification_classwise_coverage(y_true, set_pred, n_classes=2)

    with pytest.raises(ValueError, match="No instances found for class 1"):
        classification_classwise_size(y_true, set_pred, n_classes=2)


def test_regression_metrics_compute_expected_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred_lower = np.array([0.0, 2.5, 2.0])
    y_pred_upper = np.array([1.0, 3.0, 4.0])

    assert regression_mean_coverage(y_true, y_pred_lower, y_pred_upper) == 2 / 3
    assert regression_ace(y_true, y_pred_lower, y_pred_upper, alpha=0.2) == pytest.approx(
        -0.13333333333333341
    )
    assert regression_sharpness(y_pred_lower, y_pred_upper) == pytest.approx(
        7 / 6
    )


def test_object_detection_metrics_compute_expected_values():
    y_pred_outer = np.array([[0, 0, 4, 4], [1, 1, 2, 2]])
    y_true = np.array([[1, 1, 3, 3], [0, 0, 3, 3]])

    assert object_detection_mean_coverage(y_pred_outer, y_true) == 0.5
    assert object_detection_mean_area(y_pred_outer) == 8.5


def test_iou_returns_pairwise_overlap_matrix():
    bboxes1 = np.array([[0, 0, 2, 2], [5, 5, 6, 6]])
    bboxes2 = np.array([[1, 1, 3, 3], [5, 5, 6, 6]])

    expected = np.array([[2 / 7, 0.0], [0.0, 1.0]])

    np.testing.assert_allclose(iou(bboxes1, bboxes2), expected)

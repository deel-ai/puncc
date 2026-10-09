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
from deel.puncc.od.base import Box, ODPrediction, ODTarget
from deel.puncc.od.matching import AssignmentStrategy, HungarianMatcher
from tests._utils import tensor, to_numpy


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("additive", [0.5, 1.5, 3.5, 6.5]),
        ("multiplicative", [0.0, 0.0, 4.0, 8.0]),
    ],
)
def test_box_extension_modes(mode, expected):
    box = Box(tensor([1.0, 2.0, 3.0, 6.0], "float32"))

    extended = box.extend(0.5, mode=mode)

    np.testing.assert_allclose(to_numpy(extended.xyxy), expected)

    # Non-inplace extension must not modify the original box.
    np.testing.assert_allclose(
        to_numpy(box.xyxy),
        [1.0, 2.0, 3.0, 6.0],
    )


def test_box_extension_rejects_invalid_mode():
    box = Box(tensor([0.0, 0.0, 2.0, 2.0], "float32"))

    with pytest.raises(ValueError):
        box.extend(0.5, mode="invalid")


def test_hungarian_matcher():
    costs = np.array(
        [
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )

    matcher = HungarianMatcher()

    assert matcher.match(tensor(costs, "float32")) == [(0, 1), (1, 0)]

    # Only the first source has a valid assignment.
    valid_mask = np.array(
        [
            [False, True],
            [False, False],
        ]
    )

    assert matcher.match(tensor(costs, "float32"), tensor(valid_mask, "bool")) == [(0, 1)]


def test_assignment_strategy_filters_by_iou():
    predictions = ODPrediction(
        boxes=tensor(np.array(
            [
                [0.0, 0.0, 2.0, 2.0],
                [9.0, 9.0, 10.0, 10.0],
            ],
            dtype=np.float32,
        ), "float32"),
        class_scores=tensor(np.array(
            [[0.9, 0.1], [0.1, 0.9]],
            dtype=np.float32,
        ), "float32"),
        confidences=tensor(np.array([0.9, 0.8], dtype=np.float32), "float32"),
    )

    targets = ODTarget(
        boxes=tensor(np.array(
            [
                [0.0, 0.0, 2.0, 2.0],
                [5.0, 5.0, 7.0, 7.0],
                [20.0, 20.0, 22.0, 22.0],
            ],
            dtype=np.float32,
        ), "float32"),
        labels=tensor(np.array([0, 1, 0], dtype=np.int32), "int32"),
    )

    strategy = AssignmentStrategy(
        matcher=HungarianMatcher(),
        iou_threshold=0.5,
    )

    assignment = strategy.assign(predictions, targets)

    assert assignment.matched_pairs == [(0, 0)]
    assert assignment.unmatched_true_indices() == [1, 2]
    assert assignment.unmatched_pred_indices() == [1]

    aligned_pred, aligned_true = assignment.align_prediction_and_target(
        predictions,
        targets,
    )

    np.testing.assert_allclose(
        to_numpy(aligned_pred.boxes),
        [[0.0, 0.0, 2.0, 2.0]],
    )
    np.testing.assert_allclose(
        to_numpy(aligned_true.boxes),
        [[0.0, 0.0, 2.0, 2.0]],
    )


def test_assignment_strategy_respects_classes():
    predictions = ODPrediction(
        boxes=tensor(np.array(
            [
                [0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 2.0, 2.0],
            ],
            dtype=np.float32,
        ), "float32"),
        class_scores=tensor(np.array(
            [[0.9, 0.1], [0.1, 0.9]],
            dtype=np.float32,
        ), "float32"),
        confidences=tensor(np.array([0.9, 0.9], dtype=np.float32), "float32"),
    )

    targets = ODTarget(
        boxes=tensor(np.array(
            [[0.0, 0.0, 2.0, 2.0]],
            dtype=np.float32,
        ), "float32"),
        labels=tensor(np.array([1], dtype=np.int32), "int32"),
    )

    strategy = AssignmentStrategy(
        matcher=HungarianMatcher(),
        class_matching=True,
    )

    assignment = strategy.assign(predictions, targets)

    # Both predicted boxes have IoU=1, but only the second
    # predicts the correct class.
    assert assignment.matched_pairs == [(0, 1)]
    assert assignment.unmatched_pred_indices() == [0]


def test_assignment_strategy_handles_empty_predictions():
    predictions = ODPrediction(
        boxes=tensor(np.empty((0, 4), dtype=np.float32), "float32"),
        class_scores=tensor(np.empty((0, 2), dtype=np.float32), "float32"),
        confidences=tensor(np.empty((0,), dtype=np.float32), "float32"),
    )

    targets = ODTarget(
        boxes=tensor(np.array(
            [[0.0, 0.0, 2.0, 2.0]],
            dtype=np.float32,
        ), "float32"),
        labels=tensor(np.array([0], dtype=np.int32), "int32"),
    )

    assignment = AssignmentStrategy(
        matcher=HungarianMatcher(),
    ).assign(predictions, targets)

    assert assignment.matched_pairs == []
    assert assignment.unmatched_true_indices() == [0]
    assert assignment.unmatched_pred_indices() == []

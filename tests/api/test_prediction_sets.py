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
import unittest

import numpy as np
import pytest

from deel.puncc.api.prediction_sets import scaled_bbox


class prediction_sets_check(unittest.TestCase):
    """
    Test scaled_bbox prediction sets.
    """

    def setUp(self):
        n = 10
        self.bbox_pred = np.array([[1, 2, 3, 4]])
        self.scores_quantile = np.array([0.1, 0.2, 0.3, 0.4])
        self.expected_lo = np.array([[1.2, 2.4, 2.4, 3.2]])
        self.expected_hi = np.array([[0.8, 1.6, 3.6, 4.8]])

        self.bbox_pred_n = np.repeat(self.bbox_pred, n, axis=0)
        self.expected_lo_n = np.repeat(self.expected_lo, n, axis=0)
        self.expected_hi_n = np.repeat(self.expected_hi, n, axis=0)

    def test_scaled_bbox(self):
        print(
            scaled_bbox(self.bbox_pred, self.scores_quantile),
            (self.expected_lo, self.expected_hi),
        )
        assert np.array_equal(
            scaled_bbox(self.bbox_pred, self.scores_quantile),
            (self.expected_lo, self.expected_hi),
        )
        assert np.array_equal(
            scaled_bbox(self.bbox_pred_n, self.scores_quantile),
            (self.expected_lo_n, self.expected_hi_n),
        )

        # Unsupported data type for Y_pred
        with pytest.raises(TypeError):
            scaled_bbox([1, 2, 3, 4], self.scores_quantile)

        # Y_pred contains less than 4 bbox coordinates
        Y_pred = np.array([[0, 0, 2], [1, 3, 3]])
        with pytest.raises(RuntimeError):
            scaled_bbox(Y_pred, self.scores_quantile)

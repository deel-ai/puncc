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

from deel.puncc.api.nonconformity_scores import scaled_bbox_difference


class nonconformity_scores_check(unittest.TestCase):
    def setUp(self):
        n = 10

        self.bbox_pred = np.array([[1, 2, 3, 4]])
        self.bbox_true = np.array([[2, 3, 4, 5]])
        self.expected = np.array([[-0.5, -0.5, -0.5, -0.5]])

        self.bbox_pred_n = np.repeat(self.bbox_pred, n, axis=0)
        self.bbox_true_n = np.repeat(self.bbox_true, n, axis=0)
        self.expected_n = np.repeat(self.expected, n, axis=0)

    def test_scaled_bbox_difference(self):
        result = scaled_bbox_difference(self.bbox_pred, self.bbox_true)
        assert np.array_equal(result, self.expected)

        result_n = scaled_bbox_difference(self.bbox_pred_n, self.bbox_true_n)
        assert np.array_equal(result_n, self.expected_n)

        with pytest.raises(TypeError):
            scaled_bbox_difference("Not an ndarray", self.bbox_true_n)

        with pytest.raises(RuntimeError):
            scaled_bbox_difference(np.array([[1, 2, 3]]), self.bbox_true_n)

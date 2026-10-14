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

from tests._utils import tensor, to_numpy

from deel.puncc.nonconformity_scores import scaled_bbox_difference


@pytest.mark.parametrize("n_samples", [1, 10])
def test_scaled_bbox_difference(n_samples):
    bbox_pred = np.tile(
        np.array([[1.0, 2.0, 3.0, 4.0]]),
        (n_samples, 1),
    )
    bbox_pred = tensor(bbox_pred, "float32")
    bbox_true = np.tile(
        np.array([[2.0, 3.0, 4.0, 5.0]]),
        (n_samples, 1),
    )
    bbox_true = tensor(bbox_true, "float32")

    expected = np.tile(
        np.array([[-0.5, -0.5, 0.5, 0.5]]),
        (n_samples, 1),
    )

    score_function = scaled_bbox_difference()
    result = score_function(bbox_pred, bbox_true)

    assert result.shape == (n_samples, 4)
    np.testing.assert_allclose(to_numpy(result), expected)

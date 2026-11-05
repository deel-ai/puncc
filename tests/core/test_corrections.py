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

from deel.puncc.corrections import bonferroni, weighted_bonferroni
from tests._utils import tensor, to_numpy


def test_bonferroni_returns_alpha_for_single_variable():
    correction = bonferroni(nvars=1)

    assert correction(0.2) == 0.2


def test_bonferroni_splits_alpha_across_variables():
    correction = bonferroni(nvars=4)

    np.testing.assert_allclose(
        to_numpy(correction(0.2)),
        np.full(4, 0.05),
    )


@pytest.mark.parametrize("nvars", [0, -1])
def test_bonferroni_rejects_invalid_nvars(nvars):
    with pytest.raises(ValueError, match="nvars must be a positive integer"):
        bonferroni(nvars=nvars)


@pytest.mark.parametrize(
    "weights",
    [
        np.array([0.2, 0.3, 0.5]),
        np.array([2.0, 3.0, 5.0]),
    ],
)
def test_weighted_bonferroni_scales_by_normalized_weights(weights):
    correction = weighted_bonferroni(tensor(weights, "float32"))

    np.testing.assert_allclose(
        to_numpy(correction(0.4)),
        np.array([0.08, 0.12, 0.20]),
    )

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

from deel.puncc.api.utils import generate_leverage_func


def test_generate_leverage_func_matches_quadratic_form():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    leverage_func = generate_leverage_func(X)
    x_test = np.array([[1.0, 0.0], [0.5, 0.5]])

    scores = leverage_func(x_test)
    expected = np.einsum("ij,jk,ik->i", x_test, np.linalg.inv(X.T @ X), x_test)

    np.testing.assert_allclose(scores, expected)


def test_generate_leverage_func_training_scores_sum_to_rank():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    leverage_func = generate_leverage_func(X)

    scores = leverage_func(X)
    expected = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)

    np.testing.assert_allclose(scores, expected)
    np.testing.assert_allclose(np.sum(scores), X.shape[1])

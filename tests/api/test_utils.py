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
import pandas as pd
import tensorflow as tf

from deel.puncc.api.utils import alpha_calib_check
from deel.puncc.api.utils import quantile
from deel.puncc.api.utils import supported_types_check

np.random.seed(0)


class DataTypeStructureCheck(unittest.TestCase):
    def setUp(self):
        self.y_pred_np = np.random.random_sample(size=(1000, 5))
        self.y_pred_df = pd.DataFrame(self.y_pred_np)
        self.y_pred_tensor = tf.random.uniform(shape=[100, 5])

    def test_ypred_type(self):
        supported_types_check(self.y_pred_np)
        supported_types_check(self.y_pred_df)
        supported_types_check(self.y_pred_tensor)

    def test_other_type(self):
        with self.assertRaises(TypeError):
            supported_types_check((1, 2, 3))

    def test_type_consistency(self):

        supported_types_check(self.y_pred_np, self.y_pred_np)
        supported_types_check(self.y_pred_df, self.y_pred_df)
        supported_types_check(self.y_pred_tensor, self.y_pred_tensor)

        with self.assertRaises(TypeError):
            supported_types_check(self.y_pred_np, self.y_pred_df)
        with self.assertRaises(TypeError):
            supported_types_check(self.y_pred_tensor, self.y_pred_np)
        with self.assertRaises(TypeError):
            supported_types_check(self.y_pred_df, self.y_pred_tensor)


class AlphaCheck(unittest.TestCase):
    """Test if alpha is consistent with the calibration size n.

    Conditions:
        .. math::

            0 < \\alpha < 1

        .. math::

            1/(n+1) < \\alpha < 1

    """

    def test_proper_alpha(self):
        max_n = 10000
        n_vals = np.random.randint(10, high=max_n, size=100)
        for n in n_vals:
            lower_bound = 1 / (n + 1)
            alpha_vals = (1 - lower_bound) * np.random.random_sample(
                size=(100)
            ) + lower_bound
            for alpha in alpha_vals:
                alpha_calib_check(alpha=alpha, n=n)

    def test_out_lowerbound_alpha(self):
        with self.assertRaises(ValueError):
            alpha_calib_check(alpha=0, n=100000)

    def test_out_upperbound_alpha(self):
        with self.assertRaises(ValueError):
            alpha_calib_check(alpha=1, n=100000)

    def test_too_low_alpha(self):
        with self.assertRaises(ValueError):
            alpha_calib_check(alpha=0.01, n=10)


class QuantileCheck(unittest.TestCase):
    def test_simple_quantile1d(self):
        self.a_np = np.array([1, 2, 3, 4])
        self.a_tensor = tf.constant([1, 2, 3, 4])

        expected_result = 2

        self.assertEqual(expected_result, quantile(a=self.a_np, q=0.5))
        self.assertEqual(expected_result, quantile(a=self.a_tensor, q=0.5))

    def test_weight_quantile1d(self):
        self.a_np = np.array([1, 2, 3, 4])
        self.a_tensor = tf.constant([1, 2, 3, 4])

        weights = np.array([0.5, 1 / 6, 1 / 6, 1 / 6])
        expected_result = 1

        print(quantile(a=self.a_np, q=0.6, w=weights))

        self.assertEqual(expected_result, quantile(a=self.a_np, q=0.5, w=weights))
        self.assertEqual(expected_result, quantile(a=self.a_tensor, q=0.5, w=weights))

    def test_simple_quantile2d(self):
        self.a_np = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
        self.a_df = pd.DataFrame(self.a_np)
        self.a_tensor = tf.constant([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])

        expected_result = np.array([3, 30])

        np.testing.assert_array_equal(expected_result, quantile(a=self.a_np, q=0.5))
        np.testing.assert_array_equal(expected_result, quantile(a=self.a_df, q=0.5))
        np.testing.assert_array_equal(expected_result, quantile(a=self.a_tensor, q=0.5))

    def test_weight_quantile2d(self):
        self.a_np = np.array([[1, 2, 3, 4, 5], [20, 10, 50, 40, 100]])
        self.a_df = pd.DataFrame(self.a_np)
        self.a_tensor = tf.constant([[1, 2, 3, 4, 5], [20, 10, 50, 40, 100]])

        weights = np.array([3 / 20, 3 / 20, 3 / 20, 2 / 5, 3 / 20])
        expected_result = np.array([4, 50])

        np.testing.assert_array_equal(
            expected_result, quantile(a=self.a_np, q=0.5, w=weights)
        )
        np.testing.assert_array_equal(
            expected_result, quantile(a=self.a_df, q=0.5, w=weights)
        )
        np.testing.assert_array_equal(
            expected_result, quantile(a=self.a_tensor, q=0.5, w=weights)
        )

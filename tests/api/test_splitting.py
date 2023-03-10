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

from deel.puncc.api.splitting import IdSplitter
from deel.puncc.api.splitting import KFoldSplitter
from deel.puncc.api.splitting import RandomSplitter

np.random.seed(0)


class SplitterCheck(unittest.TestCase):
    def test_proper_initialization(self):
        id_splitter = IdSplitter(
            X_fit=np.empty([20, 20]),
            y_fit=np.empty([20]),
            X_calib=np.empty([10, 20]),
            y_calib=np.empty([10]),
        )
        kfold_splitter = KFoldSplitter(K=10, random_state=0)
        Random_splitter = RandomSplitter(ratio=0.1, random_state=0)

    def test_bad_initialization(self):
        with self.assertRaises(ValueError):
            id_splitter = IdSplitter(
                X_fit=np.empty([20, 20]),
                y_fit=np.empty([2]),
                X_calib=np.empty([10, 10]),
                y_calib=np.empty([10]),
            )
        with self.assertRaises(ValueError):
            kfold_splitter = KFoldSplitter(K=-10, random_state=0)

        with self.assertRaises(ValueError):
            Random_splitter = RandomSplitter(ratio=2, random_state=0)

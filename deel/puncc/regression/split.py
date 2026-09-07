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
"""
This module implements usual conformal regression wrappers.
"""
from deel.puncc.nonconformity_scores import absolute_difference, scaled_ad, cqr_score
from deel.puncc.prediction_sets import constant_interval, scaled_interval, cqr_interval
from deel.puncc.core.split import SplitConformalPredictor, PresetSplitConformalPredictor


class SplitConformalRegression(PresetSplitConformalPredictor):
    nc_score_function=absolute_difference()
    pred_set_function=constant_interval()

class CQR(PresetSplitConformalPredictor):
    nc_score_function=cqr_score()
    pred_set_function=cqr_interval()

# TODO : put the eps somewhere else
class LocallyAdaptiveCP(SplitConformalPredictor):
    def __init__(self, model,
                 fit_function=None,
                 eps:float=1e-12):
        super().__init__(
            model=model,
            nc_score_function=scaled_ad(eps=eps),
            pred_set_function=scaled_interval(eps=eps),
            fit_function=fit_function
        )

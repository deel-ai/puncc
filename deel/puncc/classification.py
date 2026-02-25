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
This module implements conformal classification procedures.
"""
from deel.puncc.api.nonconformity_scores import lac_score, aps_score, raps_score
from deel.puncc.api.prediction_sets import lac_set, aps_set, raps_set
from deel.puncc.api.conformal_predictor import AutoConformalPredictor, ConformalPredictor

class LAC(AutoConformalPredictor):
    nc_score_function=lac_score()
    pred_set_function=lac_set()

class APS(AutoConformalPredictor):
    nc_score_function = aps_score()
    pred_set_function = aps_set(rand=True)

class RAPS(ConformalPredictor):
    # TODO : add random state propagation to control randomized tie breaking
    def __init__(self, model, lambd:float=0, k_reg:int=1, rand:bool=False, weight_function=None, fit_function=None):
        nc_score_function = raps_score(lambd=lambd, k_reg=k_reg)
        pred_set_function = raps_set(lambd=lambd, k_reg=k_reg, rand=rand)
        super().__init__(model, nc_score_function, pred_set_function, weight_function, fit_function)
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

from deel.puncc.cloning import ModelCloningError
from deel.puncc.core.predictors import (
    IDPredictor,
    MeanVarPredictor,
    MultiPredictorStack,
    make_predictor,
)


class DummyModel:
    def __init__(self, prediction):
        self.prediction = tensor(prediction, "float32")
        self.fit_calls = []

    def fit(self, X, y):
        self.fit_calls.append((X, y))
        return self

    def predict(self, X):
        return self.prediction


def test_id_predictor():
    predictor = IDPredictor()

    X = tensor([[1.0], [2.0]], "float32")
    y = tensor([1.0, 2.0], "float32")

    assert predictor.fit(X, y) is predictor

    np.testing.assert_array_equal(to_numpy(predictor.predict(X)), to_numpy(X))
    np.testing.assert_array_equal(to_numpy(predictor(X)), to_numpy(X))


def test_multi_predictor_stack_fit_and_predict():
    model1 = DummyModel([1.0, 2.0])
    model2 = DummyModel([3.0, 4.0])

    predictor = MultiPredictorStack(model1, model2)

    X = tensor([[1.0], [2.0]], "float32")
    y = tensor([1.0, 2.0], "float32")

    assert predictor.fit(X, y) is predictor

    assert len(model1.fit_calls) == 1
    assert len(model2.fit_calls) == 1

    np.testing.assert_array_equal(
        to_numpy(predictor(X)),
        [[1.0, 3.0], [2.0, 4.0]],
    )


def test_multi_predictor_stack_clone():
    model1 = DummyModel([1.0, 2.0])
    model2 = DummyModel([3.0, 4.0])

    predictor = MultiPredictorStack(model1, model2)
    cloned = predictor.clone(clone_weights=True)

    X = tensor([[1.0], [2.0]], "float32")

    assert cloned is not predictor
    assert cloned.models[0] is not predictor.models[0]
    assert cloned.models[1] is not predictor.models[1]

    np.testing.assert_array_equal(to_numpy(cloned(X)), to_numpy(predictor(X)))

    # Modifying the original model must not affect the clone.
    model1.prediction = tensor([99.0, 2.0], "float32")

    np.testing.assert_array_equal(
        to_numpy(cloned(X)),
        [[1.0, 3.0], [2.0, 4.0]],
    )


def test_mean_var_predictor_fit():
    mean_model = DummyModel([2.0, 4.0])
    dispersion_model = DummyModel([0.5, 0.25])

    predictor = MeanVarPredictor(
        mean_model=mean_model,
        dispersion_model=dispersion_model,
    )

    X = tensor([[1.0], [2.0]], "float32")
    y = tensor([1.0, 3.0], "float32")

    assert predictor.fit(X, y) is predictor

    assert len(mean_model.fit_calls) == 1
    assert len(dispersion_model.fit_calls) == 1

    # The dispersion model is trained on absolute residuals.
    dispersion_targets = dispersion_model.fit_calls[0][1]

    np.testing.assert_allclose(
        to_numpy(dispersion_targets),
        [1.0, 1.0],
    )

    np.testing.assert_allclose(
        to_numpy(predictor(X)),
        [[2.0, 0.5], [4.0, 0.25]],
    )


def test_make_predictor_adapter_and_clone():
    model = DummyModel([1.0, 2.0])

    predictor = make_predictor(model)

    np.testing.assert_array_equal(
        to_numpy(predictor(tensor([[0.0], [1.0]], "float32"))),
        [1.0, 2.0],
    )

    # Extra attributes are delegated to the wrapped model.
    predictor.extra = {"key": 1}

    assert model.extra == {"key": 1}

    cloned = predictor.clone(clone_weights=True)

    assert cloned is not predictor
    assert cloned.extra == {"key": 1}

    model.prediction = tensor([99.0, 2.0], "float32")

    np.testing.assert_array_equal(
        to_numpy(cloned(tensor([[0.0], [1.0]], "float32"))),
        [1.0, 2.0],
    )


def test_multi_predictor_stack_clone_failure():
    class UncopyableModel:
        def predict(self, X):
            return tensor(np.zeros(len(X), dtype=np.float32))

        def __deepcopy__(self, memo):
            raise RuntimeError("Cannot deepcopy this model.")

    predictor = MultiPredictorStack(UncopyableModel())

    with pytest.raises(ModelCloningError):
        predictor.clone(clone_weights=True)

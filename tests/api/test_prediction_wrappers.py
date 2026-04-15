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

from deel.puncc.api.prediction import BasePredictor
from deel.puncc.api.prediction import DualPredictor
from deel.puncc.api.prediction import IdPredictor
from deel.puncc.api.prediction import MeanVarPredictor


class DummyModel:
    def __init__(self, prediction=None):
        self.prediction = np.array([1.0, 2.0]) if prediction is None else prediction
        self.fit_calls = []
        self.compile_calls = []

    def compile(self, **kwargs):
        self.compile_calls.append(kwargs)

    def fit(self, X, y=None, **kwargs):
        self.fit_calls.append((X, y, kwargs))

    def predict(self, X, **kwargs):
        del X, kwargs
        return self.prediction


class UncopyableModel:
    def __deepcopy__(self, memo):
        raise RuntimeError("no deepcopy")

    def predict(self, X, **kwargs):
        del X, kwargs
        return [1.0, 2.0]


def test_id_predictor_branches():
    predictor = IdPredictor(model=DummyModel(), tag="x")

    assert predictor.kwargs == {"tag": "x"}
    assert predictor.predict(np.array([1.0, 2.0])) is not None
    np.testing.assert_allclose(
        predictor.predict_with_model(np.array([0.0])),
        np.array([1.0, 2.0]),
    )

    with pytest.raises(RuntimeError):
        predictor.fit(np.array([1.0]))


def test_dual_predictor_validation_and_non_numpy_prediction():
    model1 = DummyModel(prediction=np.array([1.0, 2.0]))
    model2 = DummyModel(prediction=np.array([3.0, 4.0]))
    predictor = DualPredictor(
        models=[model1, model2],
        is_trained=[False, True],
        compile_args=[{"alpha": 1}, {}],
    )

    assert model1.compile_calls == [{"alpha": 1}]
    assert predictor.get_is_trained() is False

    predictor.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]), [{}, {}])
    assert len(model1.fit_calls) == 1
    assert len(model2.fit_calls) == 0

    np.testing.assert_allclose(
        predictor.predict(np.array([[1.0], [2.0]]), [{}, {}]),
        np.array([[1.0, 3.0], [2.0, 4.0]]),
    )

    bad_predictor = DualPredictor(models=[UncopyableModel(), UncopyableModel()])
    with pytest.raises(NotImplementedError):
        bad_predictor.predict(np.array([[1.0]]), [{}, {}])


def test_dual_predictor_copy_and_meanvar_fit():
    model1 = DummyModel(prediction=np.array([1.0, 2.0]))
    model2 = DummyModel(prediction=np.array([3.0, 4.0]))
    predictor = DualPredictor(models=[model1, model2])
    predictor.extra = "kept"

    copied = predictor.copy()

    assert copied is not predictor
    assert copied.extra == "kept"
    assert copied.models[0] is not predictor.models[0]

    mean_model = DummyModel(prediction=np.array([2.0, 4.0]))
    sigma_model = DummyModel(prediction=np.array([0.5, 0.25]))
    meanvar = MeanVarPredictor(models=[mean_model, sigma_model])

    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 3.0])
    meanvar.fit(X, y, [{}, {"epochs": 2}])

    assert len(mean_model.fit_calls) == 1
    assert len(sigma_model.fit_calls) == 1
    sigma_fit_args = sigma_model.fit_calls[0]
    np.testing.assert_allclose(sigma_fit_args[1], np.array([1.0, 1.0]))
    assert sigma_fit_args[2] == {"epochs": 2}


def test_dual_predictor_copy_runtime_error_branch(monkeypatch):
    import deel.puncc.api.prediction as prediction_module

    class ReallyUncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("no deepcopy")

    predictor = DualPredictor(models=[ReallyUncopyable(), ReallyUncopyable()])

    class FailingClone:
        @staticmethod
        def clone_model(model):
            raise RuntimeError("clone failed")

    class FakeKeras:
        models = FailingClone()

    class FakeTf:
        keras = FakeKeras()

    monkeypatch.setattr(prediction_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(prediction_module, "tf", FakeTf())

    with pytest.raises(RuntimeError, match="Cannot copy models"):
        predictor.copy()


def test_dual_predictor_copy_re_raises_without_tensorflow(monkeypatch):
    import deel.puncc.api.prediction as prediction_module

    class ReallyUncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("no deepcopy")

    predictor = DualPredictor(models=[ReallyUncopyable(), ReallyUncopyable()])
    monkeypatch.setattr(prediction_module.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(Exception, match="no deepcopy"):
        predictor.copy()


def test_base_predictor_copy_preserves_extra_attributes():
    predictor = BasePredictor(DummyModel(), is_trained=True)
    predictor.extra = {"k": 1}

    copied = predictor.copy()

    assert copied is not predictor
    assert copied.extra == {"k": 1}

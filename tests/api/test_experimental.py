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

import pytest

torch = pytest.importorskip("torch")

from deel.puncc.api.experimental import TorchPredictor


class TinyNet(torch.nn.Module):
    def __init__(self, input_feat, output_feat):
        super().__init__()
        self.linear = torch.nn.Linear(input_feat, output_feat, bias=True)

    def forward(self, x):
        return self.linear(x)


def test_torch_predictor_fit_updates_weights_and_predicts():
    torch.manual_seed(0)
    model = TinyNet(input_feat=2, output_feat=1)
    predictor = TorchPredictor(
        model=model,
        optimizer=torch.optim.SGD,
        criterion=torch.nn.MSELoss(reduction="sum"),
        lr=0.1,
    )

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([[1.0], [2.0]])
    weights_before = model.linear.weight.detach().clone()

    predictor.fit(x, y, epochs=3)
    y_pred = predictor.predict(x)

    assert y_pred.shape == (2, 1)
    assert not torch.allclose(model.linear.weight.detach(), weights_before)


def test_torch_predictor_copy_clones_model_and_metadata():
    torch.manual_seed(0)
    model = TinyNet(input_feat=2, output_feat=1)
    predictor = TorchPredictor(
        model=model,
        is_trained=True,
        optimizer=torch.optim.SGD,
        criterion=torch.nn.MSELoss(reduction="sum"),
        lr=0.05,
    )

    predictor_copy = predictor.copy()

    assert predictor_copy is not predictor
    assert predictor_copy.model is not predictor.model
    assert predictor_copy.is_trained is True
    assert predictor_copy.optimizer is torch.optim.SGD
    assert predictor_copy.compile_kwargs == {"lr": 0.05}

    for original_param, copied_param in zip(
        predictor.model.parameters(), predictor_copy.model.parameters()
    ):
        assert torch.allclose(original_param, copied_param)

    with torch.no_grad():
        predictor.model.linear.weight.add_(1.0)

    assert not torch.allclose(
        predictor.model.linear.weight, predictor_copy.model.linear.weight
    )
    assert torch.allclose(
        predictor.model.linear.bias, predictor_copy.model.linear.bias
    )


def test_torch_predictor_fit_defaults_to_one_epoch():
    torch.manual_seed(0)
    model = TinyNet(input_feat=2, output_feat=1)
    predictor = TorchPredictor(
        model=model,
        optimizer=torch.optim.SGD,
        criterion=torch.nn.MSELoss(reduction="sum"),
        lr=0.1,
    )

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([[1.0], [2.0]])
    weights_before = model.linear.weight.detach().clone()

    predictor.fit(x, y)

    assert not torch.allclose(model.linear.weight.detach(), weights_before)

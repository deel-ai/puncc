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
This module provides experimental features. Use cautiously.
"""
import importlib.util

if importlib.util.find_spec("torch") is not None:
    import torch

    class TorchPredictor:
        """Wrapper of a torch prediction model :math:`\hat{f}`. Enables to
        standardize the interface of torch predictors and to expose generic :func:`fit`,
        :func:`predict` and :func:`copy` methods.

        :param Any model: torch prediction model :math:`\hat{f}`
        :param bool is_trained: boolean flag that informs if the model is
            pre-trained. If True, the call to :func:`fit` will be skipped
        :param torch.optim.Optimizer optimizer: torch optimizer.
            Defaults to `torch.optim.Adam`.
        :param torch.nn.modules.loss criterion: criterion that measures the distance
            between predictions and outputs. Default to  `torch.nn.MSELoss`.
        :param compile_kwargs: keyword arguments to be used if needed during the
            call :func:`model.compile` on the underlying model

        .. note::

            The `model` constructor has to take as argument both
            `input_feat`:int and `output_feat`:int, corresponding to the number
            of features (or channels) for each input and output, respectively.

        """

        def __init__(
            self,
            model,
            is_trained=False,
            optimizer=torch.optim.Adam,
            criterion=torch.nn.MSELoss(reduction="sum"),
            **compile_kwargs,
        ):
            self.model = model
            self.is_trained = is_trained
            self.compile_kwargs = compile_kwargs
            self.optimizer = optimizer
            self.criterion = criterion

        def fit(self, X, y, **kwargs):
            """Fit model to the training data.

            :param torch.Tensor X: train features.
            :param torch.Tensor y: train labels.
            :param kwargs: keyword arguments to configure the training.

            """
            if "epochs" in kwargs.keys():
                epochs = kwargs["epochs"]
            else:
                epochs = 1

            optimizer = self.optimizer(
                self.model.parameters(), **self.compile_kwargs
            )
            for _ in range(epochs):
                y_pred = self.model(X)
                loss = self.criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        def predict(self, X):
            """Compute predictions on new examples.

            :param torch.Tensor X: new examples' features.

            :returns: predictions :math:`\hat{f}(X)` associated to the new
                examples X.
            :rtype: torch.Tensor

            """
            return self.model(X)

        def copy(self):
            """Returns a copy of the predictor.

            :returns: copy of the predictor.
            :rtype: TorchPredictor

            """
            input_feat = (
                (self.model.state_dict()).popitem(last=False)[1].shape[-1]
            )
            output_feat = (
                (self.model.state_dict()).popitem(last=True)[1].shape[-1]
            )
            model = type(self.model)(
                input_feat=input_feat, output_feat=output_feat
            )
            model.load_state_dict(self.model.state_dict())
            predictor_copy = TorchPredictor(
                model=model,
                is_trained=self.is_trained,
                optimizer=self.optimizer,
                criterion=self.criterion,
                **self.compile_kwargs,
            )

            return predictor_copy

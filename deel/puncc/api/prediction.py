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
This module provides standard wrappings for ML models.
"""
import importlib
from copy import deepcopy
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from deel.puncc.api import nonconformity_scores
from deel.puncc.api.utils import dual_predictor_check
from deel.puncc.api.utils import supported_types_check

if importlib.util.find_spec("tensorflow") is not None:
    import tensorflow as tf


class BasePredictor:
    """Wrapper of a point prediction model :math:`\hat{f}`. Enables to
    standardize the interface of predictors and to expose generic :func:`fit`,
    :func:`predict` and :func:`copy` methods.

    :param Any model: prediction model :math:`\hat{f}`
    :param bool is_trained: boolean flag that informs if the model is
        pre-trained. If True, the call to :func:`fit` will be skipped
    :param compile_kwargs: keyword arguments to be used if needed during the
        call :func:`model.compile` on the underlying model

    .. _example basepredictor:

    Sklearn regression examples::

        from deel.puncc.api.prediction import BasePredictor
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np

        # Generate data
        X_train = np.random.uniform(0, 10, 1000)
        X_new = np.random.uniform(0, 10, 100)
        y_train = np.sin(X)

        # Instantiate two random forest models composed of 100 trees.
        rf_model1 = RandomForestRegressor(n_estimators=100)
        rf_model2 = RandomForestRegressor(n_estimators=100)

        # Consider that `rf_model2` is previously trained
        rf_model2.fit(X_train, y_train)

        # We will instantiate two wrappers:
        # - `predictor1` will wrap `rf_model1`.
        # - `predictor2` will wrap `rf_model2`. Also, we don't want to retrain our model,
        #   so we will specify it to the constructor by setting the argument
        #   `is_trained` to True. If `fit` is called, it will be skipped.
        # The argument `is_trained` defaults to False.
        predictor1 = BasePredictor(rf_model1, is_trained=False)
        predictor2 = BasePredictor(rf_model2, is_trained=True)

        # Fit `predictor2` to the training data.
        # No need to call fit on `predictor2`. But if you do, it will be skipped.
        predictor1.fit(X_train,y_train)

        # Predict on X_new
        y_pred1 = predictor1.predict(X_new)
        y_pred2 = predictor2.fit(X_new)


    Keras classification example::

        from deel.puncc.api.prediction import BasePredictor
        import tensorflow as tf
        import numpy as np

        # Generate data
        X_train = np.random.uniform(0, 10, 1000)
        X_train = np.expand_dims(X_train,-1)
        X_new = np.random.uniform(0, 10, 100)
        y_train = np.array([1 if x>5 else 0 for x in X_train])
        y_train = tf.keras.utils.to_categorical(y_train)

        # Instantiate a classifier as sequential model and add two dense layers
        cl_model = tf.keras.Sequential()
        cl_model.add(tf.keras.layers.Dense(10, activation="relu"))
        cl_model.add(tf.keras.layers.Dense(2, activation="softmax"))

        # The compile options need be passed to the constructor `BasePredictor`.
        # The wrapper will call compile(**compile_kwargs) on an internal copy of the model.
        # Our model is a classifier, we use categorical crossentropy as loss function.
        compile_kwargs={"optimizer":"rmsprop", "loss":"categorical_crossentropy"}
        predictor = BasePredictor(cl_model, is_trained=False, **compile_kwargs)

        # The fit method is provided with a given training dataset (X,y) and
        # with the train configuration of the underlying model. In the example below,
        # we train the model over 5 epochs on batches containing 128 examples.
        predictor.fit(X_train, y_train, **{"epochs":5,"batch_size":128})

        # The `BasePredictor.predict` method enables to pass keyword arguments
        # to the `predict`call of the underlying model (e.g., verbose).
        y_pred = predictor.predict(X_new, **{"verbose":1})

    """

    def __init__(self, model: Any, is_trained: bool = False, **compile_kwargs):
        self.model = model
        self.is_trained = is_trained
        self.compile_kwargs = compile_kwargs

        if self.compile_kwargs:
            _ = self.model.compile(**self.compile_kwargs)

    def get_is_trained(self) -> bool:
        """Get flag that informs if the model is pre-trained."""
        return self.is_trained

    def fit(self, X: Iterable, y: Optional[Iterable] = None, **kwargs) -> None:
        """Fit model to the training data.

        :param Iterable X: train features.
        :param Optional[Iterable] y: train labels. Defaults to None (unsupervised).
        :param kwargs: keyword arguments to be passed to the call :func:`fit`
            on the underlying model :math:`\hat{f}`.

        .. note::

            For more details, check this :ref:`code snippets
            <example basepredictor>`.

        """
        if y is None:
            self.model.fit(X, **kwargs)
        else:
            self.model.fit(X, y, **kwargs)
        # self.is_trained = True

    def predict(self, X: Iterable, **kwargs) -> np.ndarray:
        """Compute predictions on new examples.

        :param Iterable X: new examples' features.
        :param dict kwargs: predict configuration to be passed to the `predict`
            method of the underlying model :math:`\hat{f}`.

        :returns: predictions :math:`\hat{f}(X)` associated to the new
            examples X.
        :rtype: ndarray

        .. note::

            For more details, check this :ref:`code snippets
            <example basepredictor>`.

        """
        # Remove axis of length one to avoid broadcast in computation
        # of non-conformity scores
        return np.squeeze(self.model.predict(X, **kwargs))

    def copy(self):
        """Returns a copy of the predictor. The underlying model is either
        cloned (Keras model) or deepcopied (sklearn and similar models).

        :returns: copy of the predictor.
        :rtype: BasePredictor

        :raises RuntimeError: copy unsupported for provided models.

        """

        model_type_str = str(type(self.model))

        if (
            "tensorflow" in model_type_str
            or "keras" in model_type_str
            and importlib.util.find_spec("tensorflow") is not None
        ):
            # pylint: disable=E1101
            model = tf.keras.models.clone_model(self.model)
            try:
                model.set_weights(self.model.get_weights())
            except Exception:
                pass
        else:
            model = deepcopy(self.model)

        predictor_copy = self.__class__(
            model=model, is_trained=self.is_trained, **self.compile_kwargs
        )
        return predictor_copy


class IdPredictor(BasePredictor):
    """Subclass of :class:`BasePredictor` to directly wrap existing predictions.
    The predictions are directly returned without any modification.

    :param model: model to be wrapped.
    """

    def __init__(self, model=None, **kwargs):
        self.kwargs = kwargs
        super().__init__(model, is_trained=True)

    def predict(self, X: Iterable):
        """Returns the input argument as output data.

        :param Iterable X: predictions.

        :return: predictions.
        :rtype: np.ndarray
        """
        return X

    def predict_with_model(self, X):
        """
        Predicts the output using the wrapped model.

        :param Iterable X: the input features to build predictions.

        :return: predictions.
        :rtype: np.ndarray
        """
        return self.model.predict(X)


class DualPredictor:
    """Wrapper of **two** joint point prediction models
    :math:`(\hat{f_1},\hat{f_2})`.
    The prediction :math:`\hat{y}` of a :class:`DualPredictor` is a tuple
    :math:`\hat{y}=(\hat{y}_1,\hat{y}_2)`, where :math:`\hat{y}_1`
    (resp. :math:`\hat{y}_2`) is the prediction of :math:`\hat{f}_1`
    (resp. :math:`\hat{f}_2`).

    Enables to standardize the interface of predictors and to expose generic
    :func:`fit`, :func:`predict` and :func:`copy` methods.

    :param List[Any] model: list of two prediction models
        :math:`[\hat{f_1},\hat{f_2}]`.
    :param List[bool] is_trained: list of boolean flag that informs if the
        models are pre-trained. True value will skip the fitting of the
        corresponding model.
    :param List compile_kwargs: list of keyword arguments to be used if needed
        to compile the underlying models.

    .. _example dualpredictor:

    Joint conditional mean / quantile regression example::

        from deel.puncc.api.prediction import DualPredictor
        import tensorflow_addons as tfa
        import tensorflow as tf
        from sklearn.ensemble import RandomForestRegressor

        # Instantiate conditional mean model
        rf_model = RandomForestRegressor(n_estimators=100)

        # Instantiate 90-th quantile model
        q_model = tf.keras.Sequential()
        q_model.add(tf.keras.layers.Dense(10, activation="relu"))
        q_model.add(tf.keras.layers.Dense(1, activation="relu"))
        pinball_loss=tfa.losses.PinballLoss(tau=.9)
        compile_kwargs={'optimizer':'sgd', 'loss':pinball_loss}

        # The compile options need be passed to the constructor `DualPredictor`.
        # The wrapper will call compile on the internal copy of each model if needed.
        # Our predictor combines two regressors, of which only the second needs to be compiled.
        tf_predictor = DualPredictor(models=[rf_model, q_model],
                                     compile_args=[{}, compile_kwargs])

        # The fit method is provided with a given training dataset (X,y) and
        # the train configurations of the underlying models. In the example below,
        # no specific configuration is passed to `rf_model` whereas we want to
        # train the `q_model` over 10 epochs on batches containing 32 examples.
        predictor.fit(X_train, y_train, [{}, {"epochs":10,"batch_size":32}])

        # The `DualPredictor.predict` method enables to pass keyword arguments
        # to the `predict` call of the underlying models. In the example below,
        # we want to turn off the verbosity of `q_model.predict`.
        # Besides, `y_pred` consists of a couple (y1, y2) for each new example.
        # If `X_new` is a (n,m) matrix, the shape of `y_pred` will be (n, 2).
        y_pred = predictor.predict(X_new, [{}, {"verbose":0}])


    """

    def __init__(
        self,
        models: List[Any],
        is_trained: List[bool] = [False, False],
        compile_args: List[dict] = [{}, {}],
    ):
        dual_predictor_check(models, "models", "models")
        dual_predictor_check(is_trained, "is_trained", "booleans")
        dual_predictor_check(compile_args, "compile_args", "dictionnaries")
        self.models = models
        self.is_trained = is_trained
        self.compile_args = compile_args

        if len(self.compile_args[0].keys()) != 0 and is_trained[0] is False:
            _ = self.models[0].compile(**self.compile_args[0])

        if len(self.compile_args[1].keys()) != 0 and is_trained[1] is False:
            _ = self.models[1].compile(**self.compile_args[1])

    def get_is_trained(self) -> bool:
        """Get flag that informs if the models are pre-trained.
        Returns True only when both models are pretrained.
        """
        return self.is_trained[0] and self.is_trained[1]

    def fit(
        self, X: Iterable, y: Iterable, dictargs: List[dict] = [{}, {}]
    ) -> None:
        """Fit model to the training data.

        :param Iterable X: train features.
        :param Iterable y: train labels.
        :param List[dict[str]] dictargs: list of fit configurations to be
            passed to the `fit` method of the underlying models
            :math:`\hat{f}_1` and :math:`\hat{f}_2`, respectively.

        .. note::

            For more details, check this :ref:`code snippet
            <example dualpredictor>`.

        """
        dual_predictor_check(dictargs, "dictargs", "dictionnaries")
        for count, model in enumerate(self.models):
            if not self.is_trained[count]:
                model.fit(X, y, **dictargs[count])

    def predict(self, X, dictargs: List[dict] = [{}, {}]) -> Tuple[np.ndarray]:
        """Compute predictions on new examples.

        :param Iterable X: new examples' features.
        :param dict kwargs: list of predict configurations to be passed to the
            `predict` method of the underlying models :math:`\hat{f}_1` and
            :math:`\hat{f}_2`, respectively.

        :returns: predictions :math:`\hat{y}=\hat{f}(X)` associated to the new
            examples X. For an instance :math:`X_i`, the prediction consists of
            a couple :math:`\hat{f}(X_i)=(\hat{f}_1(X_i), \hat{f}_2(X_i))`.
        :rtype: Tuple[ndarray]

        :raises NotImplementedError: predicted values not formated as numpy
            ndarrays.

        .. note::

            For more details, check this :ref:`code snippet
            <example dualpredictor>`.

        """
        dual_predictor_check(dictargs, "dictargs", "dictionnaries")
        model1_pred = self.models[0].predict(X, **dictargs[0])
        model2_pred = self.models[1].predict(X, **dictargs[1])
        supported_types_check(model1_pred, model2_pred)

        if isinstance(model1_pred, np.ndarray):
            Y_pred = np.column_stack((model1_pred, model2_pred))
        else:  # In case the models return something other than an np.ndarray
            raise NotImplementedError(
                "Predicted values must be of type numpy.ndarray."
            )

        return np.squeeze(Y_pred)

    def copy(self):
        """Returns a copy of the predictor. The underlying models are either
        cloned (Keras model) or deepcopied (sklearn and similar models).

        :returns: copy of the predictor.
        :rtype: DualPredictor

        :raises RuntimeError: copy unsupported for provided models.

        """
        models_copy = []
        for model in self.models:
            try:
                model_copy = deepcopy(model)
            except Exception as e_outer:
                if importlib.util.find_spec("tensorflow") is not None:
                    try:
                        # pylint: disable=E1101
                        model_copy = tf.keras.models.clone_model(model)
                    except Exception as e_inner:
                        msg = (
                            f"Cannot copy models. Many possible reasons:\n"
                            f" 1- {e_inner} \n 2- {e_outer}"
                        )
                        raise RuntimeError(msg)
                else:
                    raise Exception(e_outer)
            models_copy.append(model_copy)

        predictor_copy = self.__class__(
            models=models_copy,
            is_trained=self.is_trained,
            compile_args=self.compile_args,
        )
        return predictor_copy


class MeanVarPredictor(DualPredictor):
    """Subclass of :class:`DualPredictor` to specifically wrap a conditional
    mean estimator :math:`\hat{\mu}` and a conditional dispersion estimator
    :math:`\hat{\sigma}`.\n

     Specifically, the dispersion model :math:`\hat{\sigma}` is trained on the
     mean absolute deviation of :math:`\hat{\mu}`'s predictions from the true
     labels :math:`y`. Given two training algorithms :math:`{\cal A}_1` and
     :math:`{\cal A}_2` and a training dataset :math:`(X_{train}, y_{train})`:

         .. math::

             \hat{\mu} \Leftarrow {\cal A}_1(X_{train}, y_{train})

         .. math::
             \hat{\sigma} \Leftarrow {\cal A}_2(X_{train},
             |\hat{\mu}(X_{train})-y_{train}|)


     :param List[Any] model: list of two prediction models
        :math:`[\hat{\mu},\hat{\sigma}]`.
     :param List[bool] is_trained: list of boolean flag that informs if the
        models are pre-trained. True value will skip the fitting of the
        corresponding model.
     :param List compile_kwargs: list of keyword arguments to be used if
        needed to compile the underlying models :math:`\hat{\mu}` and
        :math:`\hat{\sigma}`, respectively.

     .. _example MeanVarPredictor:

    Here follows an example of wrapping conditional mean and dispersion models::

        from deel.puncc.api.prediction import MeanVarPredictor
        from sklearn.ensemble import RandomForestRegressor

        # Instantiate conditional mean model
        mu_model = linear_model.LinearRegression()
        # Instantiate conditional dispersion model
        sigma_model = RandomForestRegressor(
            n_estimators=100, random_state=random_seed
        )

        # The instantiation of a :class:`MeanVarPredictor` is simple as the
        # selected models do not require any compilation
        mean_var_predictor = MeanVarPredictor([mu_model, sigma_model])

        # The fit method is provided with a given training dataset (X,y).
        # We do not choose any specific the train configurations.
        mean_var_predictor.fit(X_train, y_train)

        # The method `predict` yields `y_pred` that consists of a couple
        # (y1, y2) for each new example.
        # If `X_new` is a (n,m) matrix, the shape of `y_pred` will be (n, 2).
        y_pred = mean_var_predictor.predict(X_new)


    To see an example how to pass compilation/fit/predict configurations as
    arguments to the underlying models, check this
    :ref:`code snippet <example DualPredictor>`.

    """

    def fit(
        self, X: Iterable, y: Iterable, dictargs: List[dict] = [{}, {}]
    ) -> None:
        """Fit models to the training data. The dispersion model
        :math:`\hat{\sigma}` is trained on the mean absolute deviation of
        :math:`\hat{mu}`'s predictions :math:`\hat{\mu}` from the true labels
        :math:`y`.

        :param Iterable X: train features.
        :param Iterable y: train labels.
        :param List[dict] dictargs: list of fit configurations to be passed to
            the `fit` method of the underlying models :math:`\hat{\mu}` and
            :math:`\hat{\sigma}`, respectively.
        """
        self.models[0].fit(X, y, **dictargs[0])
        mu_pred = self.models[0].predict(X)
        mads = nonconformity_scores.absolute_difference(mu_pred, y)
        self.models[1].fit(X, mads, **dictargs[1])

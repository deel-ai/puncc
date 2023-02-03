==================
🚀 Quickstart
==================

Conformal Regression
--------------------

Let's consider a simple regression problem on diabetes data provided by
`Scikit-learn <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`_.
We want to evaluate the uncertainty associated with the prediction using **inductive (or split) conformal prediction**.

Data
****

By construction, data are indepent and identically distributed (i.i.d) (for
more information, check the official
`documentation <https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html>`_).
Great: we fullfill the exchangeability condition to apply conformal prediction!
The next step is spliting the data into three subsets:

* Fit subset :math:`{\cal D_{fit}}` to train the model.
* Calibration subset :math:`{\cal D_{calib}}` on which nonconformity scores are
  computed.
* Test subset :math:`{\cal D_{test}}` on which the prediction intervals are
  estimated.

The following code implements all the aforementioned steps::

   import numpy as np
   from sklearn import datasets

   # Load the diabetes dataset
   diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

   # Use only one feature
   diabetes_X = diabetes_X[:, np.newaxis, 2]

   # Split the data into training/testing sets
   X_train = diabetes_X[:-100]
   X_test = diabetes_X[-100:]

   # Split the targets into training/testing sets
   y_train = diabetes_y[:-100]
   y_test = diabetes_y[-100:]

   # Split fit and calibration data
   X_fit, X_calib = X_train[:-100], X_train[-100:]
   y_fit, y_calib = y_train[:-100], y_train[-100:]

.. warning::

   Rigorously, for the probabilistic guarantee to hold, the calibration subset
   needs to be sampled for each new example in the test set.

Prediction model
****************

We consider a simple linear regression model from
`scikit-learn regression module <https://scikit-learn.org/stable/modules/linear_model.html>`_,
to be trained later on :math:`{\cal D_{fit}}`::

   from sklearn import linear_model

   # Create linear regression model
   lin_reg_model = linear_model.LinearRegression()

Such model needs to be wrapped in a wrapper provided in the module
:mod:`deel.puncc.api.prediction`.
The wrapper makes it possible to use various models from different ML/DL
libraries such as `Scikit-learn <https://scikit-learn.org/>`__,
`Keras <https://keras.io/>`_ or
`XGBoost <https://xgboost.readthedocs.io/en/stable/>`_.
For more information about model wrappers and supported ML/DL libraries,
checkout :doc:`here <prediction>`.

For a linear regression from scikit-learn, we use
:class:`deel.puncc.api.prediction.BasePredictor` as follows::

   from deel.puncc.api.prediction import BasePredictor

   # Create a predictor to wrap the linear regression model defined earlier
   lin_reg_predictor =  BasePredictor(lin_reg_model)


Conformal prediction
**************************

For this example, the prediction intervals are obtained throught the split
conformal prediction method provided by the class
:class:`deel.puncc.regression.SplitCP`. Other methods are presented
:doc:`here <regression>`.


.. code-block:: python

   from deel.puncc.regression import SplitCP

   # Coverage target is 1-alpha = 90%
   alpha=.1

   # Instanciate the split cp wrapper on the linear predictor.
   # The `train` argument is set to True such that the linear model is trained
   # before the calibration. You can initialize it to False if the model is
   # already trained and you want to save time.
   split_cp = SplitCP(lin_reg_predictor, train=True)

   # Train model (if argument `train` is True) on the fitting dataset and
   # compute the residuals on the calibration dataset.
   split_cp.fit(X_fit, y_fit, X_calib, y_calib)

   # The `predict` returns the output of the linear model `y_pred` and
   # the calibrated interval [`y_pred_lower`, `y_pred_upper`].
   y_pred, y_pred_lower, y_pred_upper = split_cp.predict(X_test, alpha=alpha)

Our library also provides plotting tools in :mod:`deel.puncc.plotting`
to visualize the prediction intervals and whether or not the observations
are covered::

   from deel.puncc.plotting import plot_prediction_interval

   # Figure of the prediction bands

   plot_prediction_interval(
      X = X_test[:,0],
      y_true=y_test,
      y_pred=y_pred,
      y_pred_lower=y_pred_lower,
      y_pred_upper=y_pred_upper,
      sort_X=True,
      size=(10, 6),
      loc="upper left")


.. figure:: results_quickstart_split_cp_pi.png
   :width: 600px
   :align: center
   :height: 300px
   :figclass: align-center

   90%-prediction interval with the split conformal prediction method

In the long run, 90% of the examples are included in the prediction interval.

Conformal Classification
------------------------

Let's tackle the classic problem of
`MNIST handwritten digits <https://en.wikipedia.org/wiki/MNIST_database>`_
classification. The goal is to evaluate through **conformal prediction** the
uncertainty associated to predictive classifiers.

Data
****

MNIST dataset contains a large number of digit images to which are associated digit labels.
As the data generating process is considered i.i.d (check `this post <https://newsletter.altdeep.ai/p/the-story-of-mnist-and-the-perils>`_),
conformal prediction is applicable 👏.\n

We split the data into three subsets:

* Fit subset :math:`{\cal D_{fit}}` to train the model.
* Calibration subset :math:`{\cal D_{calib}}` on which nonconformity scores are
  computed.
* Test subset :math:`{\cal D_{test}}` on which the prediction intervals are
  estimated.

In addition to data preprocessing, the following code implements the
aforementioned steps:

.. code-block:: python

   from tensorflow.keras.datasets import mnist

   # Load MNIST Database
   (X_train, y_train), (X_test, y_test) = mnist.load_data()

   # Preprocessing: reshaping and standardization
   X_train = X_train.reshape((len(X_train), 28 * 28))
   X_train = X_train.astype('float32') / 255
   X_test = X_test.reshape((len(X_test), 28 * 28))
   X_test = X_test.astype('float32') / 255

   # Split fit and calib datasets
   X_fit, X_calib  = X_train[:50000], X_train[50000:]
   y_fit, y_calib  = y_train[:50000], y_train[50000:]

   # One hot encoding of classes
   y_fit_cat = to_categorical(y_fit)
   y_calib_cat = to_categorical(y_calib)
   y_test_cat = to_categorical(y_test)


Model
*****

.. code-block:: python

   # Classification model: MLP composed of two layers
   nn_model = models.Sequential()
   nn_model.add(layers.Dense(4, activation='relu', input_shape=(28 * 28,)))
   nn_model.add(layers.Dense(10, activation='softmax'))
   # The configuration of the compilation and the fit calls are gathered in two
   # dictionnaries
   compile_kwargs = {"optimizer":"rmsprop", "loss":"categorical_crossentropy","metrics":[]}
   fit_kwargs = {"epochs":5,"batch_size":128, "verbose":1}

   class_predictor = BasePredictor(nn_model, is_trained=False, **compile_kwargs)

Conformal classification
************************

.. code-block:: python

   alpha = .1

   raps_cp = Raps(class_predictor, k_reg=1, lambd=0)
   raps_cp.fit(X_fit, y_fit_cat, X_calib, y_calib, **fit_kwargs)
   y_pred, set_pred = raps_cp.predict(X_test, alpha=alpha)

   mean_coverage = metrics.classification_mean_coverage(y_test, set_pred)
   mean_size = metrics.classification_mean_size(set_pred)

   print(f"Empirical coverage : {mean_coverage}%")
   print(f"Average set size : {mean_size}")

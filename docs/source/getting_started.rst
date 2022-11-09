🚀 Quickstart
==================

Let's consider a simple regression problem on diabetes data provided by `Scikit-learn <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`_.
We want to evaluate the uncertainty associated with the prediction using **inductive conformal prediction**.

Data
****

Data are assumed to be indepent and identically distributed. For more information, the reader is referred to `the documentation <https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html>`_.

Data processing results in three subsets:

* Fit subset :math:`{\cal D_{fit}}` to train the model
* Calibration subset :math:`{\cal D_{calib}}` on which nonconformity scores are computed
* Test subset :math:`{\cal D_{test}}` to evaluate the validity and efficiency of the prediction intervals

.. code-block:: python

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

Linear regression
*****************

We consider a simple linear regression model from
`scikit-learn regression module <https://scikit-learn.org/stable/modules/linear_model.html>`_,
to be trained later on :math:`{\cal D_{fit}}` and approximate the conditional
mean :math:`\mathbb{E}[Y|X]`:

.. code-block:: python

   from sklearn import linear_model

   # Create linear regression model
   regr = linear_model.LinearRegression()


Conformal prediction
**************************

For this example, the prediction intervals are obtained throught the split
conformal prediction method provided by :mod:`deel.puncc.regression`. Other
methods are presented :doc:`here <regression>`.


.. code-block:: python

   from deel.puncc.regression import SplitCP

   # Coverage target is 1-alpha = 90%
   alpha=.1
   # Instanciate the split cp wrapper on the linear model
   split_cp = SplitCP(regr)
   # Train model on the fitting dataset and compute residuals on the calibration
   # dataset
   split_cp.fit(X_fit, y_fit, X_calib, y_calib)
   # Estimate the prediction interval
   y_pred, y_pred_lower, y_pred_upper = split_cp.predict(X_test, alpha=alpha)


.. code-block:: python

   from deel.puncc.utils import plot_prediction_interval

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

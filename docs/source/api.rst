.. _api:

💻 API
=======

The **low-level API** offers more flexibility into defining conformal prediction wrappers.
Let's say we want to fit/calibrate a neural-network interval-estimator with a cross-validation plan;
or that we want to experiment different user-defined nonconformity scores.
In such cases and others, the user can fully construct their wrappers using the
proposed **Predictor-Calibrator-Splitter** paradigm. It boils down to assembling
into :class:`deel.puncc.api.conformalization.ConformalPredictor`:

* Prediction model(s).
* An estimator of nonconformity scores for construction/calibration of prediction intervals.
* A strategy to assign data into fitting and calibration sets (case of inductive CP).

.. contents:: Table of Contents
    :depth: 3

API's Modules
*************

.. toctree::
    :maxdepth: 2

    conformalization
    prediction
    calibration
    splitting
    utils
    nonconformity_scores
    prediction_sets

Overview
********

ConformalPredictor
------------------

:class:`deel.puncc.api.conformalization.ConformalPredictor` is the canvas of conformal prediction procedures.
An object instance is constructed by, as we will explain later, a **predictor**, a **calibrator** and a **splitter**:

.. code-block:: python

    # Imports
    from sklearn import linear_model
    from deel.puncc.api.conformalization import ConformalPredictor
    from deel.puncc.api.prediction import BasePredictor
    from deel.puncc.api.calibration import BaseCalibrator
    from deel.puncc.api.splitting import KFoldSplitter

    # Regression linear model
    model = linear_model.LinearRegression()

    # Definition of a predictor (This will be explained later)
    my_predictor = BasePredictor(model) # Predictor

    # Definition of a calibrator, built for a given nonconformity scores and a
    # procedure to build the prediction sets

    ## Definition of a custom nonconformity scores function.
    ## Alternatively, several ready-to-use nonconf scores are provided in
    ## the module deel.puncc.nonconformity_scores (more on this later)
    def my_ncf(y_pred, y_true):
        return np.abs(y_pred-y_true)

    ## Definition of a custom function to build prediction sets.
    ## Alternatively, several ready-to-use procedure are provided in
    ## the module deel.puncc.prediction_sets (more on this later)
    def my_psf(y_pred, nonconf_scores_quantile):
        y_lower = y_pred - nonconf_scores_quantile
        y_upper = y_pred + nonconf_scores_quantile
        return y_lower, y_upper

    ## Calibrator construction
    my_calibrator = BaseCalibrator(nonconf_score_func=my_ncf,
                                   pred_set_func=my_psf)

    # Definition of a K-fold splitter that produces 20 folds of fit/calibration
    kfold_splitter = KFoldSplitter(K=20, random_state=42)

    # Conformal prediction canvas
    conformal_predictor = ConformalPredictor(predictor=my_predictor,
                                            calibrator=my_calibrator,
                                            splitter=kfold_splitter)

:class:`deel.puncc.api.conformalization.ConformalPredictor` implements two methods:


* A :func:`fit` method that fits the predictor model and computes nonconformity scores accodingly to the calibrator and to the data split strategy provided by the splitter

.. code-block:: python

    # X_train and y_train are the full training dataset
    # The splitter passed as argument to ConformalPredictor assigns data
    # to the fit and calibration sets based on the provided splitting strategy
    conformal_predictor.fit(X_train, y_train)

* And a :func:`predict` method that estimates for new samples the point predictions and prediction intervals [y_pred_lower, y_pred_upper], w.r.t a chosen error (significance) level :math:`\\alpha`

.. code-block:: python

    # Coverage target of 1-alpha = 90%
    y_pred, y_pred_lower, y_pred_upper = conformal_predictor.predict(X_new, alpha=.1)

The full code snippet of the previous CVplus-like procedure with a randomly generated dataset is provided below:

.. code-block:: python

    # Imports
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn import linear_model
    from deel.puncc.api.conformalization import ConformalPredictor
    from deel.puncc.api.prediction import BasePredictor
    from deel.puncc.api.calibration import BaseCalibrator
    from deel.puncc.api.splitting import KFoldSplitter
    from deel.puncc.plotting import plot_prediction_intervals
    from deel.puncc import metrics

    # Data
    ## Generate a random regression problem
    X, y = make_regression(n_samples=1000, n_features=4, n_informative=2,
                            random_state=0, shuffle=False)

    ## Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=0
    )

    # Regression linear model
    model = linear_model.LinearRegression()

    # Definition of a predictor (This will be explained later)
    my_predictor = BasePredictor(model) # Predictor

    # Definition of a calibrator, built for a given nonconformity scores and a
    # procedure to build the prediction sets

    ## Definition of a custom nonconformity scores function.
    ## Alternatively, several ready-to-use nonconf scores are provided in
    ## the module deel.puncc.nonconformity_scores (more on this later)
    def my_ncf(y_pred, y_true):
        return np.abs(y_pred-y_true)

    ## Definition of a custom function to build prediction sets.
    ## Alternatively, several ready-to-use procedure are provided in
    ## the module deel.puncc.prediction_sets (more on this later)
    def my_psf(y_pred, nonconf_scores_quantile):
        y_lower = y_pred - nonconf_scores_quantile
        y_upper = y_pred + nonconf_scores_quantile
        return y_lower, y_upper

    ## Calibrator construction
    my_calibrator = BaseCalibrator(nonconf_score_func=my_ncf,
                                    pred_set_func=my_psf) # Calibrator

    # Definition of a K-fold splitter that produces 20 folds of fit/calibration
    kfold_splitter = KFoldSplitter(K=20, random_state=42) # Splitter

    # Conformal prediction canvas
    conformal_predictor = ConformalPredictor(predictor=my_predictor,
                                            calibrator=my_calibrator,
                                            splitter=kfold_splitter)
    conformal_predictor.fit(X_train, y_train)
    y_pred, y_pred_lower, y_pred_upper = conformal_predictor.predict(X_test, alpha=.1)

    # Compute empirical marginal coverage and average width of the prediction intervals
    coverage = metrics.regression_mean_coverage(y_test, y_pred_lower, y_pred_upper)
    width = metrics.regression_sharpness(y_pred_lower=y_pred_lower,
                                        y_pred_upper=y_pred_upper)
    print(f"Marginal coverage: {np.round(coverage, 2)}")
    print(f"Average width: {np.round(width, 2)}")

    # Figure of the prediction bands
    plot_prediction_intervals(
        X = X_test[:,0],
        y_true=y_test,
        y_pred=y_pred,
        y_pred_lower=y_pred_lower,
        y_pred_upper=y_pred_upper,
        sort_X=True,
        size=(10, 6),
        loc="upper left")

Predictor
---------

The :class:`deel.puncc.api.prediction.BasePredictor` and :class:`deel.puncc.api.prediction.DualPredictor` classes are wrappers of ML/DL models
that aims to expose a standardized interface and guarantee compliance with the `puncc`'s framework.
The predictors have to implement:

* a :func:`fit` method used to train the model. It takes as arguments two iterables X, Y (collection of data such as ndarray and tensors) and any additional configuration of the underlying model (e.g., random seed).
* a :func:`predict` method used to predict targets for a given iterable X. It takes as arguments an iterable X and any additional configuration of the underlying model (e.g., batch size).
* a :func:`copy` method that returns a copy of the predictor (useful in cross validation for example). It has to deepcopy the underlying model.

The constructor of :class:`deel.puncc.api.prediction.BasePredictor` takes in the model to be wrapped, a flag to inform if the model is already trained
and compilation keyword arguments if the underlying model needs to be compiled (such as in keras).

The constructor of :class:`deel.puncc.api.prediction.DualPredictor` is conceptually similar but take as arguments
a list of two models, a list of two trained flags and a list of two compilation kwargs.
Such predictor is useful when the calibration relies of several models (such as upper and lower quantiles in CQR).
Note that the output `y_pred` of the :func:`predict` method are a collection of couples,
where the first (resp. second) axis is associated to the output of the first (resp. second) model.
Specifically, :class:`deel.puncc.api.prediction.MeanVarPredictor` is a subclass of :class:`deel.puncc.api.prediction.DualPredictor` that
trains the first model on the data and the second one to predict the absolute error of the former model.

These three predictor classes cover plenty of use case in conformal prediction.
But if you have a special need, you can subclass :class:`deel.puncc.api.prediction.BasePredictor` or :class:`deel.puncc.api.prediction.DualPredictor` or
even create a predictor from scratch.

Here is a example of situation where you need to define your own predictor:
you have a classification problem and you build a :class:`RandomForestClassifier`
from sklearn. The procedure :ref:`RAPS <theory raps>` to conformalize the classifier requires
a :func:`predict` method that outputs the estimated probabily of each class. This is not the case
as :func:`RandomForestClassifier.predict` returns only the most likely class. In this case,
we need to create a predictor in which we redefine the :func:`predict` call:

.. code-block:: python

        from sklearn.ensemble import RandomForestClassifier

        # Create rf classifier
        rf_model = (n_estimators=100, random_state=0)

        # Create a wrapper of the random forest model to redefine its predict method
        # into logits predictions. Make sure to subclass BasePredictor.
        # Note that we needed to build a new wrapper (over BasePredictor) only because
        # the predict(.) method of RandomForestClassifier does not predict logits.
        # Otherwise, it is enough to use BasePredictor (e.g., neural network with softmax).
        class RFPredictor(BasePredictor):
            def predict(self, X, **kwargs):
                return self.model.predict_proba(X, **kwargs)

        # Wrap model in the newly created RFPredictor
        rf_predictor = RFPredictor(rf_model)

Calibrator
----------

The calibrator provides a structure to estimate the nonconformity scores
on the calibration set and to compute the prediction sets. At the constructor :class:`deel.puncc.api.calibration.BaseCalibrator`,
one decides which nonconformity score and prediction set functions to use.
Then, the calibrator instance computes **nonconformity scores** (e.g., mean absolute deviation) by calling
:func:`deel.puncc.api.calibration.Calibrator.fit` on the calibration dataset. Based on the estimated quantiles of nonconformity scores,
the method :func:`deel.puncc.api.calibration.BaseCalibrator.calibrate` enables to **construct** and/or **calibrate** prediction sets.

For example, the `BaseCalibrator` in the split conformal prediction procedure
uses the mean absolute deviation as nonconformity score and and prediction set
are built as constant intervals. These two functions are already provided in
:func:`deel.puncc.api.nonconformity_scores.mad` and :func:`deel.puncc.api.prediction_sets.constant_interval`, respectively:

.. code-block:: python

    from deel.puncc.api.calibration import BaseCalibrator
    from deel.puncc.api import nonconformity_scores
    from deel.puncc.api import prediction_sets

    ## Calibrator construction
    my_calibrator = BaseCalibrator(nonconf_score_func=nonconformity_scores.mad,
                                   pred_set_func=prediction_sets.constant_interval)

Alternatively, one can define custom functions and pass them as arguments to the calibrator:

.. code-block:: python

    from deel.puncc.api.calibration import BaseCalibrator

    ## Definition of a custom nonconformity scores function.
    ## Alternatively, several ready-to-use nonconf scores are provided in
    ## the module deel.puncc.nonconformity_scores
    def my_ncf(y_pred, y_true):
        return np.abs(y_pred-y_true)

    ## Definition of a custom function to build prediction sets.
    ## Alternatively, several ready-to-use procedure are provided in
    ## the module deel.puncc.prediction_sets
    def my_psf(y_pred, nonconf_scores_quantile):
        y_lower = y_pred - nonconf_scores_quantile
        y_upper = y_pred + nonconf_scores_quantile
        return y_lower, y_upper

    ## Calibrator construction
    my_calibrator = BaseCalibrator(nonconf_score_func=my_ncf,
                                   pred_set_func=my_psf)

Splitter
--------

In conformal prediction, the assignement of data into fit and calibration sets is motivated by two criteria:
data availability and computational resources. If quality data is abundant,
we can split the training samples into disjoint subsets :math:`D_{fit}` and :math:`D_{calib}`.
When data is scarce, a cross-validation strategy is preferred but is more
ressource-consuming as different models are trained and nonconformity scores
are computed for different disjoint folds.

The two plans are implemented in :mod:`deel.puncc.api.splitting` module,
and are agnostic to the data structure (which can be ndarrays, tensors and dataframes):

- :class:`deel.puncc.api.splitting.RandomSplitter`: random assignement of samples in :math:`D_{fit}` and :math:`D_{calib}`
- :class:`deel.puncc.api.splitting.KFoldSplitter`: random assignement of samples into K disjoint folds. Note that if K equals the size of training set, the split is identified with the leave-one-out strategy

Additionnaly, if the user already implemted a split plan, the obtained data asignement
is wrapped in :class:`deel.puncc.api.splitting.IdSplitter` to produce iterables.

These methods produce **iterables** that are used by the :class:`ConformalPredictor` instance.

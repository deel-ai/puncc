.. _regression:

📈 Regression
==============


Currently implemented conformal prediction methods for regression are listed in
this page.

Each of these wrappers conformalize point-based or interval-based models that
are passed as argument in the object constructor. Such models **need** to
implement the :func:`fit` and :func:`predict` methods.
:doc:`Prediction module <prediction>` from the :doc:`API <api>` ensures the
compliance of models from various ML/DL libraries (such as Keras and scikit-learn) to **puncc**.

.. autoclass:: deel.puncc.regression.SplitCP

.. autoclass:: deel.puncc.regression.LocallyAdaptiveCP

.. autoclass:: deel.puncc.regression.CQR

.. autoclass:: deel.puncc.regression.CVPlus

.. autoclass:: deel.puncc.regression.EnbPI

.. autoclass:: deel.puncc.regression.AdaptiveEnbPI

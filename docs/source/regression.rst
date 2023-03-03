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

.. autoclass:: deel.puncc.Regression.SplitCP

.. autoclass:: deel.puncc.Regression.LocallyAdaptiveCP

.. autoclass:: deel.puncc.Regression.CQR

.. autoclass:: deel.puncc.Regression.CVPlus

.. autoclass:: deel.puncc.Regression.EnbPI

.. autoclass:: deel.puncc.Regression.AdaptiveEnbPI

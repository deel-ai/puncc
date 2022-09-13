.. _regression:

Regression
==========


Currently implemented conformal prediction methods for regression are listed in this page.

Each of these wrappers conformalize point-based or interval-based models that
are passed as argument in the object constructor. Such models **need** to
implement the :func:`fit` and :func:`predict` methods and operate on numpy arrays. :doc:`Prediction module <prediction>`
from the :doc:`API <api>` enables to use models even when these requirements are not met (*e.g.*,
using pytorch or tensorflow on time series).

.. autoclass:: deel.puncc.regression.SplitCP

.. autoclass:: deel.puncc.regression.LocallyAdaptiveCP

.. autoclass:: deel.puncc.regression.CQR

.. autoclass:: deel.puncc.regression.CvPlus

.. autoclass:: deel.puncc.regression.EnbPI
    :members: fit, predict

.. autoclass:: deel.puncc.regression.AdaptiveEnbPI
    :members: fit, predict

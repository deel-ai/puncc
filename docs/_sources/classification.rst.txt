.. _classification:

📊 Classification
==================

Currently implemented conformal prediction methods for classification are listed
in this page.

Each of the wrappers conformalize models that are passed as argument in the
object constructor. Such models **need** to implement the :func:`fit`
and :func:`predict` methods.
:doc:`Prediction module <prediction>` from the :doc:`API <api>` ensures the
compliance of models from various ML/DL libraries (such as Keras and scikit-learn) to **puncc**.

.. autoclass:: deel.puncc.classification.LAC

.. autoclass:: deel.puncc.classification.RAPS

.. autoclass:: deel.puncc.classification.APS

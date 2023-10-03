.. _anomaly_detection:

🚩 Anomaly detection
====================


Currently implemented conformal anomaly detectors are listed in this page.

Each of these wrappers calibrate the decision threshold for anomaly detectors 
that are passed as argument in the object constructor. Such models **need** to
implement the :func:`fit` and :func:`predict` methods.
:doc:`Prediction module <prediction>` from the :doc:`API <api>` ensures the
compliance of models from various ML/DL libraries (such as Keras and scikit-learn) to **puncc**.

.. autoclass:: deel.puncc.anomaly_detection.SplitCAD

<a id="anomaly-detection"></a>

# 🚩 Anomaly detection

Currently implemented conformal anomaly detectors are listed in this page.

Each of these wrappers calibrate the decision threshold for anomaly detectors
that are passed as argument in the object constructor. Such models **need** to
implement the `fit` and `predict` methods.
[Prediction module](prediction.md) from the [API](api.md) ensures the
compliance of models from various ML/DL libraries (such as Keras and scikit-learn) to **puncc**.

::: deel.puncc.anomaly_detection.SplitCAD
    options:
      members:
        - fit
        - predict
      inherited_members: false
      show_source: false

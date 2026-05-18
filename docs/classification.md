<a id="classification"></a>

# 📊 Classification

Currently implemented conformal prediction methods for classification are listed
in this page.

Each of the wrappers conformalize models that are passed as argument in the
object constructor. Such models **need** to implement the `fit`
and `predict` methods.
[Prediction module](prediction.md) from the [API](api.md) ensures the
compliance of models from various ML/DL libraries (such as Keras and scikit-learn) to **puncc**.

::: deel.puncc.classification.LAC
    options:
      show_source: false

::: deel.puncc.classification.ClasswiseLAC
    options:
      show_source: false

::: deel.puncc.classification.RAPS
    options:
      members:
        - fit
        - predict
      inherited_members: false
      show_source: false

::: deel.puncc.classification.APS
    options:
      show_source: false

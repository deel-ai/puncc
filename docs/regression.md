<a id="regression"></a>

# 📈 Regression

Currently implemented conformal prediction methods for regression are listed in
this page.

Each of these wrappers conformalize point-based or interval-based models that
are passed as argument in the object constructor. Such models **need** to
implement the `fit` and `predict` methods.
[Prediction module](prediction.md) from the [API](api.md) ensures the
compliance of models from various ML/DL libraries (such as Keras and scikit-learn) to **puncc**.

::: deel.puncc.regression.SplitCP
    options:
      show_source: false

::: deel.puncc.regression.LocallyAdaptiveCP
    options:
      show_source: false

::: deel.puncc.regression.LeverageWeightedCP
    options:
      members:
        - fit
      inherited_members: false
      show_source: false

::: deel.puncc.regression.CQR
    options:
      show_source: false

::: deel.puncc.regression.CVPlus
    options:
      members:
        - fit
        - predict
        - get_nonconformity_scores
      inherited_members: false
      show_source: false

::: deel.puncc.regression.EnbPI
    options:
      members:
        - fit
        - predict
      inherited_members: false
      show_source: false

::: deel.puncc.regression.AdaptiveEnbPI
    options:
      show_source: false

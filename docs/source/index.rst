Welcome to puncc's documentation!
=================================


**Puncc** (**P**\redictive **un**\certainty **c**\alibration and **c**\onformalization) is an open-source
Python library that integrates a collection of state-of-the-art conformal prediction algorithms
and related techniques for regression and classification problems.
It can be used with any predictive model to provide rigorous uncertainty estimations.
Under data exchangeability (or *i.i.d*), the generated prediction sets are guaranteed to cover the
true outputs within a user-defined error :math:`\alpha`.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   getting_started
   regression
   classification
   anomaly_detection
   api
   metrics
   plotting
   theory_overview

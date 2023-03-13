.. _theory_overview:

📈 Theory overview
===================

Uncertainty Quantification
--------------------------

In machine learning, we build predictive models from experience,
by choosing the right approach for the right problem, and from the accessible
data, via algorithms. Despite our best efforts, we can encounter some
underlying uncertainty that could stem from various sources or causes.

Typically, uncertainty in the machine learning process can be categorized into two types:
    - Aleatoric uncertainty, also known as statistical uncertainty, which is *irreducible* as due to the intrinsic randomness of the phenomenon being modeled
    - Epistemic uncertainty, also known as systematic uncertainty, which is *reducible* through additional information, e.g. via more data or better models

Depending on the application fields of machine learning models, uncertainty can have major impacts on performance and/or safety.

Conformal Prediction
--------------------

Conformal Prediction (CP) is a set of *distribution-free*, *model-agnostic* and
*non-asymptotic* methods to estimate uncertainty by constructing **valid** *prediction sets*, i.e. with guaranteed probability of marginal coverage.

Given a error rate (or significance level) :math:`\alpha \in (0,1)`, set by the user, a set of exchangeable (or more simply i.i.d.)
train data :math:`\{ (X_i, Y_i) \}_{i=1}^{n}` and test point
:math:`(X_{new}, Y_{new})` generated for a joint distribution :math:`\mathbb{P}_{XY}`,
a conformal prediction procedure builds prediction sets :math:`{C}_{\alpha}(\cdot)` so that:

.. math::

    \mathbb{P} \Big\{ Y_{new} \in {C}_{\alpha}\left(X_{new}\right) \Big\} \geq 1 - \alpha.


Over many calibration and test sets, :math:`{C}_{\alpha}(X_{new})` will contain
the observed values of :math:`Y_{new}` with frequency of *at least* :math:`(1-\alpha)`.

Within the conformal prediction framework, the inequality above holds for any model,
any data distribution :math:`\mathbb{P}_{XY}` and any finite sample sizes.
It is noteworthy that the coverage probability is marginalized over :math:`X`.
Therefore, it is likely to undercover conditionally to some specific regions in the space of :math:`X`.

Conformal prediction can act as a *post-processing procedure* to attain rigorous probability coverages,
as it can "conformalize" any existing predictor during or after training (black box predictors),
yielding marginally valid prediction sets.

In this page, we present the most common conformal prediction methods of the
literature used on regression and classification models. We also refer to
Angelopoulos and Bates [Angelopoulos2022]_ for a hands-on introduction to conformal prediction
and awesome conformal prediction `github <https://github.com/valeman/awesome-conformal-prediction>`_ for additional ressources.

In the following, let :math:`D_{train} = {(X_i, Y_i)}_{i=1..n_{train}} \sim P_{XY}`
be the training data and :math:`\alpha \in (0, 1)` the significance level (target maximum error rate).

Conformal Regression
--------------------

Split (inductive) Conformal
***************************
.. _theory splitcp:

The split (also called inductive) conformal prediction [Papadopoulos2002]_ [Lei2018]_ requires a hold-out calibration
dataset :math:`D_{calibration}` to estimate prediction errors and use them to build the prediction interval for a new sample :math:`X_{new}`.

Given a prediction model :math:`\widehat{f}` trained on :math:`D_{train}`, the algorithm is summarized in the following:

#. Choose a nonconformity score :math:`s`: :math:`R = s(\widehat{f}(X),Y)`. For example, one can pick the mean absolute deviation :math:`R = |\widehat{f}(X)-Y|`.
#. Compute the nonconformity scores on the calibration dataset: :math:`\bar{R} = \{R_i\}_{}`, for :math:`i=1,\dots,|D_{calibration}|`, where :math:`|D_{calibration}|` is the cardinality of :math:`D_{calibration}`.
#. Compute the error margin :math:`\delta_{\alpha}` as the :math:`(1-\alpha)(1 + \frac{1}{| D_{calibration} |})`-th empirical quantile of :math:`\bar{R}`.
#. Build the prediction interval :math:`\widehat{C}_{\alpha}(X_{new}) = \Big[ \widehat{f}(X_{new}) - \delta_{\alpha}^{f} \,,\, \widehat{f}(X_{new}) + \delta_{\alpha}^{f} \Big]`.

Note that this procedure yields a constant-width prediction interval centered on the point estimate :math:`\widehat{f}(X_{new})`.

In the literature, the split conformal procedure has been combined with different nonconformity scores,
which produced several methods. Some of them are presented hereafter.


Locally Adaptive Conformal Regression
#####################################
.. _theory lacp:

The locally adaptive conformal regression [Papadopoulos2008]_ relies on scaled nonconformity scores:

.. math::

    R_i = \frac{|\widehat{f}(X_i) - Y_i|}{\widehat{\sigma}(X_i)},

where :math:`\widehat{\sigma}(X_i)` is a measure of dispersion of the nonconformity scores at :math:`X_i`.
Usually, :math:`\widehat{\sigma}` is trained to estimate the absolute prediction
error :math:`|\widehat{f}(X)-Y|` given :math:`X=x`. The prediction interval is again
centered on :math:`\widehat{f}(X_{new})` but the margins are scaled w.r.t to the estimated local variability at :math:`Y | X = X_{new}`:

.. math::

    \widehat{C}_{\alpha}(X_{new})=
    \Big[ \widehat{f}(X_{new}) - \widehat{\sigma}(X_{new})\, \delta_{\alpha} \,,\, \widehat{f}(X_{new}) + \widehat{\sigma}(X_{new}) \, \delta_{\alpha} \Big].

The prediction intervals are therefore of variable width, which is more adaptive to heteroskedascity and
usually improve the conditional coverage. The price is the higher computational cost due to fitting two functions
:math:`\widehat{f}` and :math:`\widehat{\sigma}`, on the proper training set.


Conformalized Quantile Regression (CQR)
#######################################
.. _theory cqr:

Split conformal prediction can be extended to `quantile predictors <https://en.wikipedia.org/wiki/Quantile_regression>`_  :math:`q(\cdot)`
by using the nonconformity score
:math:`R_i^{} = \text{max}\{ \widehat{q}_{\alpha_{lo}}(X_i) - Y_i, Y_i - \widehat{q}_{1 - \alpha_{hi}}(X_i)\}`,
for :math:`i=1,\dots,|D_{calibration}|`. :math:`\widehat{q}_{\alpha_{lo}}` and :math:`\widehat{q}_{\alpha_{hi}}` are
the predictors of the :math:`\alpha_{lo}` *-th* and :math:`(1-\alpha_{hi})` *-th* quantiles of :math:`Y | X`, respectively.

.. note::

    It is common to split evenly :math:`\alpha` as: :math:`\alpha_{lo} = \frac{\alpha}{2}` and :math:`\alpha_{hi}=1-\frac{\alpha}{2}`, but the users are free to do otherwise.

The procedure,
named *Conformalized Quantile Regression* [Romano2019]_, yields the
following prediction interval:

.. math::

    \widehat{C}_{\alpha}(X_{new}) = \Big[ \widehat{q}_{\alpha_{lo}}(X_{new}) - \delta_{\alpha} \,,\, \widehat{q}_{1 - \alpha_{hi}}(X_{new}) + \delta_{\alpha} \Big].

When data are exchangeable, the correction margin :math:`\delta_{\alpha}` guarantees finite-sample marginal coverage for the quantile predictions, and this holds also for misspecified (i.e. "bad") predictors.

If the fitted :math:`\widehat{q}_{\alpha_{lo}}` and :math:`\widehat{q}_{1-\alpha_{hi}}` approximate well (empirically), the conditional distribution :math:`Y | X` of the data, we will get a small margin :math:`\delta_{\alpha}`: this means that on average, the prediction errors on the :math:`D_{calibration}` were small.
Also, if the base predictors have strong theoretical properties, our CP procedure  we inherit the theoretical properties of :math:`\widehat{q}_{}(\cdot)`.
We could have an asymptotically, conditionally accurate predictor and also have a theoretically valid, distribution-free guarantee on the marginal coverage!


..
    Weighted Split Conformal
    ########################
    .. _theory weightedcp:

Cross-validation + (CV+)
************************
.. _theory cvplus:

Ensemble Batch Prediction Intervals (EnbPI)
*******************************************
.. _theory enbpi:


Conformal Classification
------------------------

Adaptive Prediction Sets (APS)
*******************************************
.. _theory aps:

Regularized Adaptive Prediction Sets (RAPS)
*******************************************
.. _theory raps:


References
----------

.. [Angelopoulos2022] Angelopoulos, A.N. and Bates, S., 2021. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv preprint arXiv:2107.07511.
.. [Romano2019] Romano, Y., Patterson, E. and Candes, E., 2019. Conformalized quantile regression. Advances in neural information processing systems, 32.
.. [Papadopoulos2008] Papadopoulos, H., Gammerman, A. and Vovk, V., 2008, February. Normalized nonconformity measures for regression conformal prediction. In Proceedings of the IASTED International Conference on Artificial Intelligence and Applications (AIA 2008) (pp. 64-69).
.. [Papadopoulos2002] Papadopoulos, H., Proedrou, K., Vovk, V. and Gammerman, A., 2002. Inductive confidence machines for regression. In Machine Learning: ECML 2002: 13th European Conference on Machine Learning Helsinki, Finland, August 19–23, 2002 Proceedings 13 (pp. 345-356). Springer Berlin Heidelberg.
.. [Lei2018] Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R.J. and Wasserman, L., 2018. Distribution-free predictive inference for regression. Journal of the American Statistical Association, 113(523), pp.1094-1111.

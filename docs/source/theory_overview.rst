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
    * Aleatoric uncertainty, also known as statistical uncertainty, which is *irreducible* as due to the intrinsic randomness of the phenomenon being modeled
    * Epistemic uncertainty, also known as systematic uncertainty, which is *reducible* through additional information, e.g. via more data or better models

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
Therefore, it is likely to undercover conditionnaly to some specific regions in the space of :math:`X`.

Conformal prediction can act as a *post-processing procedure* to attain rigorous probability coverages,
as it can "conformalize" any existing predictor during or after training (black box predictors),
yielding marginally valid prediction sets.

Conformal Regression
--------------------

Split Conformal Prediction
**************************
.. _theory splitcp:

Locally Adaptive Conformal Prediction
*************************************
.. _theory lacp:

Conformalized Quantile Regression (CQR)
***************************************
.. _theory cqr:

Cross-validation + (CV+)
************************
.. _theory cvplus:


Conformal Classification
------------------------

Adaptive Prediction Sets (APS)
*******************************************
.. _theory aps:

Regularized Adaptive Prediction Sets (RAPS)
*******************************************
.. _theory raps:

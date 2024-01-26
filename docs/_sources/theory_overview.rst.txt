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

Given an error rate (or significance level) :math:`\alpha \in (0,1)`, set by the user, a set of exchangeable (or more simply i.i.d.)
train data :math:`\{ (X_i, Y_i) \}_{i=1}^{n}` and test point
:math:`(X_{new}, Y_{new})` generated for a joint distribution :math:`\mathbb{P}_{XY}`,
a conformal prediction procedure builds prediction sets :math:`{C}_{\alpha}(\cdot)` so that:

.. math::

    \mathbb{P} \left( Y_{new} \in {C}_{\alpha}\left(X_{new}\right) \right) \geq 1 - \alpha.


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
Angelopoulos and Bates [Angelo2022]_ for a hands-on introduction to conformal prediction
and `awesome conformal prediction github <https://github.com/valeman/awesome-conformal-prediction>`_ for additional ressources.

In the following, let :math:`D_{train} = {(X_i, Y_i)}_{i=1..n_{train}} \sim P_{XY}`
be the training data and :math:`\alpha \in (0, 1)` the significance level (target maximum error rate).

Conformal Regression
--------------------

Split (inductive) Conformal
***************************
.. _theory splitcp:

The split (also called inductive) conformal prediction [Papado2002]_ [Lei___2018]_ requires a hold-out calibration
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

The locally adaptive conformal regression [Papado2008]_ relies on scaled nonconformity scores:

.. math::

    R_i = \frac{|\widehat{f}(X_i) - Y_i|}{\widehat{\sigma}(X_i)},

where :math:`\widehat{\sigma}(X_i)` is a measure of dispersion of the nonconformity scores at :math:`X_i`.
Usually, :math:`\widehat{\sigma}` is trained to estimate the absolute prediction
error :math:`|\widehat{f}(X)-Y|` given :math:`X=x`. The prediction interval is again
centered on :math:`\widehat{f}(X_{new})` but the margins are scaled w.r.t the estimated local variability at :math:`Y | X = X_{new}`:

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
by using the nonconformity score:

.. math::

    R_i^{} = \text{max}\{ \widehat{q}_{\alpha_{lo}}(X_i) - Y_i, Y_i - \widehat{q}_{1 - \alpha_{hi}}(X_i)\},

for :math:`i=1,\dots,|D_{calibration}|`. :math:`\widehat{q}_{\alpha_{lo}}` and :math:`\widehat{q}_{1-\alpha_{hi}}` are
the predictors of the :math:`\alpha_{lo}` *-th* and :math:`(1-\alpha_{hi})` *-th* quantiles of :math:`Y | X`, respectively.
For example, if we set :math:`\alpha = 0.1`, we would fit two predictors :math:`\widehat{q}_{0.05}(\cdot)` and :math:`\widehat{q}_{0.95}(\cdot)` on training data :math:`D_{train}` and compute the scores on :math:`D_{calibration}`.


.. note::

    It is common to split evenly :math:`\alpha` as: :math:`\alpha_{lo} = \alpha_{hi}= \frac{\alpha}{2}`, but users are free to do otherwise.

The procedure, named *Conformalized Quantile Regression* [Romano2019]_, yields the following prediction interval:

.. math::

    \widehat{C}_{\alpha}(X_{new}) = \Big[ \widehat{q}_{\alpha_{lo}}(X_{new}) - \delta_{\alpha} \,,\, \widehat{q}_{1 - \alpha_{hi}}(X_{new}) + \delta_{\alpha} \Big].

When data are exchangeable, the correction margin :math:`\delta_{\alpha}` guarantees finite-sample marginal coverage for the quantile predictions, and this holds also for misspecified (i.e. "bad") predictors.

If the fitted :math:`\widehat{q}_{\alpha_{lo}}` and :math:`\widehat{q}_{1-\alpha_{hi}}` approximate (empirically) well  the conditional distribution :math:`Y | X` of the data, we will get a small margin :math:`\delta_{\alpha}`: this means that on average, the prediction errors on the :math:`D_{calibration}` were small.

Also, if the base predictors have strong theoretical properties, our CP procedure inherits these properties of :math:`\widehat{q}_{}(\cdot)`.
We could have an asymptotically, conditionally accurate predictor and also have a theoretically valid, distribution-free guarantee on the marginal coverage!


..
    Weighted Split Conformal
    ########################
    .. _theory weightedcp:


Cross-validation+ (CV+), Jackknife+
************************************
.. _theory cvplus:

The `leave-one-out (LOO) and the k-fold cross-validation <https://en.wikipedia.org/wiki/Cross-validation_(statistics)>`_ are well known schemes used to estimate regression residuals on out-of-sample data.
As shown below, one first splits the data into K partitions and then *holds out* a partition at a time to compute errors (nonconformity scores, in our case).
Following this principle, [Barber2021]_ introduced the LOO *jackknife+* (JP) and the k-fold *Cross-validation+* (CV+).
With these methods, one does *not need* a dedicated calibration set.

.. image:: img/k-fold-scheme.png
   :width: 600
   :align: center

|

The CV+ algorithm goes as follows.
Let :math:`n = |D_{train}|`, and let :math:`D_{train}` be partitioned disjointly into the sets :math:`S_1, S_2, \dots, S_K`.
Each training point :math:`(X_i,Y_i) \in D_{train}` belongs to one partition, noted as :math:`S_{k(i)}`.

At training, we fit and store in memory :math:`K` models, referred to as :math:`\widehat{f}_{-S_{K}}` to indicate that it was fitted using all data points *except* those in partition :math:`S_{K}`.
Then, the conformalization step boils down to computing, for each :math:`(X_i,Y_i) \in D_{train}`, the score:

.. math::
    R_i^{CV} = | Y_i - \widehat{f}_{-S_{k(i)}}(X_i)|, i=1, \dots, n

If :math:`K = n`, we obtain the *Jackknife+*, **leave-one-out** version of the algorithm.


**Inference**

.. Let :math:`(X_{new}, Y_{new})` be a test point, where :math:`Y_{new}` is not observable at inference time.

The lower and upper bounds of the prediction interval are given by:

    1. Compute :math:`\bar{R}_{L} = \{ \widehat{f}_{-S_{k(i)}}(X_{new}) - R_i^{CV} \}_{i=1}^{n}`
    2. :math:`\widehat{L}_{\alpha}(X_{new}) = \lfloor \alpha (n+1) \rfloor`-th smallest value in :math:`\bar{R}_{L}` (lower bound)
    3. Compute :math:`\bar{R}_{U} = \{ \widehat{f}_{-S_{k(i)}}(X_{new}) + R_i^{CV} \}_{i=1}^{n}`
    4. :math:`\widehat{U}_{\alpha}(X_{new}) = \lceil (1-\alpha) (n+1) \rceil`-th smallest value in :math:`\bar{R}_{U}` (upper bound)


.. math::

    \widehat{C}_{\alpha}(X_{new}) = \Big[ \widehat{L}_{\alpha}(X_{new}), \widehat{U}_{\alpha}(X_{new}) \Big].


Ensemble Batch Prediction Intervals (EnbPI)
*******************************************
.. _theory enbpi:

Source: [Xu____2021]_

TBC

.. Introduced in [Xu2021]_, the EnbPI algorithms builds prediction intervals for time series data of the form :math:`Y_t = f(X_t) + \epsilon_t`, where :math:`\epsilon_t` are identically distributed.
.. Unlike the proper conformal algorithms seen above, EnbPI requires some additional hypothesis to attain the coverage guarantee.

..
    Summary: guarantees
    *******************
    .. _theory guarantees:

    * split
    * JP



Conformal Classification
------------------------

Classification aims to assign a label or category :math:`\widehat{y}` to a given 
input :math:`X`. In practice, as the prediction could be subject to 
uncertainty, conformal methods build **sets of likely labels** :math:`\widehat{C}(X)` that 
includes the true label :math:`y` with high probability. More formally, the 
guarantee is written as follows:

.. math::

    \mathbb{P}\left(y \in \widehat{C}(X) \right) \geq 1 - \alpha, 

such that :math:`\alpha` is the maximum error rate tolerated by the user 
(for example 10\%).

One can argue that most classification models come with a probability 
(uncertainty) estimate for each possible class given an input :math:`X`. This 
is for example the case when using neural networks with softmax outputs. Let's 
first formalize the notation then briefly discuss why uncertainty estimated this 
way is not reliable (calibration issue).

Let :math:`\left(\pi_1(X), \pi_2(X), \dots, \pi_k(X) \right) = \widehat{f}(X) \in [0,1]^k` 
be the output scores (e.g. softmax) of classifier :math:`\widehat{f}` and 
:math:`\left(\pi_{(1)}(x), \pi_{(2)}(x), \dots, \pi_{(k)}(x)\right)` the sequence 
of scores sorted in decreasing order (most likely to least likely class). 
The :math:`j`-th score :math:`\pi_{j \in \{1, 2, \dots, k\}}(x) \in [0,1]` is 
the score that quantifies how likely it is that :math:`j` is the true (unknown) 
label for :math:`X` according to the classifier :math:`\widehat{f}`. The 
prediction is chosen to be the **most likely** class :math:`(1)` associated to 
:math:`\pi_{(1)}(X)`.

Ideally, if the model :math:`\widehat{f}` has a good grasp of uncertainty,
then it is well calibrated. This means that each predicted score :math:`\pi_i(X)` 
is a precise approximation of the probability of observing class :math:`i` given 
:math:`X`:

.. math::

    \mathbb{P}(i=y \,|\, \pi_i(X) = p) = p

When the model is well-calibrated, conformal prediction is straightforward. 
This involves including the top-ranked classes starting by the most likely one up 
to rank :math:`i`, such that the culumated probability mass exceeds the 
desired confidence level :math:`1-\alpha`:

.. math::

    \widehat{C}(X) = \{(1), (2), \dots, (j); \sum_{j=1}^{i}\pi_{(j)}(X) \ge 1-\alpha\}

In reality, classification models are often poorly calibrated. Therefore, the 
straightforward approach will lead to prediction sets that are either overconfident or 
underconfident, failing to rigously cover the true labels. To address this issue, 
researchers in the conformal prediction community have proposed several methods 
to obtain theoretical guarantees in broader settings.


Adaptive Prediction Sets (APS)
*******************************************
.. _theory aps:

Adaptive Prediction Sets (APS), proposed by [Romano2020]_, is the equivalent of 
:ref:`CQR <theory cqr>` for classification. For a user-defined :math:`\alpha`, 
the main idea is to provide a threshold :math:`\tau` of cumulative probability mass required 
to obtain the desired coverage :math:`1-\alpha`. The prediction set is 
constructed by including the classes ranked by their predicted scores in 
decreasing order, until their cumulative scores exceed :math:`\tau`.

Let :math:`\left(\pi_1(X), \pi_2(X), \dots, \pi_k(X) \right) = \widehat{f}(X) \in [0,1]^k` 
be the output scores (e.g. softmax) of classifier :math:`\widehat{f}` and 
:math:`\left(\pi_{(1)}(x), \pi_{(2)}(x), \dots, \pi_{(k)}(x)\right)` the sequence 
of scores sorted in decreasing order (most likely to least likely class). 
The APS defines the nonconformity score as the cumulative sum 
of the classifier's ranked scores, from the most likely up to the score 
associated to true label :math:`y`:      

.. math::

    R = \sum_{j=(1)}^{(i)=y}\pi_{(j)}(X)

The procedure aligns with split (inductive) conformal prediction:

#. Compute nonconformity scores on the calibration dataset: :math:`\bar{R} = \{R_i\}_{}`, for :math:`i=1,\dots,n`, where :math:`n` is the size of :math:`D_{calibration}`.
#. Compute the calibrated threshold :math:`\tau_{\alpha}` as the :math:`(1-\alpha)(1 + \frac{1}{n})`-th empirical quantile of :math:`\bar{R}`.
#. Build the prediction interval :math:`\widehat{C}_{\alpha}(X_{new}) = \{(1), (2), \dots, (c)\}` such that :math:`c = \inf\{(i), \text{ }\sum_{j=(1)}^{(i)}\pi_{(j)}(X_{new}) \ge \tau_{\alpha}\}`.

.. note::

    In the original paper, the nonconformity scores and the prediction set are  
    functions of a random variable :math:`u \sim Uniform(0,1)` (independent of :math:`X`). 
    This enables to obtain a tighter coverage close to :math:`1-\alpha`. In puncc, 
    you can choose to activate or not on the randomization.   


Regularized Adaptive Prediction Sets (RAPS)
*******************************************
.. _theory raps:

Source: [Angelo2021]_

TBC

Conformal Anomaly Detection
---------------------------
Consider a trained anomaly detector :math:`\hat{f}` that predicts if a data point :math:`X` 
is anomalous based on an anomaly score :math:`\hat{s}` --or more generally any score 
function that evaluates the strangeness or inadequacy of a data point-- such 
that:  


.. math::
    \hat{f}(X) =
    \begin{cases}
        \text{not anomaly} & \text{if $\hat{s}(X) \le M_\text{threshold}$}\\
      \text{anomaly} & \text{if $\hat{s}(X) > M_\text{threshold}$}
    \end{cases}    


where :math:`M_\text{threshold}` is the anomaly score threshold, that is the lowest 
anomaly score associated to a data point to be considered anomalous. It is a 
critical parameter in anomaly detection, as it determines which data points are 
flagged as anomalies. 


Conformal Anomaly Detection (CAD) is a procedure based on conformal prediction and was introduced by R. Laxhammar et al. [Laxham2015]_. 
It enables to calibrate the anomaly threshold to control the False Detection Rate (FDR), 
i.e. to enforce an upper bound on the false alarm rate under a user-specified limit :math:`\alpha`. 

CAD is based on split conformal prediction with the following particularities:

- It works in an unsupervised setup: the calibration dataset does not need to be labeled.
- For a data point :math:`X`, the nonconformity score consists of the anomaly score :math:`\hat{s}(X)`.  

Given a calibration dataset :math:`D_{calibration}=\{(X_i)\}_{i=1..n}`, CAD procedure is as follows:

#. Compute nonconformity scores on the calibration dataset: :math:`\bar{S} = \{\hat{s}(X_i)\}_{}`, for :math:`i=1,\dots,n`.
#. Compute :math:`\hat{q}_{\alpha}` as the :math:`(1-\alpha)(1 + \frac{1}{n})`-th empirical quantile of :math:`\bar{S}`.
#. Predict anomalies using the calibrated anomaly detector :math:`\hat{C}`:


.. math::
    \hat{C}_{\alpha}(X_{new}) =
    \begin{cases}
        \text{not anomaly} & \text{if $\hat{s}(X_{new}) \le \hat{q}_{\alpha}$}\\
      \text{anomaly} & \text{if $\hat{s}(X_{new}) > \hat{q}_{\alpha}$}
    \end{cases}  

When :math:`X_1, \dots, X_n, X_{new}` are i.i.d, :math:`\hat{C}` guarantees the FDR control property:

.. math::
    \mathbb{P}\left(\hat{C}_{\alpha}(X_{new}) = \text{anomaly}\right) \le \alpha


Conformal Object Detection
--------------------------
.. _theory splitboxwise:

Source: [deGran2022]_

TBC

References
----------

.. [Angelo2021] Angelopoulos, A. N., Bates, S., Jordan, M., & Malik, J (2021). Uncertainty Sets for Image Classifiers using Conformal Prediction. In Proceedings of ICLR 2021. https://openreview.net/forum?id=eNdiU_DbM9
.. [Angelo2022] Angelopoulos, A.N. and Bates, S., (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv preprint arXiv:2107.07511. https://arxiv.org/abs/2107.07511
.. [Barber2021] Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2021). Predictive inference with the jackknife+. Ann. Statist. 49 (1) 486 - 507, February 2021. https://arxiv.org/abs/1905.02928
.. [Laxham2015] Laxhammar, R., & Falkman, G. (2015). Inductive conformal anomaly detection for sequential detection of anomalous sub-trajectories. Annals of Mathematics and Artificial Intelligence, 74, 67-94.
.. [Lei___2018] Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J. and Wasserman, L., (2018). Distribution-free predictive inference for regression. Journal of the American Statistical Association, 113(523), pp.1094-1111. https://arxiv.org/abs/1604.04173
.. [Papado2002] Papadopoulos, H., Proedrou, K., Vovk, V. and Gammerman, A., (2002). Inductive confidence machines for regression. In Proceedings of ECML 2002, Springer. https://link.springer.com/chapter/10.1007/3-540-36755-1_29
.. [Papado2008] Papadopoulos, H., Gammerman, A. and Vovk, V., (2008). Normalized nonconformity measures for regression conformal prediction. In Proceedings of the IASTED International Conference on Artificial Intelligence and Applications (AIA 2008) (pp. 64-69).
.. [deGran2022] de Grancey, F., Adam, J.L., Alecu, L., Gerchinovitz, S., Mamalet, F. and Vigouroux, D., 2022, June. Object detection with probabilistic guarantees: A conformal prediction approach. In International Conference on Computer Safety, Reliability, and Security.
.. [Romano2019] Romano, Y., Patterson, E. and Candes, E., (2019). Conformalized quantile regression. In Proceedings of NeurIPS, 32. https://arxiv.org/abs/1905.03222
.. [Romano2020] Romano, Y., Sesia, M., & Candes, E. (2020). Classification with valid and adaptive coverage. In Proceedings of NeurIPS, 33. https://arxiv.org/abs/2006.02544
.. [Xu____2021] Xu, C. & Xie, Y.. (2021). Conformal prediction interval for dynamic time-series. Proceedings of ICML 2021. https://proceedings.mlr.press/v139/xu21h.html.

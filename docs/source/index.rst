.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   regression
   api

Welcome to puncc's documentation!
=================================

*Predictive UNcertainty Calibration and Conformalization* (*PUNCC*) is an open-source
library that enables ad-hoc integration of DL/ML models into a theoretically 
sound uncertainty evaluation framework based on conformal prediction. 
Prediction regions are constructed with guaranteed marginal coverage probability 
with respect to a nominal level of error :math:`\alpha`. 


Installation
============

* First, clone the repo:

.. code-block:: bash

   git clone ssh://git@forge.deel.ai:22012/statistic-guarantees/puncc.git


It is recommended to create a virtual environment for running **puncc**. 

* To start with, install `virtualenv` (if not already done): 

.. code-block:: bash

   pip install virtualenv

* Then create a virtual environment named `env-puncc` and activate it:

.. code-block:: bash

   virtualenv env-puncc
   # Under Mac OS / Linux
   source env-puncc/bin/activate

Now we are ready to install `puncc` and its requirements.

* For users:

.. code-block:: bash

   pip install -e .[interactive]

* For developers:

.. code-block:: bash
   
   pip install -e .[dev]

Finally, to use the current virtual environment in a jupyter notebook, 
make sure to add it:

.. code-block:: bash
   
   python -m ipykernel install --user --name=env-puncc   
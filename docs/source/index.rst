Welcome to puncc's documentation!
=================================


**Puncc** (**P**\redictive **un**\certainty **c**\alibration and **c**\onformalization) is an open-source 
Python library that integrates a collection of state-of-the-art conformal prediction algorithms 
and related techniques for regression and classification problems. 
It can be used with any predictive model to provide rigorous uncertainty estimations. 
Under data exchangeability (or *i.i.d*), the generated prediction sets are guaranteed to cover the 
true outputs within a user-defined error :math:`\alpha`.

Installation
============

* First, clone the repo:

.. code-block:: bash

   git clone https://github.com/deel-ai/puncc.git


For users
---------

.. code-block:: bash

   pip install -e .[interactive]


You can alternatively use the makefile to automatically create a virtual environment
`puncc-user-env` and install user requirements:

.. code-block:: bash

   make install-user


Finally, to use the current virtual environment in a jupyter notebook,
make sure to add it:

.. code-block:: bash

   python -m ipykernel install --user --name=puncc-user-env

For developpers
---------------

.. code-block:: bash

   pip install -e .[dev]

You can alternatively use the makefile to automatically create a virtual environment
`puncc-dev-env` and install the dev requirements:

.. code-block:: bash

   make prepare-dev

.. toctree::
   :maxdepth: 1
   :caption: Contents

   theory_overview
   getting_started
   regression
   classification
   api
   metrics
   plotting
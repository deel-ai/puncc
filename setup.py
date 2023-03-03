# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from os import path

import setuptools


# read the contents of your README file

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "joblib>=1.1.0",
    "matplotlib>=3.5.1",
    "numpy>=1.22.3",
    "scikit-learn>=1.0.2",
    "scipy>=1.8.0",
    "seaborn>=0.11.2",
    "tqdm>=4.63.0",
]

interactive_requirements = [
    "ipykernel>=6.9.1",
    "ipython>=8.1.1",
    "jupyter-client>=7.1.2",
    "jupyter-core>=4.9.2",
    "jupyterthemes>=0.20.0",
    "matplotlib-inline>=0.1.3",
]

dev_requirements = [
    "flake8>=4.0.1",
    "pytest>=7.1.2",
    "pytest-cov>=3.0.0",
    "black==22.3.0",
    "pre-commit>=2.20.0",
    "sphinx>=5.1.1",
    "sphinx-rtd-theme>=1.0.0",
    "sphinx-autodoc-typehints>=1.19.2",
    "tensorflow",
    "tox>=3.25.1",
]

setuptools.setup(
    name="puncc",
    version="0.1dev",
    author=", ".join(["Mouhcine Mendil", "Luca Mossina"]),
    author_email=", ".join(
        [
            "mouhcine.mendil@irt-saintexupery.com",
            "luca.mossina@irt-saintexupery.com",
        ]
    ),
    description="Predictive Uncertainty Calibration and Conformalization Lib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deel-ai/puncc",
    packages=setuptools.find_namespace_packages(include=["deel.*"]),
    install_requires=requirements,
    extras_require={
        "interactive": interactive_requirements,
        "dev": dev_requirements,
    },
    license="MIT",
    python_requires=">=3.8",
)

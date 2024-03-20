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
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-learn",
    "tqdm",
]

interactive_requirements = [
    "ipykernel",
]

dev_requirements = [
    "flake8",
    "pylint",
    "pytest",
    "pytest-cov",
    "black",
    "pre-commit",
    "sphinx",
    "sphinx-rtd-theme",
    "sphinx-autodoc-typehints",
    "tensorflow",
    "tensorflow-addons",
    "tox",
]

setuptools.setup(
    name="puncc",
    version="0.7.7",
    author=", ".join(["Mouhcine Mendil", "Luca Mossina", "Joseba Dalmau"]),
    author_email=", ".join(
        [
            "mouhcine.mendil@irt-saintexupery.com",
            "luca.mossina@irt-saintexupery.com",
            "joseba.dalmau@irt-saintexupery.com",
        ]
    ),
    description="Predictive UNcertainty Calibration and Conformalization Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deel-ai/puncc",
    packages=setuptools.find_namespace_packages(include=["deel.*"]),
    install_requires=requirements,
    extras_require={
        "interactive": interactive_requirements,
        "dev": dev_requirements,
    },
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

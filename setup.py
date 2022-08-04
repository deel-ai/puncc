from setuptools import setup, find_packages

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
    "pre-commit",
]

setup(
    name="puncc",
    version="0.1dev",
    description="Predictive Uncertainty Calibration and Conformalization Lib",
    author="IRT Saint Exupery",
    author_email="mouhcine.mendil@irt-saintexupery.com",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "interactive": interactive_requirements,
        "dev": dev_requirements,
    },
    license="MIT",
)

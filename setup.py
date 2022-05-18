from setuptools import setup, find_packages

setup(
    name="puncc",
    version="1.0dev",
    description="Predictive Uncertainty Calibration and Conformalization Library",
    author="IRT Saint Exupery",
    author_email="mouhcine.mendil@irt-saintexupery.com",
    packages=find_packages(),
    install_requires=[
        "ipykernel>=6.9.1",
        "ipython>=8.1.1",
        "joblib>=1.1.0",
        "jupyter-client>=7.1.2",
        "jupyter-core>=4.9.2",
        "jupyterthemes>=0.20.0",
        "matplotlib>=3.5.1",
        "matplotlib-inline>=0.1.3",
        "numpy>=1.22.3",
        "scikit-learn>=1.0.2",
        "scipy>=1.8.0",
        "seaborn>=0.11.2",
        "tqdm>=4.63.0",
    ],
    license="MIT",
)

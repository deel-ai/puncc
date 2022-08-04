import numpy as np
from sklearn import datasets
import pytest


@pytest.fixture
def diabetes_data():

    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]  # type: ignore

    # Split the data into training/testing sets
    X_train = diabetes_X[:-100]
    X_test = diabetes_X[-100:]

    # Split the targets into training/testing sets
    y_train = diabetes_y[:-100]
    y_test = diabetes_y[-100:]

    return X_train, X_test, y_train, y_test

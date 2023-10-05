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
import numpy as np
import pytest
from sklearn import datasets
from sklearn.datasets import make_moons
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


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


@pytest.fixture
def mnist_data():
    # Load MNIST Database

    # Split train and test datasets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshaping and standardization
    X_train = X_train.reshape((len(X_train), 28 * 28))
    X_train = X_train.astype("float32") / 255
    X_test = X_test.reshape((len(X_test), 28 * 28))
    X_test = X_test.astype("float32") / 255

    # One hot encoding of classes
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat


@pytest.fixture
def rand_reg_data():
    X_pred_calib = 10 * np.random.randn(100, 4)
    X_pred_test = 10 * np.random.randn(100, 4)
    X_test = 10 * np.random.randn(100, 4)
    y_pred_calib = 4 * np.random.randn(100) + 1
    y_calib = 2 * np.random.randn(100) + 1
    y_pred_test = 4 * np.random.randn(100) + 2
    y_test = 2 * np.random.randn(100) + 1

    return y_pred_calib, y_calib, y_pred_test, y_test


@pytest.fixture
def rand_class_data():
    X_pred_calib = 10 * np.random.randn(100, 4)
    X_pred_test = 10 * np.random.randn(100, 4)
    X_test = 10 * np.random.randn(100, 4)
    y_pred_calib = np.random.random_sample(size=(100, 5))
    # Normalized logits
    y_pred_calib /= np.sum(y_pred_calib, axis=-1)[:, np.newaxis]
    y_calib = np.random.randint(4, size=100)
    y_pred_test = np.random.random_sample(size=(100, 5))
    # Normalized logits
    y_pred_test /= np.sum(y_pred_test, axis=-1)[:, np.newaxis]
    y_test = np.random.randint(4, size=100)

    return y_pred_calib, y_calib, y_pred_test, y_test


@pytest.fixture
def rand_anomaly_detection_data():
    # First, we generate the two moons dataset
    dataset = 4 * make_moons(n_samples=1000, noise=0.05, random_state=0)[
        0
    ] - np.array([0.5, 0.25])

    # Generate uniformly new data point
    rng = np.random.RandomState(42)
    new_samples = rng.uniform(low=-6, high=6, size=(150, 2))

    return dataset, new_samples

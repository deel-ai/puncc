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

from tests._utils import tensor, to_numpy

from deel.puncc.core.splitters import (
    IdSplitter,
    KFoldSplitter,
    RandomSplitter,
)


@pytest.fixture
def sample_data():
    """Return sample-aligned features and labels."""
    y = np.arange(100)
    X = np.column_stack((y, -y))
    return tensor(X, "int32"), tensor(y, "int32")


def test_id_splitter_initialization(sample_data):
    X, y = sample_data

    X_fit, y_fit = X[:80], y[:80]
    X_calib, y_calib = X[80:], y[80:]

    splitter = IdSplitter(
        X_fit=X_fit,
        y_fit=y_fit,
        X_calib=X_calib,
        y_calib=y_calib,
    )

    splits = splitter.split()

    assert len(splits) == 1

    (X_fit_out, y_fit_out), (X_calib_out, y_calib_out) = splits[0]

    np.testing.assert_array_equal(to_numpy(X_fit_out), to_numpy(X_fit))
    np.testing.assert_array_equal(to_numpy(y_fit_out), to_numpy(y_fit))
    np.testing.assert_array_equal(to_numpy(X_calib_out), to_numpy(X_calib))
    np.testing.assert_array_equal(to_numpy(y_calib_out), to_numpy(y_calib))


def test_bad_initialization(sample_data):
    X, y = sample_data

    with pytest.raises(ValueError, match="same number of samples"):
        IdSplitter(
            X_fit=X[:20],
            y_fit=y[:2],
            X_calib=X[20:30],
            y_calib=y[20:30],
        )

    with pytest.raises(ValueError, match="K must be"):
        KFoldSplitter(K=-10, random_state=0)

    with pytest.raises(ValueError, match="Ratio must be"):
        RandomSplitter(ratio=2, random_state=0)


def test_kfold_splitter_output_shape(sample_data):
    X, y = sample_data

    splitter = KFoldSplitter(
        K=10,
        random_state=0,
    )

    splits = splitter.split(X=X, y=y)

    assert len(splits) == 10

    calibration_labels = []

    for (X_fit, y_fit), (X_calib, y_calib) in splits:
        assert X_fit.shape == (90, 2)
        assert y_fit.shape == (90,)
        assert X_calib.shape == (10, 2)
        assert y_calib.shape == (10,)

        # Splitting must preserve feature-label alignment.
        np.testing.assert_array_equal(to_numpy(X_fit[:, 0]), to_numpy(y_fit))
        np.testing.assert_array_equal(to_numpy(X_calib[:, 0]), to_numpy(y_calib))

        calibration_labels.append(to_numpy(y_calib))

    # Each sample must appear in exactly one calibration fold.
    all_calibration_labels = np.concatenate(calibration_labels)

    np.testing.assert_array_equal(
        np.sort(all_calibration_labels),
        to_numpy(y),
    )


def test_random_splitter_output_shape(sample_data):
    X, y = sample_data

    splitter = RandomSplitter(
        ratio=0.1,
        random_state=0,
    )

    splits = splitter.split(X=X, y=y)

    assert len(splits) == 1

    (X_fit, y_fit), (X_calib, y_calib) = splits[0]

    assert X_fit.shape == (10, 2)
    assert y_fit.shape == (10,)
    assert X_calib.shape == (90, 2)
    assert y_calib.shape == (90,)

    # Splitting must preserve feature-label alignment.
    np.testing.assert_array_equal(to_numpy(X_fit[:, 0]), to_numpy(y_fit))
    np.testing.assert_array_equal(to_numpy(X_calib[:, 0]), to_numpy(y_calib))

    # No sample should be lost or duplicated.
    all_labels = np.concatenate((to_numpy(y_fit), to_numpy(y_calib)))

    np.testing.assert_array_equal(
        np.sort(all_labels),
        to_numpy(y),
    )

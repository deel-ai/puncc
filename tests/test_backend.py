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
import os
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "backend",
    ["numpy", "tensorflow", "torch", "jax"],
)
def test_weighted_quantile_across_backends(backend):
    if backend != "numpy" and find_spec(backend) is None:
        pytest.skip(f"{backend} is not installed.")

    # A fresh process is required to select a Keras backend.
    script = """
import numpy as np

from deel.puncc.config import set_backend

set_backend(BACKEND)

from deel.puncc.backend import ops


def check(actual, expected, shape):
    actual = np.asarray(ops.convert_to_numpy(actual))
    assert actual.shape == shape, (actual.shape, shape)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


values = ops.convert_to_tensor(
    [
        [3.0, 7.0],
        [1.0, 5.0],
        [4.0, 9.0],
        [2.0, 6.0],
    ],
    dtype="float32",
)

# Sorted columns:
# [1, 2, 3, 4] and [5, 6, 7, 9].

# Scalar quantile, including TensorFlow's searchsorted path.
median = ops.weighted_quantile(
    values,
    0.5,
    axis=0,
)
check(median, [2.0, 6.0], (2,))

# Preserve the reduced dimension.
median_keepdims = ops.weighted_quantile(
    values,
    0.5,
    axis=0,
    keepdims=True,
)
check(median_keepdims, [[2.0, 6.0]], (1, 2))

# Different quantile level for each column.
levels = ops.convert_to_tensor(
    [0.25, 0.75],
    dtype="float32",
)
per_column = ops.weighted_quantile(
    values,
    levels,
    axis=0,
)
check(per_column, [1.0, 7.0], (2,))

# Genuine non-uniform weighting.
weights = ops.convert_to_tensor(
    [0.1, 0.2, 0.6, 0.1],
    dtype="float32",
)
weighted = ops.weighted_quantile(
    values,
    0.5,
    weights=weights,
    axis=0,
)
check(weighted, [4.0, 9.0], (2,))

# Flatten the input if axis is omitted.
flat_median = ops.weighted_quantile(
    values,
    0.5,
    axis=None,
)
check(flat_median, 4.0, ())
""".replace(
        "BACKEND", repr(backend)
    )

    env = os.environ.copy()
    env["KERAS_BACKEND"] = backend

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, (
        f"Backend {backend} failed.\n"
        f"STDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}"
    )

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
"""
This define gestion of interaction with keras (and its backends) for the whole library.
It is imported by all other modules, so that they can use keras backend and random utilities without reimporting keras themselves.
"""

import os
import warnings
from deel.puncc.config import _freeze_backend_flag, set_backend

# Requires keras >= 3.3 for numpy backend support
_MIN_KERAS = (3, 3, 0)

def _parse_version(v: str) -> tuple[int, int, int]:
    parts = v.split(".")
    try:
        return tuple(int(x) for x in parts[:3])
    except Exception:
        return (0, 0, 0)

_defaulted_to_numpy = False
if not os.environ.get("KERAS_BACKEND"):
    set_backend("numpy")
    _defaulted_to_numpy = True

# Freeze backend choice
_freeze_backend_flag()

# Keras import for the whole module
import keras

_kver = getattr(keras, "__version__", "0.0.0")
if _parse_version(_kver) < _MIN_KERAS:
    raise RuntimeError(
        f"Keras {_kver} detected. This library requires Keras >= {'.'.join(map(str, _MIN_KERAS))} "
        "to use the 'numpy' backend. Upgrade with: pip install -U keras"
    )

# Prefer float32 where supported. Ignore if not available.
try:
    keras.backend.set_floatx("float32")
except Exception:
    pass

# Warn once if we silently defaulted to numpy
if _defaulted_to_numpy and keras.backend.backend() == "numpy":
    warnings.warn(
        "Puncc: no backend configured; defaulting to 'numpy'. "
        "Set explicitly with: deel.puncc.config.set_backend('torch'|'jax'|'tensorflow'|'numpy')",
        RuntimeWarning,
        stacklevel=2,
    )

# Public handles reused everywhere else
# ops = keras.ops
backend = keras.backend
random = keras.random
BACKEND_NAME = keras.backend.backend()  # "numpy" | "torch" | "jax" | "tensorflow"

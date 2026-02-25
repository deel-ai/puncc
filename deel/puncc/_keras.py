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
# backend = keras.backend
random = keras.random
BACKEND_NAME = keras.backend.backend()  # "numpy" | "torch" | "jax" | "tensorflow"


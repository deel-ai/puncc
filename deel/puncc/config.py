import os
import sys

_VALID = {"torch", "tensorflow", "jax", "numpy"}
_BACKEND = None

def set_backend(name: str) -> None:
    global _BACKEND

    name = (name or "").strip().lower()

    if name not in _VALID:
        raise ValueError(f"Invalid backend: {name!r}. Choose one of: {_VALID}.")

    if _BACKEND and name!=_BACKEND:
        raise RuntimeError(
            "Backend already initialized. Call set_backend() before importing submodules that use Puncc."
        )
    
    if "keras" in sys.modules:
        keras = sys.modules["keras"]
        backend = keras.backend.backend()
        if backend != name:
            raise RuntimeError(
                f"Cannot set PUNCC backend to {name!r}: "
                f"Keras is already initialized with backend {backend!r}. "
                "Configure the PUNCC backend before importing Keras."
            )

    os.environ["KERAS_BACKEND"] = name
    _BACKEND = name
    
def get_backend() -> str|None:
    """
    Returns:
        str|None: keras backend env var
    """
    val = _BACKEND or os.environ.get("KERAS_BACKEND")
    return val.strip().lower() if val else None

def is_backend_frozen() -> bool:
    """
    Return whether the PUNCC backend has been selected.
    """
    return _BACKEND is not None

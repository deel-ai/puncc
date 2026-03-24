import os

_VALID = {"torch", "tensorflow", "jax", "numpy"}
_FROZEN = False

def set_backend(name: str) -> None:
    if _FROZEN and name!=get_backend():
        raise RuntimeError(
            "Backend already initialized. Call set_backend() before importing submodules that use Puncc."
        )
    name = (name or "").strip().lower()
    if name not in _VALID:
        raise ValueError(f"Invalid backend: {name!r}. Choose one of: {_VALID}.")
    os.environ["KERAS_BACKEND"] = name

def get_backend() -> str|None:
    val = os.environ.get("KERAS_BACKEND")
    return val.strip().lower() if val else None

def _freeze_backend_flag() -> None:
    global _FROZEN
    _FROZEN = True
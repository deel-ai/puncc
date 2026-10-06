import os
import logging

_BACKEND_ENV_VAR = "KERAS_BACKEND"

logger = logging.getLogger(__name__)


def _backend_from_env() -> str | None:
    value = os.environ.get(_BACKEND_ENV_VAR)
    if value is None:
        return None

    value = value.strip().lower()
    if value not in _VALID:
        return None
    return value

_VALID = {"torch", "tensorflow", "jax", "numpy"}
_backend:str|None = _backend_from_env()
_backend_locked:bool = False
_backend_explicitly_set:bool = False



def lock_backend() -> None:
    """
    Lock the PUNCC backend to prevent further changes.
    """
    global _backend_locked
    _backend_locked = True

def set_backend(name: str) -> None:
    global _backend, _backend_explicitly_set
    name = (name or "").strip().lower()
    if name not in _VALID:
        raise ValueError(f"Invalid backend: {name!r}. Choose one of: {_VALID}.")

    if _backend_locked and name != _backend:
        raise RuntimeError(
            "Backend already locked. Consider calling puncc.config.set_backend() at the top of your python script."
        )

    os.environ[_BACKEND_ENV_VAR] = name
    _backend = name
    _backend_explicitly_set = True
    logger.debug("PUNCC backend configured as %s.", name)

def set_inferred_backend(name: str) -> None:
    global _backend
    name = (name or "").strip().lower()

    if _backend_explicitly_set:
        return
    if name not in _VALID:
        raise ValueError(f"Invalid backend: {name!r}. Choose one of: {_VALID}.")
    if _backend_locked and name != _backend:
        raise RuntimeError(
            "Backend already locked. Consider calling puncc.config.set_backend() at the top of your python script."
        )
    os.environ[_BACKEND_ENV_VAR] = name
    _backend = name
    logger.debug("PUNCC backend inferred as %s.", name)


def get_backend() -> str|None:
    """
    Returns:
        str|None: the backend currently selected by PUNCC.
    """
    return _backend
    #val = _backend or os.environ.get(_BACKEND_ENV_VAR)
    #return val.strip().lower() if val else None

def is_backend_locked() -> bool:
    """
    Return whether the PUNCC backend has been selected.
    """
    return _backend_locked

def is_backend_explicitly_set() -> bool:
    """
    Return whether the PUNCC backend has been explicitly set.
    """
    return _backend_explicitly_set
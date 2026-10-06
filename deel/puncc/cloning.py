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
    This module provides basic cloning utilities for simple models (used in cross-conformal methods)
    The main function is `clone_model`, which tries to clone a model using various strategies, including specific cloners for popular ML libraries and a fallback to `copy.deepcopy`.
    This is approximative and may not work for all models, especially complex ones with non-standard architectures or custom layers.
    In such cases, users are encouraged to implement their own cloning logic or expose a `clone` method in their model classes.
    The whole module should be improved (or replaced) in the future.
"""
from __future__ import annotations

import copy
import inspect
import warnings
import logging
from typing import Any

from deel.puncc.warnings import ModelCloningWarning
from deel.puncc.config import get_backend
import sys

logger = logging.getLogger(__name__)

# TODO : improve this whole module (preferably before it achieves self-awareness) !
# TODO : add better torch model cloning
# TODO : add warnings where weights can't or must be cloned !
# TODO : praise the mighty GPU spirits for not segfaulting during cloning
# TODO : figure out how to clone advanced models without breaking the space–time continuum
# TODO : ensure this code works on Mondays (historically problematic)
# TODO : rewrite this in fewer hacks and more science
# TODO : investigate why deepcopy behaves like a gremlin after midnight
# TODO : implement a universal clone that also makes coffee
# TODO : write unit tests that also judge our life choices
# TODO : maybe train a deep learning model to clone other models ?
# TODO : optimize for quantum computers (just in case)
# TODO : make the module emit confetti on successful clone
# TODO : add few more TODOs

ML_MODULES = {"keras", "torch", "tensorflow", "sklearn", "transformers", "jax"}

def get_imported_modules() -> set[str]:
    """
    Returns:
        set[str]: Set of top-level modules that have been imported in the current Python session.
    """
    return set(module.split(".")[0] for module in list(sys.modules.keys()) if module)

def get_imported_ml_modules() -> set[str]:
    """
    Returns:
        set[str]: Set of top-level ML modules that have been imported in the current Python session, filtered from a predefined list of ML libraries that are supported in Keras3.3 context.
    """
    imported = get_imported_modules()
    return ML_MODULES.intersection(imported)

def get_origin_from_model(obj)->str|None:
    cls = obj.__class__
    module = getattr(cls, "__module__", "") or ""
    mro = getattr(cls, "__mro__", ()) or ()

    def mro_has(prefix: str):
        for c in mro:
            mod = getattr(c, "__module__", "") or ""
            if mod.startswith(prefix):
                return True
        return False

    for orig in ML_MODULES:
        if module.startswith(orig) or mro_has(orig):
            return orig
    return None


class ModelCloningError(RuntimeError):
    poem = """
        Oh weary dev, take heart, take rest,
        This model resists your cloning quest.
        Its weights entwined, a secret kept,
        No copy here, no matter how you prep.

        Perhaps one day the stars align,
        And tensors dance in perfect line.
        But today, dear coder, you must concede,
        Some models are wild, they cannot be freed.
        """
    
    def __init__(
        self,
        model,
        strategy: str | None = None,
    ):
        model_type = (
            f"{type(model).__module__}."
            f"{type(model).__qualname__}"
        )

        msg = f"Could not clone model of type {model_type}."

        if strategy is not None:
            msg += f" Cloning strategy: {strategy}."

        msg += (
            " Consider providing a custom cloning implementation "
            "for this model."
        )

        super().__init__(msg)


def clone_model(
    model: Any,
    *,
    clone_weights:bool=False
) -> Any:
    """
        Clone a model across popular ML frameworks.

        Strategy:
        1) If the object exposes `.clone()`, use it.
        2) Try a cloner matching the configured backend.
        3) Try a cloner matching the model's inferred origin.
        4) Try remaining cloners.
        5) Fallback to deepcopy (restricted for ML frameworks).
    """
    # Check if model has a "clone" or a "copy" method:
    clone_method = getattr(model, "clone", None)

    if callable(clone_method):
        try:
            signature = inspect.signature(clone_method)
        except (TypeError, ValueError):
            signature = None

        if signature is not None and "clone_weights" in signature.parameters:
            try:
                clone = clone_method(clone_weights=clone_weights)
                if clone is model:
                    raise ModelCloningError(model, strategy="model.clone() returned the original object")
                return clone
            except Exception as e:
                raise ModelCloningError(model, strategy="model.clone()") from e
        if clone_weights:
            clone = clone_method()
            if clone is model:
                raise ModelCloningError(model, strategy="model.clone() returned the original object")
            return clone
        warnings.warn(
            (
                f"{type(model).__name__}.clone() does not expose a "
                "'clone_weights' argument. PUNCC cannot verify whether "
                "learned state is preserved. Trying framework-specific "
                "cloning strategies instead."
            ),
            ModelCloningWarning,
            stacklevel=2,
        )

    available_cloners = {
        "sklearn": _clone_sklearn,
        "torch": _clone_torch,
        "keras": _clone_keras,
        "transformers": _clone_hf,
        "tensorflow": _clone_keras,
        "jax": _clone_jax
    }

    # Try cloner associated to the actually used backend
    backend_guess = get_backend()
    if backend_guess == "numpy":
        backend_guess =  None #"sklearn"

    origin_guess = get_origin_from_model(model)
    logger.debug(
        "Cloning model type=%s.%s, backend=%s, origin=%s, "
        "clone_weights=%s.",
        type(model).__module__,
        type(model).__qualname__,
        backend_guess,
        origin_guess,
        clone_weights,
    )
    # most probable cloning strategies
    order = []

    if origin_guess is not None :
        order = [origin_guess]

    if backend_guess is not None and origin_guess != backend_guess:
        order += [backend_guess]

    # Possible cloning strategies
    order += [k for k in get_imported_ml_modules() if k not in order]

    # # add remaining cloners at the end of the list
    # order += [k for k in available_cloners.keys() if k not in order]

    # try cloners
    for guess in order:
        cloner = available_cloners.get(guess)
        if cloner is None:
            continue
        try:
            cloned = cloner(model, clone_weights=clone_weights)
        except ModelCloningError as e:
            logger.debug(
                "Cloning strategy %s failed with ModelCloningError: %s",
                guess,
                str(e),
            )
            continue
        if cloned is not None:
            logger.debug("Model cloned successfully using strategy=%s.", guess)
            if cloned is model:
                raise ModelCloningError(model, strategy=f"{guess} cloner returned the original object")
            return cloned

    # if model is from a known ML library but no cloner worked, raise an error instead of silently falling back to deepcopy
    if origin_guess in {"torch", "tensorflow", "keras", "transformers", "jax"}:
        raise ModelCloningError(model, strategy=None)

    try:
        # Fallback to deepcopy if no specific cloner worked
        if not clone_weights:
            warnings.warn(
                (
                    f"No dedicated cloning strategy was found for model type {type(model).__module__}.{type(model).__qualname__}. "
                    "Falling back to deepcopy. PUNCC cannot guarantee that learned model state has been reset."
                    "Please expose a `clone()` method or provide a custom cloning implementation for this model."
                ),
                ModelCloningWarning,
                stacklevel=2,
            )
        logger.debug(
            "Falling back to deepcopy for model type=%s.%s.",
            type(model).__module__,
            type(model).__qualname__,
        )
        clone = copy.deepcopy(model)
        if clone is model:
            raise ModelCloningError(model, strategy="deepcopy returned the original object")
        return clone
    except Exception as e:
        # If even deepcopy fails, raise a custom error
        raise ModelCloningError(model, strategy="deepcopy") from e

def _clone_sklearn(model: Any, *, clone_weights:bool=False) -> Any | None:
    try:
        import sklearn.base
    except ImportError:
        return None
    
    if not isinstance(model, getattr(sklearn.base, "BaseEstimator", ())):
        return None
    try:
        if clone_weights:
            return copy.deepcopy(model)
        return sklearn.base.clone(model)
    except Exception as e:
        raise ModelCloningError(
            model,
            strategy="sklearn",
        ) from e

def _clone_keras(model: Any, *, clone_weights:bool=False) -> Any | None:
    """
    Clone Keras models with optional recompilation that mirrors optimizer/loss/metrics.
    """
    if "keras" not in sys.modules:
        return None
    
    keras = sys.modules["keras"]

    if not isinstance(model, keras.Model):
        return None
    try:
        cloned = keras.models.clone_model(model)
        if clone_weights:
            cloned.set_weights(model.get_weights())
        ### Compilation should be done in the "fit_function" (compile + fit given to the conformalize)
        ### Complexe and custom compilations cannot be handled here
        # if getattr(model, "compiled", False):
        #     compile_config = model.get_compile_config()
        #     cloned.compile_from_config(compile_config)
        return cloned

    except Exception as e:
        raise ModelCloningError(
            model,
            strategy="keras",
        ) from e

def _torch_device(model):
    import torch
    try:
        for p in model.parameters(recurse=True):
            return p.device
        for b in model.buffers(recurse=True):
            return b.device
    except Exception:
        pass
    return torch.device("cpu")

def _reinit_torch_module_(m):
    # Best-effort: reinitialize common modules
    reset = getattr(m, "reset_parameters", None)
    if callable(reset):
        reset()
        return True
    
    reset_stats = getattr(m, "reset_running_stats", None)
    if callable(reset_stats):
        reset_stats()
        return True

    return False

    # Handle common buffer-like state (BatchNorm running stats)
    # Most BN layers handle it in reset_parameters, but not all custom ones.
    if hasattr(m, "running_mean") and m.running_mean is not None:
        m.running_mean.zero_()
    if hasattr(m, "running_var") and m.running_var is not None:
        m.running_var.fill_(1)
    if hasattr(m, "num_batches_tracked") and m.num_batches_tracked is not None:
        m.num_batches_tracked.zero_()


def _clone_torch(model, *, clone_weights: bool = False):
    try:
        import torch
    except ImportError:
        return None

    if not isinstance(model, getattr(torch.nn, "Module", ())):
        return None

    try:
        with torch.no_grad():
            cloned = copy.deepcopy(model)
            cloned.train(model.training)

            if not clone_weights:
                # Best-effort reinit
                missing = []
                for m in cloned.modules():
                    if _reinit_torch_module_(m):
                        continue
                    has_state = (any(True for _ in m.parameters(recurse=False))) or any(True for _ in m.buffers(recurse=False))
                    if has_state:
                        missing.append(type(m).__name__)

                if missing:
                    warnings.warn(
                        (
                            "Torch model was cloned and reinitialized on a best-effort "
                            "basis, but some parameterized modules do not expose "
                            f"reset_parameters(): {sorted(set(missing))}. "
                            "Some learned weights may therefore have been preserved."
                        ),
                        ModelCloningWarning,
                        stacklevel=2,
                    )

        return cloned
    except Exception as e:
        raise ModelCloningError(
            model,
            strategy="torch",
        ) from e

def _clone_hf(model: Any, *, clone_weights:bool=False) -> Any | None:
    try:
        import transformers
    except ImportError:
        return None
    
    try:
        # PyTorch HF
        if isinstance(model, getattr(transformers, "PreTrainedModel", ())):
            new_m = model.__class__(copy.deepcopy(model.config))
            new_m = new_m.to(_torch_device(model))
            if clone_weights:
                try:
                    import torch
                    with torch.no_grad():
                        new_m.load_state_dict(model.state_dict())
                except ImportError:
                    new_m.load_state_dict(model.state_dict())
            new_m.train(model.training)
            return new_m

        # TensorFlow HF
        if isinstance(model, getattr(transformers, "TFPreTrainedModel", ())):
            new_m = model.__class__(copy.deepcopy(model.config))
            if clone_weights:
                new_m.set_weights(model.get_weights())
            return new_m

        # Flax HF
        if isinstance(model, getattr(transformers, "FlaxPreTrainedModel", ())):
            dtype = getattr(model, "dtype", None)
            new_m = model.__class__(copy.deepcopy(model.config), dtype=dtype)
            if clone_weights:
                new_m.params = copy.deepcopy(model.params)
            return new_m
        return None
    except Exception as e:
        raise ModelCloningError(
            model,
            strategy="transformers",
        ) from e
        
def _clone_jax(model: Any, *, clone_weights:bool = False) -> Any | None:
    """Generic JAX cloning is not supported yet."""
    return None
    #raise NotImplementedError("JAX model cloning is not yet implemented, please expose a 'clone' method or use non cross conformal methods.")

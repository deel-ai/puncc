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
import warnings
from typing import Any

from deel.puncc.config import get_backend
import sys

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

ML_MODULES = {"torch", "tensorflow", "keras", "sklearn", "transformers", "jax"}

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
    name = getattr(cls, "__name__", "") or ""
    mro = getattr(cls, "__mro__", ()) or ()

    def mro_has(prefix: str):
        for c in mro:
            mod = getattr(c, "__module__", "") or ""
            if mod.startswith(prefix):
                return True
        return False

    for orig in ML_MODULES:
        if module.startswith(orig) or name.startswith(orig) or mro_has(orig):
            return orig
    return None


class ModelCannotBeClonedError(RuntimeError):
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
    def __init__(self):
        super().__init__(self.poem)


def clone_model(
    model: Any,
    *,
    clone_weights:bool=False
) -> Any:
    """
        Clone a model across popular ML frameworks.

        Strategy:
        1) If the object exposes `.clone()` or `.copy()`, use it.
        2) Try a cloner matching the configured backend.
        3) Try a cloner matching the model's inferred origin.
        4) Try remaining cloners.
        5) Fallback to deepcopy (restricted for ML frameworks).
    """
    # Check if model has a "clone" or a "copy" method:
    if hasattr(model, "clone") and callable(getattr(model, "clone")):
        return model.clone()
    if hasattr(model, "copy") and callable(getattr(model, "copy")):
        return model.copy()

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
        backend_guess = "sklearn"

    origin_guess = get_origin_from_model(model)

    # most probable cloning strategies
    order = []
    if backend_guess is not None :
        order = [backend_guess]

    if origin_guess is not None and origin_guess != backend_guess:
        order += [origin_guess]

    # Possible cloning strategies
    order += [k for k in get_imported_ml_modules() if k not in order]

    # add remaining cloners at the end of the list
    order += [k for k in available_cloners.keys() if k not in order]

    # try cloners
    for guess in order:
        cloner = available_cloners.get(guess)
        if cloner is None:
            continue
        cloned = cloner(model, clone_weights=clone_weights)
        if cloned is not None:
            return cloned

    # if model is from a known ML library but no cloner worked, raise an error instead of silently falling back to deepcopy
    if origin_guess in {"torch", "tensorflow", "keras", "transformers", "jax"}:
        raise ModelCannotBeClonedError()

    try:
        # Fallback to deepcopy if no specific cloner worked
        return copy.deepcopy(model)
    except Exception as e:
        # If even deepcopy fails, raise a custom error
        raise ModelCannotBeClonedError() from e

def _clone_sklearn(model: Any, *, clone_weights:bool=False) -> Any | None:
    try:
        import sklearn.base
    except ImportError:
        return None
    if isinstance(model, getattr(sklearn.base, "BaseEstimator", ())):
        if clone_weights:
            return copy.deepcopy(model)
        return sklearn.base.clone(model)
    return None

def _clone_keras(model: Any, *, clone_weights:bool=False) -> Any | None:
    """
    Clone Keras models with optional recompilation that mirrors optimizer/loss/metrics.
    """
    try:
        import keras
    except ImportError:
        return None

    if isinstance(model, getattr(keras, "Model", ())):
        cloned = keras.models.clone_model(model)
        if clone_weights:
            cloned.set_weights(model.get_weights())
        # TODO : eventually add compilation (optimizer/loss/metrics) of the freshly cloned model
        return cloned
    return None

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

    with torch.no_grad():
        cloned = copy.deepcopy(model)
        cloned = cloned.to(_torch_device(model))
        cloned.train(model.training)

        if not clone_weights:
            # Best-effort reinit
            missing = []
            for m in cloned.modules():
                if callable(getattr(m, "reset_parameters", None)):
                    _reinit_torch_module_(m)
                else:
                    # Not every submodule needs reset_parameters, but if a leaf has params
                    # and no reset_parameters, we can't safely reinit it.
                    has_params = any(p is not None for p in m.parameters(recurse=False))
                    if has_params:
                        missing.append(type(m).__name__)

            if missing:
                warnings.warn(
                    "Torch model cloned then best-effort reinitialized, but some "
                    f"parameterized modules lack reset_parameters(): {sorted(set(missing))}. "
                    "Weights for these modules may still be copied. For a correct clone "
                    "without weights, implement `.clone()`/factory reconstruction.",
                    RuntimeWarning,
                )

    return cloned

def _clone_hf(model: Any, *, clone_weights:bool=False) -> Any | None:
    try:
        import transformers
    except ImportError:
        return None

    # PyTorch HF
    if isinstance(model, getattr(transformers, "PreTrainedModel", ())):
        new_m = model.__class__(model.config)
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
        new_m = model.__class__(model.config)
        if clone_weights:
            new_m.set_weights(model.get_weights())
        return new_m

    # Flax HF
    if isinstance(model, getattr(transformers, "FlaxPreTrainedModel", ())):
        dtype = getattr(model, "dtype", None)
        new_m = model.__class__(model.config, dtype=dtype)
        if clone_weights:
            new_m.params = copy.deepcopy(model.params)
        return new_m
    return None

def _clone_jax(model: Any, *, clone_weights:bool = False) -> Any | None:
    return None
    #raise NotImplementedError("JAX model cloning is not yet implemented, please expose a 'clone' method or use non cross conformal methods.")

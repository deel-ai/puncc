from typing import TYPE_CHECKING, Any, Union, TypeAlias, Protocol, runtime_checkable, Callable
from collections.abc import Iterable, Sequence
from deel.puncc.cloning import clone_model

# if TYPE_CHECKING:
#     import numpy as _np
#     import torch as _torch
#     import tensorflow as _tf
#     from jax import Array as _JaxArray
#     TensorLike: TypeAlias = Union[_np.ndarray, _torch.Tensor, _tf.Tensor, _JaxArray]
# else:
#     TensorLike = Any

TensorLike:TypeAlias = Any

@runtime_checkable
class Predictor(Protocol):
    def __call__(self, X: Iterable[Any], *args, **kwargs) -> TensorLike:
        ...

@runtime_checkable
class LambdaPredictor(Predictor):
    def __call__(self, X:Iterable[Any], lambd:float, *args, **kwargs) -> Any:
        ...

@runtime_checkable
class Fitable(Protocol):
    def fit(self, X: Iterable[Any], y: TensorLike, *args, **kwargs) -> Any:
        ...

@runtime_checkable
class PredictorLike(Protocol):
    def predict(self, X: Iterable[Any], *args, **kwargs) -> TensorLike:
        ...
    
class _PredictorAdapter:
    """Wraps a .predict(...) provider into a callable."""
    def __init__(self, model: PredictorLike) -> None:
        self._model = model

    def __call__(self, X: Iterable[Any], *args: Any, **kwargs: Any) -> Any:
        return self._model.predict(X, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)
    
    def __setattr__(self, name, value):
        if name == "_model":
            super().__setattr__(name, value)
        else:
            setattr(self._model, name, value)

    def clone(self):
        return _PredictorAdapter(clone_model(self._model))

def make_predictor(model: Union[Predictor, PredictorLike]) -> Predictor:
    if callable(model):
        return model
    if hasattr(model, "predict") and callable(model.predict):
        predictor = _PredictorAdapter(model)
        return predictor
    raise TypeError("The provided model neither have __call__ nor predict method.")

NCScoreFunction:TypeAlias = Callable[[TensorLike, TensorLike], Sequence[float]]
PredSetFunction:TypeAlias = Callable[[TensorLike, float|TensorLike], Sequence[Any]]

from typing import Any

from deel.puncc.backend.keras import ops
from deel.puncc.typing import TensorLike


def check_same_shape(
    **tensors: Any,
) -> None:
    """Require tensors to have exactly the same shape."""
    if not tensors:
        return
    tensor_iter = iter(tensors.items())
    ref_name, ref_tensor = next(tensor_iter)
    ref_shape = tuple(ops.shape(ref_tensor))

    for name, tensor in tensor_iter:
        shape = tuple(ops.shape(tensor))
        if shape != ref_shape:
            raise ValueError(
                f"All tensors must have the same shape, "
                f"but {ref_name} has shape {ref_shape} and {name} has shape {shape}."
            )


def as_scalar_vector(
    value: TensorLike,
    *,
    n_samples: int | None = None,
    allow_scalar: bool = False,
) -> TensorLike:
    """
    Normalize scalar regression values to a one-dimensional tensor.

    Accept shapes (n,) and (n, 1), or a scalar when explicitly allowed.
    Reject multi-output values.
    """
    value = ops.convert_to_tensor(value)
    shape = tuple(ops.shape(value))

    if allow_scalar and shape == ():
        value = ops.reshape(value, (1,))
    elif len(shape) == 2 and shape[1] == 1:
        value = ops.squeeze(value, axis=-1)
    elif len(shape) != 1:
        raise ValueError(f"tensor must have shape (n,) or (n, 1), " f"got {shape}.")

    if n_samples is not None and len(value) != n_samples:
        raise ValueError(
            f"tensor must contain {n_samples} samples, " f"got {len(value)}."
        )

    return value

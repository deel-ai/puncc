"""Model-copy regression tests migrated from main."""

import numpy as np
import pytest

from deel.puncc.cloning import clone_model


def test_clone_model_sklearn_preserves_fitted_state():
    linear_model = pytest.importorskip("sklearn.linear_model")

    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])
    model = linear_model.LinearRegression().fit(X, y)

    copied = clone_model(model, clone_weights=True)

    assert copied is not model
    np.testing.assert_allclose(copied.predict(X), model.predict(X))

    model.coef_[0] += 1.0
    assert not np.allclose(copied.coef_, model.coef_)


def test_clone_model_torch_preserves_parameters():
    torch = pytest.importorskip("torch")

    model = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=True))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 2.0]]))
        model[0].bias.copy_(torch.tensor([0.5]))

    copied = clone_model(model, clone_weights=True)

    assert copied is not model
    for original_param, copied_param in zip(model.parameters(), copied.parameters()):
        assert torch.allclose(original_param, copied_param)

    with torch.no_grad():
        model[0].weight.add_(1.0)

    assert not torch.allclose(model[0].weight, copied[0].weight)


def test_clone_model_keras_unbuilt_model():
    import keras

    model = keras.Sequential()
    model.add(keras.layers.Dense(1))

    copied = clone_model(model, clone_weights=True)

    assert copied is not model
    assert copied.built is False


def test_clone_model_keras_preserves_weights():
    import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(2,)),
            keras.layers.Dense(1, use_bias=True),
        ]
    )
    model.set_weights(
        [
            np.array([[1.0], [2.0]], dtype=np.float32),
            np.array([0.5], dtype=np.float32),
        ]
    )

    copied = clone_model(model, clone_weights=True)

    assert copied is not model
    for original_weights, copied_weights in zip(
        model.get_weights(), copied.get_weights()
    ):
        np.testing.assert_allclose(copied_weights, original_weights)

    model.set_weights(
        [
            np.array([[3.0], [4.0]], dtype=np.float32),
            np.array([1.5], dtype=np.float32),
        ]
    )

    assert not np.allclose(model.get_weights()[0], copied.get_weights()[0])


def test_clone_model_jax_preserves_pytree_values():
    jnp = pytest.importorskip("jax.numpy")

    model = {
        "weights": jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "bias": jnp.asarray([0.5, 1.5]),
    }

    copied = clone_model(model, clone_weights=True)

    assert copied is not model
    np.testing.assert_allclose(
        np.asarray(copied["weights"]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    np.testing.assert_allclose(
        np.asarray(copied["bias"]),
        np.array([0.5, 1.5]),
    )

    model["weights"] = model["weights"] + 1.0

    np.testing.assert_allclose(
        np.asarray(copied["weights"]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )

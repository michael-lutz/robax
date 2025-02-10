"""Primarily testing optimizer learning rates"""

import jax.numpy as jnp

from robax.training.train import create_optimizer


def test_different_learning_rates() -> None:
    base_learning_rate = 0.001
    overrides = {"param2": 0.0001}
    optimizer = create_optimizer(base_learning_rate, overrides)

    params = {"param1": jnp.array([1.0, 2.0]), "param2": jnp.array([1.0, 2.0])}
    opt_state = optimizer.init(params)

    grads = {"param1": jnp.array([0.1, 0.1]), "param2": jnp.array([0.1, 0.1])}
    updates, _ = optimizer.update(grads, opt_state, params)

    assert jnp.allclose(updates["param1"], updates["param2"] * 10)


def test_same_learning_rates() -> None:
    base_learning_rate = 0.001
    overrides = {}
    optimizer = create_optimizer(base_learning_rate, overrides)

    params = {"param1": jnp.array([1.0, 2.0]), "param2": jnp.array([1.0, 2.0])}
    opt_state = optimizer.init(params)

    grads = {"param1": jnp.array([0.1, 0.1]), "param2": jnp.array([0.1, 0.1])}
    updates, _ = optimizer.update(grads, opt_state, params)

    assert jnp.allclose(updates["param1"], updates["param2"])


def test_nested_params() -> None:
    base_learning_rate = 0.001
    overrides = {"param1": 0.0001}
    optimizer = create_optimizer(base_learning_rate, overrides)

    params = {
        "param1": {"param1_1": jnp.array([1.0, 2.0]), "param1_2": jnp.array([1.0, 2.0])},
        "param2": jnp.array([1.0, 2.0]),
    }
    opt_state = optimizer.init(params)

    grads = {
        "param1": {"param1_1": jnp.array([0.1, 0.1]), "param1_2": jnp.array([0.1, 0.1])},
        "param2": jnp.array([0.1, 0.1]),
    }
    updates, _ = optimizer.update(grads, opt_state, params)

    assert jnp.allclose(updates["param1"]["param1_1"], updates["param1"]["param1_2"])
    assert jnp.allclose(updates["param1"]["param1_2"] * 10, updates["param2"])

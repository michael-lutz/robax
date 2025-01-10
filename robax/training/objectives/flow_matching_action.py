"""Implements flow matching action loss and sampling utilities needed for training"""

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn


def sample_starting_noise(prng: jax.Array, action_shape: Tuple[int, ...]) -> jax.Array:
    """Samples starting noise from the target action

    Args:
        prng: The PRNG key
        action_shape: The shape of the action [B, A, a]
    """
    assert len(action_shape) == 3, "action_shape must be [B, A, a]"
    return jax.random.normal(prng, action_shape)


def sample_tau(
    prng: jax.Array,
    timestep_shape: Tuple[int, ...],
    cutoff_value: float = 0.9,
    beta_a: float = 1.5,
    beta_b: float = 1.0,
) -> jax.Array:
    """Samples tau from the target action according to the paper's beta distribution

    Args:
        prng: The PRNG key
        timestep_shape: The shape of the timesteps [B]
        cutoff_value: The cutoff value for the beta distribution
        beta_a: The alpha parameter for the beta distribution
        beta_b: The beta parameter for the beta distribution
    """
    assert len(timestep_shape) == 1, "timestep_shape must be [B]"
    x = jax.random.beta(prng, a=beta_a, b=beta_b, shape=timestep_shape)
    return (1 - x) * cutoff_value


def optimal_transport_vector_field(
    starting_noise: jax.Array, target_action: jax.Array
) -> jax.Array:
    """Computes the ground truth vector field for flow matching

    Args:
        starting_noise: The 0-th action, i.e. pure noise
        target_action: The target action vector field
    """
    return target_action - starting_noise


def flow_matching_action_loss(
    predicted_field: jax.Array,
    target_action: jax.Array,
    starting_noise: jax.Array,
) -> jax.Array:
    """Computes the action loss for flow matching

    NOTE: we assume that the ground truth field is tau-independent.

    Args:
        predicted_field: The predicted action chunks [B, A, a]
        target_action: The target action chunk [B, A, a]
        starting_noise: The 0-th action, i.e. pure noise [B, A, a]
    """
    gt_vector_field = optimal_transport_vector_field(starting_noise, target_action)
    mse_loss = jnp.mean(jnp.square(predicted_field - gt_vector_field))
    return mse_loss


def train_step(
    prng_key: jax.Array,
    params: Dict[str, Any],
    opt_state: optax.OptState,
    images: jax.Array,
    text: jax.Array,
    proprio: jax.Array,
    action: jax.Array,
    model: nn.Module,
    optimizer: optax.GradientTransformation,
) -> Tuple[jax.Array, Dict[str, Any], optax.OptState, jax.Array]:
    assert action.shape[0] == images.shape[0] == text.shape[0] == proprio.shape[0]
    timesteps = sample_tau(prng_key, (action.shape[0],))
    prng_key, subkey = jax.random.split(prng_key)
    starting_noise = sample_starting_noise(prng_key, action.shape)
    prng_key, subkey = jax.random.split(prng_key)

    def loss_fn(params):
        predicted_field, _ = model.apply(
            params,
            images=images,
            text=text,
            proprio=proprio,
            action=action,
            timesteps=timesteps,
        )
        loss = flow_matching_action_loss(predicted_field, action, starting_noise)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return prng_key, params, opt_state, loss

"""Implements flow matching action loss and sampling utilities needed for training"""

from typing import Any, Dict, Tuple

import attrs
import flax.linen as nn
import jax
import jax.numpy as jnp

from robax.training.objectives.base_train_step import BaseTrainStep
from robax.utils.observation import Observation, get_batch_size


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


def interpolate_noise_and_target(
    starting_noise: jax.Array, target_action: jax.Array, timesteps: jax.Array
) -> jax.Array:
    """Interpolates the noise and target action at the given timesteps

    Args:
        starting_noise: The 0-th action, i.e. pure noise [B, A, a]
        target_action: The target action vector field [B, A, a]
        timesteps: The timesteps to interpolate at [B]

    Returns:
        The interpolated action vector field [B, A, a]
    """
    # Reshape timesteps to [B, 1, 1] to enable broadcasting
    timesteps = timesteps[:, None, None]
    return starting_noise + (target_action - starting_noise) * timesteps


@attrs.define(frozen=True)
class FlowMatchingActionTrainStep(BaseTrainStep):
    """Flow matching action train step."""

    cutoff_value: float = attrs.field(default=0.999)
    """Cutoff value for the beta distribution."""
    beta_a: float = attrs.field(default=1.5)
    """Alpha parameter for the beta distribution."""
    beta_b: float = attrs.field(default=1.0)
    """Beta parameter for the beta distribution."""

    def get_loss(
        self,
        params: Dict[str, Any],
        model: nn.Module,
        observation: Observation,
        target: jax.Array,
        **additional_inputs: jax.Array,
    ) -> jax.Array:
        """Computes the action loss for flow matching"""
        starting_noise: jax.Array = additional_inputs["starting_noise"]
        timesteps: jax.Array = additional_inputs["timesteps"]
        interpolated_action = interpolate_noise_and_target(starting_noise, target, timesteps)

        predicted_field, _ = model.apply(
            params,
            observation=observation,
            noisy_action=interpolated_action,
            timesteps=timesteps,
        )

        gt_vector_field = optimal_transport_vector_field(starting_noise, target)
        mse_loss = jnp.mean(jnp.square(predicted_field - gt_vector_field))
        return mse_loss

    def get_additional_inputs(
        self,
        prng_key: jax.Array,
        observation: Observation,
        unbatched_prediction_shape: Tuple[int, int],
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Generates the additional inputs for the train step."""
        batch_size = get_batch_size(observation)
        timesteps = sample_tau(prng_key, (batch_size,))
        prng_key, _ = jax.random.split(prng_key)
        starting_noise = sample_starting_noise(prng_key, (batch_size, *unbatched_prediction_shape))
        prng_key, _ = jax.random.split(prng_key)
        training_inputs = {
            "timesteps": timesteps,
            "starting_noise": starting_noise,
        }
        return prng_key, training_inputs

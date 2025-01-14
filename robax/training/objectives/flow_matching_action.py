"""Implements flow matching action loss and sampling utilities needed for training"""

from typing import Any, Dict, Tuple

import attrs
import jax
import jax.numpy as jnp

from robax.training.objectives.base_train_step import BaseTrainStep


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
        images: jax.Array,
        text: jax.Array,
        proprio: jax.Array,
        action: jax.Array,
        **additional_inputs: jax.Array
    ) -> jax.Array:
        """Computes the action loss for flow matching"""
        predicted_field, _ = self.model.apply(
            params,
            images=images,
            text=text,
            proprio=proprio,
            action=action,
            timesteps=additional_inputs["timesteps"],
        )

        starting_noise: jax.Array = additional_inputs["starting_noise"]
        gt_vector_field = optimal_transport_vector_field(starting_noise, action)
        mse_loss = jnp.mean(jnp.square(predicted_field - gt_vector_field))
        return mse_loss

    def get_additional_inputs(
        self,
        prng_key: jax.Array,
        images: jax.Array,
        text: jax.Array,
        proprio: jax.Array,
        action: jax.Array,
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Generates the additional inputs for the train step."""
        timesteps = sample_tau(prng_key, (action.shape[0],))
        prng_key, _ = jax.random.split(prng_key)
        starting_noise = sample_starting_noise(prng_key, action.shape)
        prng_key, _ = jax.random.split(prng_key)
        training_inputs = {
            "timesteps": timesteps,
            "starting_noise": starting_noise,
        }
        return prng_key, training_inputs

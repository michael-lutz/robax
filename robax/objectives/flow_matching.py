"""Implements flow matching action loss and sampling utilities needed for training"""

from typing import Any, Dict, Tuple

import attrs
import flax.linen as nn
import jax
import jax.numpy as jnp

from robax.objectives.base_inference_step import BaseInferenceStep
from robax.objectives.base_train_step import BaseTrainStep
from robax.utils.observation import Observation, get_batch_size
from robax.utils.sampling_utils import sample_gaussian_noise, sample_timesteps


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
class FlowMatchingTrainStep(BaseTrainStep):
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
        debug: bool = False,
        **additional_inputs: jax.Array,
    ) -> jax.Array:
        """Computes the flow matching action loss."""
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
        if debug:
            jax.debug.breakpoint()
        return mse_loss

    def get_additional_inputs(
        self,
        prng_key: jax.Array,
        observation: Observation,
        unbatched_prediction_shape: Tuple[int, int],
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Generates the additional inputs for the train step."""
        assert len(unbatched_prediction_shape) == 2, "unbatched_prediction_shape must be [A, a]"
        batch_size = get_batch_size(observation)
        # Using a beta distribution as done in the pi_zero paper
        prng_key, timesteps = sample_timesteps(
            prng_key,
            (batch_size,),
            distribution="complementary_beta",
            cutoff_value=self.cutoff_value,
            beta_a=self.beta_a,
            beta_b=self.beta_b,
        )

        # Standard gaussian noise to start...
        prng_key, starting_noise = sample_gaussian_noise(
            prng_key, (batch_size, *unbatched_prediction_shape)
        )
        training_inputs = {
            "timesteps": timesteps,
            "starting_noise": starting_noise,
        }
        return prng_key, training_inputs


@attrs.define(frozen=True)
class FlowMatchingInferenceStep(BaseInferenceStep):
    """Flow matching inference step."""

    unbatched_prediction_shape: Tuple[int, int] = attrs.field(default=(1, 1))
    """The shape of a single action, e.g. [A, a]."""
    num_steps: int = attrs.field(default=10)
    """The number of steps to take."""

    def generate_action(
        self,
        prng_key: jax.Array,
        params: Dict[str, Any],
        model: nn.Module,
        observation: Observation,
    ) -> Tuple[jax.Array, jax.Array]:
        """Generate an action from the policy.

        NOTE: observation["action"] here is assumed to only include action history. It does not
        include noise, whereas `__call__` does.

        Args:
            prng_key: [B] PRNG key
            observation: Observation
            num_steps: int, number of steps to take
        Returns:
            prng_key
            [B, A, a] action
        """
        batch_size = get_batch_size(observation)
        prng_key, noisy_action = sample_gaussian_noise(
            prng_key,
            (batch_size, self.unbatched_prediction_shape[0], self.unbatched_prediction_shape[1]),
        )
        delta = 1 / self.num_steps

        # basic integration of the action field
        for i in range(self.num_steps):
            tau = jnp.array(i / self.num_steps)
            tau = jnp.tile(tau, (batch_size,))

            action_field_pred, _ = model.apply(
                params,
                observation=observation,
                inference_mode=True,
                deterministic=True,
                timesteps=tau,
                noisy_action=noisy_action,
            )

            noisy_action += delta * action_field_pred

        return prng_key, noisy_action

"""Diffusion Objective"""

from typing import Any, Dict, Tuple

import attrs
import flax.linen as nn
import jax
import jax.numpy as jnp

from robax.objectives.base_inference_step import BaseInferenceStep
from robax.objectives.base_train_step import BaseTrainStep
from robax.utils.observation import Observation, get_batch_size
from robax.utils.sampling_utils import sample_gaussian_noise, sample_timesteps


def generate_betas(
    num_timesteps: int,
    schedule: str,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    dtype: jnp.dtype = jnp.float32,
    **kwargs: Any,
) -> jax.Array:
    """Generates a beta schedule for the diffusion process."""
    if schedule == "linear":
        return jnp.linspace(beta_start, beta_end, num_timesteps, dtype=dtype, endpoint=False)
    else:
        raise ValueError(f"Invalid beta schedule: {schedule}")


@attrs.define(frozen=True)
class DiffusionTrainStep(BaseTrainStep):
    """DDPM-like training objective."""

    num_train_timesteps: int = attrs.field(default=100)
    """The number of timesteps to use for the diffusion process."""
    beta_schedule: str = attrs.field(default="linear")
    """The beta schedule to use for the diffusion process."""
    beta_start: float = attrs.field(default=0.0001)
    """The starting beta value for the diffusion process."""
    beta_end: float = attrs.field(default=0.02)
    """The ending beta value for the diffusion process."""

    def get_loss(
        self,
        params: Dict[str, Any],
        model: nn.Module,
        observation: Observation,
        target: jax.Array,
        debug: bool = False,
        **additional_inputs: jax.Array,
    ) -> jax.Array:
        """Computes the diffusion objective loss.

        Args:
            params: Model parameters.
            model: The model (flax.linen Module).
            observation: The observation data structure.
            target: The uncorrupted target actions (x_0) of shape [B, A, a].
            debug: Whether to enable debugging (e.g., pdb).
            **additional_inputs: Must include
                'timesteps': jax.Array with shape [B].
                'diffusion_noise': jax.Array with shape [B, A, a].
        Returns:
            The scalar MSE loss between predicted noise and true noise.
        """
        timesteps: jax.Array = additional_inputs["timesteps"]  # shape [B], dtype=jnp.int32
        diffusion_noise: jax.Array = additional_inputs["diffusion_noise"]  # shape [B, A, a]

        # Generate beta schedule (including here since possibly learned in the future)
        betas = generate_betas(
            self.num_train_timesteps,
            self.beta_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            dtype=jnp.float32,
        )  # [B,]
        alphas = jnp.cumprod(1 - betas)[:, None, None]  # [B, 1, 1]

        z_t = (
            jnp.sqrt(alphas[timesteps]) * target + jnp.sqrt(1 - alphas[timesteps]) * diffusion_noise
        )
        standardized_timesteps = jnp.array(timesteps, dtype=jnp.float32) / self.num_train_timesteps

        predicted_noise, _ = model.apply(
            params,
            observation=observation,
            noisy_action=z_t,
            timesteps=standardized_timesteps,
        )

        # equally weighting the loss for each timestep as per the DDPM methodology
        mse_loss = jnp.mean(jnp.square(predicted_noise - diffusion_noise))

        if debug:
            jax.debug.breakpoint()

        return mse_loss

    def get_additional_inputs(
        self,
        prng_key: jax.Array,
        observation: Observation,
        unbatched_prediction_shape: Tuple[int, int],
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Generates additional inputs for the diffusion training step.

        Specifically, we sample timesteps and noise to create x_t in the forward diffusion process.

        Args:
            prng_key: A PRNG key for randomness.
            observation: The observation data structure (used for batch size).
            unbatched_prediction_shape: The shape of a single action, e.g. [A, a].

        Returns:
            A tuple of:
              - Updated PRNG key.
              - Dictionary with keys:
                {
                    "timesteps": jax.Array with shape [B],
                    "diffusion_noise": jax.Array with shape [B, A, a]
                }
        """
        batch_size = get_batch_size(observation)  # e.g., B

        # Sample timesteps t ~ Uniform(0,1)
        prng_key, timesteps = sample_timesteps(
            prng_key,
            (batch_size,),
            distribution="discrete_uniform",
            num_timesteps=self.num_train_timesteps,
            dtype=jnp.int32,  # use float32 for now... TODO parametrize
        )

        # Sample Gaussian noise
        prng_key, diffusion_noise = sample_gaussian_noise(
            prng_key, (batch_size, *unbatched_prediction_shape), dtype=jnp.float32
        )

        training_inputs = {
            "timesteps": timesteps,
            "diffusion_noise": diffusion_noise,
        }
        return prng_key, training_inputs


@attrs.define(frozen=True)
class DiffusionInferenceStep(BaseInferenceStep):
    """
    A discrete-time DDPM-like diffusion inference step, sampling backwards from t=num_steps down to 0.
    """

    unbatched_prediction_shape: Tuple[int, int] = attrs.field(default=(1, 1))
    """Shape of a single action, e.g. [A, a]."""
    beta_schedule: str = attrs.field(default="linear")
    """The beta schedule to use for the diffusion process."""
    beta_start: float = attrs.field(default=0.0001)
    """The starting beta value for the diffusion process."""
    beta_end: float = attrs.field(default=0.02)
    """The ending beta value for the diffusion process."""
    num_inference_timesteps: int = attrs.field(default=10)
    """The number of steps to take."""

    def generate_action(
        self,
        prng_key: jax.Array,
        params: Dict[str, Any],
        model: nn.Module,
        observation: Observation,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Generate an action using the reverse-diffusion approach (DDPM style).

        Args:
            prng_key: PRNG key.
            params: Model parameters for `model.apply`.
            model: A diffusion model that predicts noise, e.g. eps_theta(x_t, t).
            observation: Environment observation (used to determine batch size).
            num_steps: Number of discrete diffusion steps (T).
            **kwargs: Additional arguments (not used here but can pass custom schedules).

        Returns:
            (updated_prng_key, final_actions)
            final_actions shape: [B, A, a], the denoised actions at t=0.
        """

        # NOTE: for now, just return the noise
        batch_size = get_batch_size(observation)  # e.g., B

        betas = generate_betas(
            self.num_inference_timesteps,
            self.beta_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            dtype=jnp.float32,
        )  # shape [T]
        alphas = jnp.cumprod(1.0 - betas)  # shape [T]

        prng_key, z_t = sample_gaussian_noise(
            prng_key, (batch_size, *self.unbatched_prediction_shape), dtype=jnp.float32
        )

        @jax.jit
        def generate_prev_mu(z_t: jax.Array, t: int) -> jax.Array:
            standardized_t = jnp.array(t, dtype=jnp.float32) / self.num_inference_timesteps
            total_noise_pred, _ = model.apply(
                params,
                observation=observation,
                noisy_action=z_t,
                timesteps=jnp.tile(standardized_t, (batch_size,)),
                inference_mode=True,
                deterministic=True,
            )

            z_prev_mu = (1 / jnp.sqrt(1 - betas[t])) * (
                z_t - (betas[t] / jnp.sqrt(1 - alphas[t])) * total_noise_pred
            )
            return z_prev_mu  # type: ignore

        for t in range(self.num_inference_timesteps - 1, 1, -1):
            z_prev_mu = generate_prev_mu(z_t, t)

            prng_key, new_noise = sample_gaussian_noise(
                prng_key, (batch_size, *self.unbatched_prediction_shape), dtype=jnp.float32
            )
            z_t = z_prev_mu + jnp.sqrt(betas[t]) * new_noise

        x = generate_prev_mu(z_t, 0)

        return prng_key, x

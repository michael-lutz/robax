"""Diffusion Objective"""

from functools import cache
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
    dtype: jnp.dtype = jnp.float32,
    **kwargs: Any,
) -> jax.Array:
    """Generates a beta schedule for the diffusion process."""
    if schedule == "linear":
        return jnp.linspace(0, 1, num_timesteps, dtype=dtype)
    else:
        raise ValueError(f"Invalid beta schedule: {schedule}")


@attrs.define(frozen=True)
class DiffusionTrainStep(BaseTrainStep):
    """DDPM-like training objective."""

    num_timesteps: int = attrs.field(default=100)
    """The number of timesteps to use for the diffusion process."""
    beta_schedule: str = attrs.field(default="linear")
    """The beta schedule to use for the diffusion process."""
    beta_kwargs: Dict[str, Any] = attrs.field(default={})
    """Additional keyword arguments for the beta schedule generation.
    
    NOTE: this is intentionally unconstrained until I implement cosine, exponential, etc.
    """

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
            self.num_timesteps, self.beta_schedule, **self.beta_kwargs, dtype=jnp.float32
        )  # [B,]
        alphas = jnp.cumprod(1 - betas)[:, None, None]  # [B, 1, 1]

        z_t = (
            jnp.sqrt(alphas[timesteps]) * target + jnp.sqrt(1 - alphas[timesteps]) * diffusion_noise
        )
        standardized_timesteps = jnp.array(timesteps, dtype=jnp.float32) / self.num_timesteps

        predicted_noise, _ = model.apply(
            params,
            observation=observation,
            noisy_action=z_t,
            timesteps=standardized_timesteps,
        )

        # equally weighting the loss for each timestep as per the DDPM methodology
        mse_loss = jnp.mean(jnp.square(predicted_noise - diffusion_noise))

        if debug:
            import pdb

            pdb.set_trace()

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
            num_timesteps=self.num_timesteps,
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
    beta_kwargs: Dict[str, Any] = attrs.field(default={})
    """Additional keyword arguments for the beta schedule generation."""
    num_timesteps: int = attrs.field(default=100)
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
        return prng_key, jax.random.normal(prng_key, (batch_size, *self.unbatched_prediction_shape))

        # ----- 1. Create a beta schedule and derive alpha terms -----
        betas = generate_betas(
            self.num_timesteps, self.beta_schedule, **self.beta_kwargs
        )  # shape [T]
        alphas = 1.0 - betas  # shape [T]
        alpha_bars = jnp.cumprod(alphas)  # shape [T], cumulative product

        # ----- 2. Initialize x_T (actions at final step) as pure Gaussian noise -----
        action_shape = (batch_size, *self.unbatched_prediction_shape)  # [B, A, a]
        prng_key, x_t = sample_gaussian_noise(prng_key, action_shape)

        # ----- 3. Reverse diffusion from t=T-1 down to 0 -----
        for i in reversed(range(self.num_timesteps)):
            # a) Grab the relevant alpha/bar values
            alpha_bar_t = alpha_bars[i]  # alpha_bar_i
            alpha_bar_prev = alpha_bars[i - 1] if i > 0 else 1.0

            # b) Model predicts the noise in x_t
            t_array = jnp.full((batch_size,), i, dtype=jnp.int32)
            predicted_noise, _ = model.apply(
                params,
                observation=observation,
                timesteps=t_array,
                noisy_action=x_t,
                inference_mode=True,
                deterministic=True,
            )

            # c) Estimate the "predicted" clean action x_0
            denom = jnp.sqrt(alpha_bar_t + 1e-7)
            x0 = (x_t - jnp.sqrt(1.0 - alpha_bar_t) * predicted_noise) / denom

            # d) Compute the mean of q(x_{t-1} | x_t, x_0)
            mean_coef = jnp.sqrt(alpha_bar_prev)
            x_t_minus_1_mean = mean_coef * x0 + jnp.sqrt(1.0 - alpha_bar_prev) * predicted_noise

            # e) Compute the posterior variance
            beta_t = betas[i]
            var_coef = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-7)
            posterior_var = beta_t * var_coef

            # f) Sample x_{t-1} from N(mean, var) unless we are at t=0
            if i > 0:
                prng_key, subkey = jax.random.split(prng_key)
                z = jax.random.normal(subkey, x_t.shape)
                x_t = x_t_minus_1_mean + jnp.sqrt(posterior_var) * z
            else:
                x_t = x_t_minus_1_mean

        return prng_key, x_t

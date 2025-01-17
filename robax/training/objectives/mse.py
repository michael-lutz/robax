"""Basic MSE objective."""

from typing import Any, Dict, Tuple

import attrs
import flax.linen as nn
import jax
import jax.numpy as jnp

from robax.training.objectives.base_train_step import BaseTrainStep
from robax.utils.observation import Observation


@attrs.define(frozen=True)
class MSEObjective(BaseTrainStep):
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
        predicted_action, _ = model.apply(params, observation=observation)
        mse_loss = jnp.mean(jnp.square(predicted_action - target))
        return mse_loss

    def get_additional_inputs(
        self,
        prng_key: jax.Array,
        observation: Observation,
        unbatched_prediction_shape: Tuple[int, int],
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Generates the additional inputs for the train step."""
        return prng_key, {}

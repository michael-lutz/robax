"""Basic MSE objective."""

from typing import Any, Dict, Tuple

import attrs
import flax.linen as nn
import jax
import jax.numpy as jnp

from robax.objectives.base_inference_step import BaseInferenceStep
from robax.objectives.base_train_step import BaseTrainStep
from robax.utils.observation import Observation


@attrs.define(frozen=True)
class MSEObjective(BaseTrainStep):
    """MSE objective."""

    def get_loss(
        self,
        params: Dict[str, Any],
        model: nn.Module,
        observation: Observation,
        target: jax.Array,
        debug: bool = False,
        **additional_inputs: jax.Array,
    ) -> jax.Array:
        """Computes the MSE loss."""

        predicted_action, _ = model.apply(params, observation=observation)
        mse_loss = jnp.mean(jnp.square(predicted_action - target))
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
        """Generates the additional inputs for the train step."""
        return prng_key, {}


@attrs.define(frozen=True)
class MSEInferenceStep(BaseInferenceStep):
    """MSE inference step."""

    def generate_action(
        self,
        prng_key: jax.Array,
        params: Dict[str, Any],
        model: nn.Module,
        observation: Observation,
        **kwargs: Any,
    ) -> Tuple[jax.Array, jax.Array]:
        """Generates an action from the policy."""
        predicted_action, _ = model.apply(params, observation=observation)
        return prng_key, predicted_action

"""Basic MLP Policy"""

from typing import Any, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from robax.model.policy.base_policy import BasePolicy
from robax.utils.observation import Observation


class MLPPolicy(BasePolicy):
    """Basic MLP Policy that predicts the next action given the current state."""

    num_layers: int
    embed_dim: int
    unbatched_prediction_shape: Tuple[int, ...]

    def embed_action(self, action: jax.Array, **additional_inputs: jax.Array) -> jax.Array:
        """Embed the action into the action expert"""
        return nn.Dense(features=self.embed_dim)(action)

    def embed_proprio(self, proprio: jax.Array, **additional_inputs: jax.Array) -> jax.Array:
        """Embed the proprioceptive features into the action expert"""
        return nn.Dense(features=self.embed_dim)(proprio)

    def embed_images(self, images: jax.Array, **additional_inputs: jax.Array) -> jax.Array:
        """Embed the images into the gemma expert"""
        raise NotImplementedError

    def embed_text(self, text: jax.Array, **additional_inputs: jax.Array) -> jax.Array:
        """Embed the text"""
        raise NotImplementedError

    @nn.compact
    def __call__(
        self,
        observation: Observation,
        *,
        inference_mode: bool = False,
        deterministic: bool = True,
        return_intermediates: bool = False,
        **additional_inputs: jax.Array,  # only for inputs, hyperparameters should be object vars
    ) -> Tuple[jax.Array, Dict[str, Any]]:
        """Primary call function for the policy"""
        action = observation["action"]
        proprio = observation["proprio"]
        assert proprio is not None

        proprio = proprio.reshape((proprio.shape[0], -1))
        proprio_emb = self.embed_proprio(proprio)

        if action is not None:
            action = action.reshape((action.shape[0], -1))
            action_emb = self.embed_action(action)
            x = jnp.concatenate([action_emb, proprio_emb], axis=-1)
        else:
            x = proprio_emb

        batch_size = x.shape[0]
        if return_intermediates:
            intermediates = {"x_init": x}
        else:
            intermediates = {}
        for i in range(self.num_layers - 1):
            x = nn.Dense(features=self.embed_dim)(x)
            if return_intermediates:
                intermediates[f"x_{i}"] = x

        prediction_shape_flattened = (
            self.unbatched_prediction_shape[0] * self.unbatched_prediction_shape[1]
        )

        x = nn.Dense(features=prediction_shape_flattened)(x)
        if return_intermediates:
            intermediates[f"x_{self.num_layers - 1}"] = x

        x = x.reshape((batch_size, *self.unbatched_prediction_shape))
        return x, intermediates

    def generate_action(
        self,
        prng: jax.Array,
        observation: Observation,
        **kwargs: Any,
    ) -> Tuple[jax.Array, jax.Array]:
        """Generate an action from the policy. Returns prng key and current step action."""
        action, _ = self(observation, return_intermediates=False)
        return prng, action

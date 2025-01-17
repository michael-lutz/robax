"""Base Policy Class"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import flax.linen as nn
import jax

from robax.utils.observation import Observation


class BasePolicy(ABC, nn.Module):
    """Abstract Base Class for PiZero Policy"""

    @abstractmethod
    def embed_action(self, action: jax.Array, **additional_inputs: jax.Array) -> jax.Array:
        """Embed the action into the action expert"""
        pass

    @abstractmethod
    def embed_proprio(self, proprio: jax.Array, **additional_inputs: jax.Array) -> jax.Array:
        """Embed the proprioceptive features into the action expert"""
        pass

    @abstractmethod
    def embed_images(self, images: jax.Array, **additional_inputs: jax.Array) -> jax.Array:
        """Embed the images into the gemma expert"""
        pass

    @abstractmethod
    def embed_text(self, text: jax.Array, **additional_inputs: jax.Array) -> jax.Array:
        """Embed the text"""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def generate_action(
        self,
        prng: jax.Array,
        observation: Observation,
        **kwargs: Any,
    ) -> Tuple[jax.Array, jax.Array]:
        """Generate an action from the policy. Returns prng key and current step action."""
        pass

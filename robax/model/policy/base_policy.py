"""Base Policy Class"""

from abc import ABC, abstractmethod
from typing import Tuple

import jax


class PiZeroBase(ABC):
    """Abstract Base Class for PiZero Policy"""

    @abstractmethod
    def embed_action(self, action: jax.Array, timesteps: jax.Array) -> jax.Array:
        """Embed the action into the action expert"""
        pass

    @abstractmethod
    def embed_proprio(self, proprio: jax.Array) -> jax.Array:
        """Embed the proprioceptive features into the action expert"""
        pass

    @abstractmethod
    def embed_images(self, images: jax.Array) -> jax.Array:
        """Embed the images into the gemma expert"""
        pass

    @abstractmethod
    def embed_text(self, text: jax.Array) -> jax.Array:
        """Embed the text"""
        pass

    @abstractmethod
    def __call__(
        self,
        images: jax.Array,
        text: jax.Array,
        proprio: jax.Array,
        action: jax.Array,
        *,
        inference_mode: bool = False,
        deterministic: bool = True,
        return_intermediates: bool = False,
        **bespoke_inputs: jax.Array,  # only for inputs, hyperparameters should be object vars
    ) -> Tuple[jax.Array, dict]:
        """Primary call function for the policy"""
        pass

    @abstractmethod
    def generate_action(
        self,
        prng: jax.Array,
        images: jax.Array,
        text: jax.Array,
        proprio: jax.Array,
        action_shape: Tuple[int, ...],
        *,
        num_steps: int = 10,
    ) -> jax.Array:
        """Generate an action from the policy."""
        pass

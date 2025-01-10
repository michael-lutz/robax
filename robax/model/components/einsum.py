"""Parametrized Einsum implementation, as used in Gemma"""

import jax
import jax.numpy as jnp
import flax.linen as nn


class Einsum(nn.Module):
    """Parametrized Einsum implementation

    Note: this is taken from the parameterised Einsum implementation in Gemma

    Attributes:
        shape: shape of the output
        w_init: initializer for the weights
    """

    shape: tuple[int, ...]
    w_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        """Apply einsum to input x

        Args:
            eqn: equation to apply
            x: input to apply einsum to

        Returns:
            [B, L, D] output
        """
        w = self.param("w", self.w_init, self.shape)
        return jnp.einsum(eqn, x, w)

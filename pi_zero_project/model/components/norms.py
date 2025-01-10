"""Base implementation of norms (e.g. RMSNorm)"""

import flax.linen as nn
import jax
import jax.numpy as jnp


class RMSNorm(nn.Module):
    """RMSNorm implementation"""

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply RMSNorm to input x

        Args:
            x: [B, L, D] input to normalize

        Returns:
            [B, L, D] normalized input
        """
        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs

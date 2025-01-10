"""Base token embedding table implementation"""

import flax.linen as nn
import jax
import jax.numpy as jnp


class Embedder(nn.Module):
    """Token embedding table.

    Attributes:
        vocab_size: size of the input vocabulary
        embed_dim: dimension of the embeddings
    """

    vocab_size: int
    embed_dim: int

    def setup(self) -> None:
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_in",
                distribution="normal",
                in_axis=1,
                out_axis=0,
            ),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x: jax.Array) -> jax.Array:
        """Encode input tokens into embeddings

        Args:
            x: [B, L] input tokens

        Returns:
            [B, L, D] output embeddings
        """
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x: jax.Array) -> jax.Array:
        """Decode embeddings into logits

        Args:
            x: [B, L, D] embeddings

        Returns:
            [B, L, K] logits where K is the size of the output vocabulary
        """
        return jnp.dot(x, self.input_embedding_table.T)

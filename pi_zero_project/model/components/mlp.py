"""Linear layer implementation"""

from typing import Optional
import flax.linen as nn
import jax
import jax.numpy as jnp


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

    Note: this is taken from MlpBlock initially used in ViT
    """

    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        """Applies Transformer MlpBlock module.

        Args:
            x: [..., D] input tokens
            deterministic: whether to apply dropout

        Returns:
            [..., D] output embeddings
        """
        inits = dict(
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )

        d = x.shape[-1]
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        # In some extreme batch-size cases, this is needed as of Sept 2024:
        x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        x = nn.Dense(d, dtype=self.dtype_mm, **inits)(x)
        return x


class FeedForward(nn.Module):
    """Gated feed forward module with GELU activation and linear transformation.

    Attributes:
        features: embedding dimension
        hidden_dim: hidden dimension
    """

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Gated feed forward module forward pass.

        Args:
            x: [..., D] input tokens

        Returns:
            [..., D] output embeddings
        """
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.variance_scaling(
                1.0, "fan_in", "truncated_normal", in_axis=(1,), out_axis=(0, 2), batch_axis=()
            ),
            ((2, self.features, self.hidden_dim)),
        )
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.variance_scaling(
                1.0, "fan_in", "truncated_normal", in_axis=(0,), out_axis=(1,), batch_axis=()
            ),
            (self.hidden_dim, self.features),
        )
        outputs = jnp.dot(activations, w_linear)  # [..., D]

        return outputs

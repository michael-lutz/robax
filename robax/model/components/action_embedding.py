"""Implements the action embedding module following the Pi_Zero paper"""

from flax import linen as nn
import jax
import jax.numpy as jnp


def sinusoidal_positional_encoding(timesteps: jnp.ndarray, embed_dim: int) -> jnp.ndarray:
    """Maps timesteps [0, 1] to sinusoidal positional encodings of dimension D.

    Args:
        timesteps (jnp.ndarray): Array of shape [B] with timesteps in the range [0, 1].
        D (int): Dimension of the positional encoding. Must be even.

    Returns:
        jnp.ndarray: Array of shape [B, D] with sinusoidal positional encodings.
    """
    assert embed_dim % 2 == 0, "Dimension D must be even."

    frequencies = jnp.exp(jnp.linspace(0, jnp.log(10000.0), embed_dim // 2))  # Shape: [D // 2]
    timesteps = timesteps[:, None]  # Shape: [B, 1]

    sinusoidal_input = timesteps * frequencies  # Shape: [B, D // 2]
    pos_enc = jnp.concatenate(
        [jnp.sin(sinusoidal_input), jnp.cos(sinusoidal_input)], axis=-1
    )  # Shape: [B, D]

    return pos_enc


class ActionEmbedder(nn.Module):
    """Embeds noisy actions into a transformer embedding dimension using a flow-matching timestep.

    As specified in the paper, the embedding is computed as:
        W_3 · swish(W_2 · concat(W_1 · a_{t'}^τ, ϕ(τ))),
    where:
        - a_{t'}^τ is the noisy action.
        - ϕ(τ) is the sinusoidal positional encoding of the timestep.
        - W_1, W_2, W_3 are learnable linear transformations.
        - swish is the non-linear activation function.

    Args:
        action_dim: Dimension of the noisy action input (a).
        embed_dim: Embedding dimension or width (D).
    """

    embed_dim: int  # Embedding dimension or width (w)

    @nn.compact
    def __call__(self, noisy_action: jax.Array, timestep: jax.Array) -> jax.Array:
        """Computes the embedding for a noisy action and timestep.

        Args:
            noisy_action: Noisy action chunk [B, A, a].
            timestep: Flow matching timestep [B].

        Returns:
            [B, A, D] Transformer embedding.
        """
        assert len(noisy_action.shape) == 3, "Noisy action must have shape [B, A, a]."
        assert len(timestep.shape) == 1, "Timestep must have shape [B]."

        timestep_embed = sinusoidal_positional_encoding(timestep, self.embed_dim)  # [B, D]
        timestep_embed = jnp.tile(
            timestep_embed[:, None, :], (1, noisy_action.shape[1], 1)
        )  # [B, A, D]

        W1 = nn.Dense(self.embed_dim, name="W1")
        W2 = nn.Dense(self.embed_dim, name="W2")
        W3 = nn.Dense(self.embed_dim, name="W3")

        action_embed = W1(noisy_action)  # [B, A, a] -> [B, A, D]
        combined_input = jnp.concatenate([action_embed, timestep_embed], axis=-1)  # [B, A, 2D]
        expanded_input = W2(combined_input)
        swish_output = nn.swish(expanded_input)
        final_embedding = W3(swish_output)  # [B, A, 2D] -> [B, A, D]

        return final_embedding

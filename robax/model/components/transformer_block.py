"""Transformer Block Implementation"""

import flax.linen as nn
import jax
import jax.numpy as jnp

from robax.model.components.attention import Attention
from robax.model.components.mlp import FeedForward
from robax.model.components.norms import RMSNorm


class TransformerBlock(nn.Module):
    """Transformer block.

    Attributes:
        num_heads: number of attention heads
        num_kv_heads: number of key/value heads
        embed_dim: embedding dimension
        head_dim: dimension of each attention head
        hidden_dim: hidden dimension
    """

    num_heads: int
    num_kv_heads: int
    embed_dim: int
    head_dim: int
    hidden_dim: int

    query_pre_attn_norm: str
    attn_logits_softcap: float | None
    post_norms: bool

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    cache_dtype: str | None = None

    def setup(self) -> None:
        self.pre_attention_norm = RMSNorm()
        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            features=self.embed_dim,
            head_dim=self.head_dim,
            cache_dtype=self.cache_dtype,
            query_pre_attn_norm=self.query_pre_attn_norm,
            attn_logits_softcap=self.attn_logits_softcap,
        )
        self.pre_ffw_norm = RMSNorm()
        self.mlp = FeedForward(features=self.embed_dim, hidden_dim=self.hidden_dim)
        if self.dropout:
            self.drop = nn.Dropout(self.dropout, self.dropout_bdims)
        else:
            self.drop = lambda x, _: x
        if self.post_norms:
            self.post_attention_norm = RMSNorm()
            self.post_ffw_norm = RMSNorm()

    def __call__(
        self,
        x: jax.Array,
        *,
        attn_mask: jax.Array,
        use_kv_cache: bool,
        deterministic: bool = True,
    ) -> jax.Array:
        """Transformer block forward pass.

        Args:
            x: [B, L, D] input tokens
            attn_mask: [B, 1, L, L] attention mask
            use_kv_cache: whether to use kv-cache
            deterministic: whether to use dropout

        Returns:
            [N, L, D] output embeddings
        """
        x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))  # type: ignore
        inputs_normalized = self.pre_attention_norm(x)  # [N, L, D]
        positions = jnp.arange(x.shape[1])[None, :]  # [1, L]
        attn_output = self.attn(inputs_normalized, positions, attn_mask, use_kv_cache)  # [N, L, D]

        if self.post_norms:
            attn_output = self.post_attention_norm(attn_output)

        attn_output = self.drop(attn_output, deterministic)
        attn_output += x  # residual connection
        residual = attn_output
        attn_output = self.pre_ffw_norm(attn_output)
        outputs = self.mlp(attn_output)
        outputs = self.drop(outputs, deterministic)  # [N, L, D]
        if self.post_norms:
            outputs = self.post_ffw_norm(outputs)
        outputs = residual + outputs  # [N, L, D]
        return outputs

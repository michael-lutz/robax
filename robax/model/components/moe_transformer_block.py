"""Mixture of Experts Attention Block Implementation, following pi_zero implementation"""

from typing import Dict, List, Tuple, TypedDict

import flax.linen as nn
import jax
import jax.numpy as jnp

from robax.model.components.attention import Attention, apply_attention
from robax.model.components.mlp import FeedForward
from robax.model.components.norms import RMSNorm


class MixtureSpec(TypedDict):
    """Specification for a mixture of experts."""

    mlp_dim: int
    embed_dim: int


class MoETransformerBlock(nn.Module):
    """Mixture of Experts Transformer block.

    Attributes:
        mixture_specs: Dictionary of specifications for each mixture's 'mlp_dim', 'embed_dim'.
        num_heads: number of attention heads
        num_kv_heads: number of key/value heads
        head_dim: dimension of each attention head
        query_pre_attn_norm: normalization strategy before attention
        attn_logits_softcap: soft cap for attention logits
        post_norms: whether to apply post-attention norms
        dropout: dropout rate
        dropout_bdims: dimensions for dropout
        cache_dtype: data type for cache
    """

    mixture_specs: Dict[str, MixtureSpec]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    query_pre_attn_norm: str
    attn_logits_softcap: float | None
    post_norms: bool
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    cache_dtype: str | None = None

    def setup(self) -> None:
        self.pre_attention_norms = {
            name: RMSNorm(name=f"{name}_pre_attn_norm") for name in self.mixture_specs
        }

        self.attentions = {
            name: Attention(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                features=spec["embed_dim"],
                head_dim=self.head_dim,
                cache_dtype=self.cache_dtype,
                query_pre_attn_norm=self.query_pre_attn_norm,
                attn_logits_softcap=self.attn_logits_softcap,
                name=f"{name}_attn",
            )
            for name, spec in self.mixture_specs.items()
        }

        self.pre_ffw_norms = {
            name: RMSNorm(name=f"{name}_pre_ffw_norm") for name in self.mixture_specs
        }
        self.mlps = {
            name: FeedForward(features=spec["embed_dim"], hidden_dim=spec["mlp_dim"])
            for name, spec in self.mixture_specs.items()
        }

        if self.dropout:
            self.drop = nn.Dropout(self.dropout, self.dropout_bdims)
        else:
            self.drop = lambda x, _: x
        if self.post_norms:
            self.post_attention_norms = {
                name: RMSNorm(name=f"{name}_post_attn_norm") for name in self.mixture_specs
            }
            self.post_ffw_norms = {
                name: RMSNorm(name=f"{name}_post_ffw_norm") for name in self.mixture_specs
            }

    def _process_attention_output(
        self, attn_output: jax.Array, residual: jax.Array, deterministic: bool, mixture_name: str
    ) -> jax.Array:
        """Standard attention processing with residual connections, norms, and MLPs.

        Args:
            attn_output: [N, L, D] attention output
            residual: [N, L, D] residual connection
            deterministic: whether to use dropout
            name: name of the mixture
        Returns:
            [N, L, D] processed attention output
        """
        if self.post_norms:
            attn_output = self.post_attention_norms[mixture_name](attn_output)
        attn_output = self.drop(attn_output, deterministic)
        attn_output += residual
        attn_output = self.pre_ffw_norms[mixture_name](attn_output)
        attn_output = self.mlps[mixture_name](attn_output)
        attn_output = self.drop(attn_output, deterministic)
        if self.post_norms:
            attn_output = self.post_ffw_norms[mixture_name](attn_output)
        return attn_output + residual

    def __call__(
        self,
        x: List[Tuple[str, jax.Array]],
        *,
        attn_mask: jax.Array,
        use_kv_cache: bool,
        deterministic: bool,
    ) -> List[Tuple[str, jax.Array]]:
        """Transformer block forward pass.

        Args:
            x: list of (mixture name, [B, L, D] input embeddings)
            attn_mask: [B, 1, L, S] attention mask
            use_kv_cache: whether to use kv-cache
            deterministic: whether to use dropout
        Returns:
            [B, L, D] output embeddings
        """
        mixtures_shapes = []
        all_q, all_k, all_v = [], [], []
        for i, (mixture_name, x_mixture) in enumerate(x):
            mixtures_shapes.append(x_mixture.shape[1])
            inputs_normalized = self.pre_attention_norms[mixture_name](x_mixture)  # [B, L, D]

            start_idx = sum(mixtures_shapes[:i])
            end_idx = start_idx + mixtures_shapes[i]
            positions = jnp.arange(start_idx, end_idx, dtype=jnp.int32)[None, :]

            q_mixture, k_mixture, v_mixture = self.attentions[mixture_name].get_qkv(
                inputs_normalized,
                positions,
                cache_size=attn_mask.shape[-1],
                decode=use_kv_cache,
            )  # 3 x [B, L, H]

            all_q.append(q_mixture)
            all_k.append(k_mixture)
            all_v.append(v_mixture)

        all_q = jnp.concatenate(all_q, axis=1)
        all_k = jnp.concatenate(all_k, axis=1)
        all_v = jnp.concatenate(all_v, axis=1)
        # all [B, L, H]

        # Apply attention using the reordered attention mask in a single operation
        attn_output = apply_attention(
            all_q,
            all_k,
            all_v,
            attn_mask,
            attn_logits_softcap=self.attn_logits_softcap,
        )

        # Breaking down to the individual mixtures, projecting back to respective embed dims
        output = []
        for i, (mixture_name, x_mixture) in enumerate(x):
            # isolate for mixture and project back to embed dim
            start_idx = sum(mixtures_shapes[:i])
            end_idx = start_idx + mixtures_shapes[i]
            attn_output_mixture = self.attentions[mixture_name].proj_to_embed_dim(
                attn_output[:, start_idx:end_idx, :]
            )

            # residual connection + MLP
            output.append(
                (
                    mixture_name,
                    self._process_attention_output(
                        attn_output_mixture, x_mixture, deterministic, mixture_name
                    ),
                )
            )

        return output

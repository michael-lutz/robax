# https://github.com/google-research/big_vision/tree/main/big_vision/models/proj/paligemma

# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gemma reimplementation for big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")

Example Colab using the models via the PaliGemma decoding logic:
(internal link)

Doc locating the variable initializers in the original code and validating them:
(internal link)

This implementation does *not* currently support the local sliding attention
pattern used in the v2 models. But since we mostly use sequences <4096 tokens,
this shouldn't make any difference. Since RoPE embedding is used throughout,
it's unclear if there is any practical difference (other than wasting some
memory).
"""


from typing import Any, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import orbax.checkpoint

from pi_zero_project.model.components.attention import Attention
from pi_zero_project.model.components.mlp import FeedForward
from pi_zero_project.model.components.norms import RMSNorm
from pi_zero_project.model.components.token_embed import Embedder
from pi_zero_project.model.components.transformer_block import Block
from pi_zero_project.model.vlm.llm_model.base_llm_model import BaseLlmModel
import pi_zero_project.utils.param_utils as u


def get_config(variant):
    """Returns config for specified gemma variant."""
    if variant == "gemma_2b":
        return ml_collections.ConfigDict(
            dict(
                variant=variant,
                width=2048,
                depth=18,
                mlp_dim=16_384,
                num_heads=8,
                num_kv_heads=1,
                head_dim=256,
                norm_eps=1e-6,
                vocab_size=256_000,
                scan=True,
                remat_policy="nothing_saveable",
            )
        )
    if variant == "gemma_7b":
        return ml_collections.ConfigDict(
            dict(
                variant=variant,
                width=3072,
                depth=28,
                mlp_dim=24_576,
                num_heads=16,
                num_kv_heads=16,
                head_dim=256,
                norm_eps=1e-6,
                vocab_size=256_000,
                scan=True,
                remat_policy="nothing_saveable",
            )
        )
    if variant == "gemma2_2b":
        return ml_collections.ConfigDict(
            dict(
                variant=variant,
                width=2304,
                depth=26,
                mlp_dim=9_216,
                num_heads=8,
                num_kv_heads=4,
                head_dim=256,
                norm_eps=1e-6,
                vocab_size=256_000,
                final_logits_softcap=30.0,
                attn_logits_softcap=50.0,
                post_norms=True,
                scan=True,
                remat_policy="nothing_saveable",
            )
        )
    if variant == "gemma2_9b":
        return ml_collections.ConfigDict(
            dict(
                variant=variant,
                width=3584,
                depth=42,
                mlp_dim=14_336,
                num_heads=16,
                num_kv_heads=8,
                head_dim=256,
                norm_eps=1e-6,
                vocab_size=256_000,
                final_logits_softcap=30.0,
                attn_logits_softcap=50.0,
                post_norms=True,
                scan=True,
                remat_policy="nothing_saveable",
            )
        )
    if variant == "gemma2_27b":
        return ml_collections.ConfigDict(
            dict(
                variant=variant,
                width=4608,
                depth=46,
                mlp_dim=36_864,
                num_heads=32,
                num_kv_heads=16,
                head_dim=128,
                norm_eps=1e-6,
                vocab_size=256_000,
                query_pre_attn_norm="rsqrt_emb_per_head",
                final_logits_softcap=30.0,
                attn_logits_softcap=50.0,
                post_norms=True,
                scan=True,
                remat_policy="nothing_saveable",
            )
        )
    raise ValueError(f"Unknown variant: {variant}")


class Model(nn.Module, BaseLlmModel):
    """The Gemma Model

    Attributes:
        variant: variant of the model
        width: embedding dimension
        depth: number of layers
        mlp_dim: hidden dimension
        num_heads: number of attention heads
        num_kv_heads: number of key/value heads
        head_dim: dimension of each attention head
        norm_eps: epsilon for normalization
        vocab_size: size of the input vocabulary
        query_pre_attn_norm: query pre-attention normalization
        final_logits_softcap: final logits softcap
        attn_logits_softcap: attention logits softcap
        post_norms: whether to use post-norms
        dropout: dropout rate
        dropout_bdims: dimensions to apply dropout to
        cache_dtype: cache dtype
        embed_dtype: embedding dtype
        scan: whether to use scan
        remat_policy: remat policy
    """

    variant: str

    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    norm_eps: float
    vocab_size: int

    query_pre_attn_norm: str = "rsqrt_head_dim"
    final_logits_softcap: float = 0.0
    attn_logits_softcap: float = 0.0
    post_norms: bool = False

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    cache_dtype: str | None = None

    # TODO: Wire this in all places needed so that the model can be
    # run with different activation dtype. For now only float32 runs.
    embed_dtype: str = "float32"

    scan: bool = False
    remat_policy: str = "none"

    @nn.compact
    def __call__(
        self,
        tokens,
        *,
        embedded_prefix=None,
        embed_only=False,
        pre_logits=None,
        positions=None,
        mask=None,
        decode=False,
        deterministic=True,
    ):
        """Embed only, or complete forward pass.

        Args:
            tokens: Embedded, then and appended to `embedded_prefix`. Can be None.
            embedded_prefix: Optional prefix that is already embedded.
            embed_only: Whether to compute embeddings only.
            pre_logits: If present computes logits from pre_logits and returns.
            positions: Optional `[B, T]` allows to specify the absolute position of the tokens.
            mask: Optional attention mask `[B, T, S]`.
            decode: Whether to use kv-cache. Caller must pass masks and positions.
            deterministic: Forwarded to all dropout layers.

        Returns:
            If `embed_only=False`, then `(logits, out)` will be returned.
            If `embed_only=True`, then the embeddings will be returned.
        """
        out = {}

        embedder = Embedder(vocab_size=self.vocab_size, embed_dim=self.width, name="embedder")

        if pre_logits is not None:
            x = out["pre_logits"] = pre_logits
            logits = out["logits"] = embedder.decode(x)
            return logits, out

        x = []
        if embedded_prefix is not None:
            x.append(embedded_prefix)
        if tokens is not None:
            x.append(embedder.encode(tokens))

        x = jnp.concatenate(x, axis=-2)
        x = x.astype(self.embed_dtype)
        batch_size, seq_len, width = x.shape

        if embed_only:
            return x

        if decode:
            assert (
                positions is not None and mask is not None
            ), "Must explicitly pass positions and mask for decoding."

        if positions is None:
            positions = jnp.arange(seq_len).astype(jnp.int32)[None, :]
        assert positions.shape[1] == x.shape[1], (positions.shape, x.shape)

        if mask is None:
            mask = nn.attention.make_causal_mask(jnp.ones([batch_size, seq_len]))
        if mask.ndim == 3:
            mask = mask[:, None, :, :]
        cache_size = max(seq_len, mask.shape[-1])
        assert mask.shape == (batch_size, 1, seq_len, cache_size), mask.shape

        if self.remat_policy == "none":
            block_cls = Block
        else:
            block_cls = nn.remat(
                Block,
                prevent_cse=not self.scan,
                static_argnums=(5, 6),  # 0=self, 5=decode, 6=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy),
            )

        block_kw = dict(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            embed_dim=width,
            hidden_dim=self.mlp_dim,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            cache_dtype=self.cache_dtype,
            query_pre_attn_norm=self.query_pre_attn_norm,
            attn_logits_softcap=self.attn_logits_softcap,
            post_norms=self.post_norms,
        )
        layers = self.scope.push("layers")  # type: ignore
        if self.scan:
            blocks = [
                nn.scan(
                    block_cls,
                    # cache has axis 1 since we want leading dimension to be batch size.
                    variable_axes={"params": 0, "cache": 1},
                    split_rngs={"params": True, "dropout": True},
                    in_axes=nn.broadcast,  # type: ignore
                    length=self.depth,
                )(
                    parent=layers, **block_kw  # type: ignore
                )
            ]
        else:
            blocks = [
                block_cls(
                    parent=layers.push(str(layer)),  # type: ignore
                    **block_kw,  # type: ignore
                )
                for layer in range(self.depth)
            ]
        unused_scan_arg = ()
        for block in blocks:
            x, unused_scan_arg = block(x, unused_scan_arg, positions, mask, decode, deterministic)

        assert x.dtype == jnp.dtype(self.embed_dtype)  # Sanity check.
        out["encoded"] = x

        x = RMSNorm(name="final_norm")(x)
        out["pre_logits"] = x

        x = embedder.decode(x)
        out["logits_pre_norm"] = x
        if self.final_logits_softcap:
            x = jnp.tanh(x / self.final_logits_softcap) * self.final_logits_softcap
        out["logits"] = x

        return x, out


_ORBAX_INITS = {}
_BV_INITS = {}


def _load_orbax(path):
    """Loads and coverts Orbax gemma checkpoint."""
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    params = checkpointer.restore(path)
    params = flax.traverse_util.unflatten_dict(params, sep="/")["transformer"]  # type: ignore
    n = sum(1 for k in params if k.startswith("layer_"))
    params["layers"] = jax.tree.map(
        lambda *xs: np.stack(xs), *(params.pop(f"layer_{i}") for i in range(n))
    )
    mlp = params["layers"]["mlp"]
    mlp["gating_einsum"] = mlp["gating_einsum"].pop("w")
    mlp["linear"] = mlp["linear"].pop("w")
    return params


def _del_pad_rows(params):
    """Some checkpoints have 128 unused padding tokens."""
    emb = params["embedder"]["input_embedding"]
    if emb.shape[0] == 256_128:
        params["embedder"]["input_embedding"] = jax.device_get(emb)[:256_000]
    assert params["embedder"]["input_embedding"].shape[0] == 256_000


def _maybe_transpose_gating_einsum(params):
    """The `transpose_gating_einsum` case in gemma/modules.py."""
    mlp = params["layers"]["mlp"]
    *_, d1, d2 = mlp["gating_einsum"].shape
    if d1 > d2:
        *ns, n1, n2 = range(len(mlp["gating_einsum"].shape))
        mlp["gating_einsum"] = mlp["gating_einsum"].transpose(*ns, n2, n1)


def _load_like_bv(params):
    params = jax.tree.map(lambda x: x, params)
    _del_pad_rows(params)
    _maybe_transpose_gating_einsum(params)
    return params


def load(init_params, init_file, model_cfg=None, dont_load=()):
    """Loads existing weights."""
    model_cfg = model_cfg or {}
    variant = model_cfg.get("variant", "gemma_2b")
    init_variant = f"{init_file} {variant}"
    if init_variant in _ORBAX_INITS:
        params = _load_like_bv(_load_orbax(_ORBAX_INITS[init_variant]))
    elif init_variant in _BV_INITS:
        params = _load_like_bv(u.load_params(_BV_INITS[init_variant]))
    else:
        params = u.load_params(init_file)

    def extend_rows(emb1, target_rows):
        if (missing_rows := target_rows - emb1.shape[0]) == 0:
            return emb1
        assert missing_rows > 0, "You're asking to shrink vocab?!"
        new_rows = np.random.randn(missing_rows, emb1.shape[1])
        new_rows = (new_rows * 0.02).astype(emb1.dtype)
        return np.r_[np.asarray(emb1), new_rows]

    if "vocab_size" in model_cfg:
        params["embedder"]["input_embedding"] = extend_rows(  # type: ignore
            params["embedder"]["input_embedding"],  # type: ignore
            model_cfg["vocab_size"],
        )

    return u.merge_params(params, init_params, dont_load)

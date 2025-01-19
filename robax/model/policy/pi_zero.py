"""Pi Zero Policy Implementation

There are many dimensions... For the reader's benefit:
- B: batch size
- I: number of input images
- T: number of input tokens
- A: number of actions to predict
- L: overall sequence length (L = I + T + 1 + A)
- D: hidden dimension
- H: hidden dimension per head

- num_heads: number of attention heads
- h_i, w_i: height and width of the image
- p: number of proprioceptive features
- a: size of action dimension


Essentially runs attention with the following mask:

    --gemma-- --action expert--
    img + txt    prop   act   ]
    [i, i, t, t, p, p, a, a, a]

    [1, 1, 1, 1, 0, 0, 0, 0, 0]
    [1, 1, 1, 1, 0, 0, 0, 0, 0]
    [1, 1, 1, 1, 0, 0, 0, 0, 0]
    [1, 1, 1, 1, 0, 0, 0, 0, 0]
    [1, 1, 1, 1, 1, 1, 0, 0, 0]
    [1, 1, 1, 1, 1, 1, 0, 0, 0]
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
"""

from typing import Any, Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from robax.model.components.action_embedding import ActionEmbedder
from robax.model.components.attention import make_attn_mask
from robax.model.components.moe_transformer_block import (
    MixtureSpec,
    MoETransformerBlock,
)
from robax.model.components.norms import RMSNorm
from robax.model.components.token_embed import Embedder
from robax.model.img_model import vit
from robax.model.policy.base_policy import BasePolicy
from robax.utils.observation import Observation


class PiZero(BasePolicy):
    """Pi_Zero Policy Implementation"""

    # Initialization parameters
    vit_variant: str
    llm_vocab_size: int

    # Broader training
    unbatched_prediction_shape: Tuple[int, int]  # NOTE: this is unbatched...

    # Attention parameters
    mixture_specs: Dict[str, MixtureSpec]
    input_expert_map: Dict[str, str]
    depth: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    query_pre_attn_norm: str = "rsqrt_head_dim"
    attn_logits_softcap: float = 0.0
    post_norms: bool = False
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    scan: bool = False

    cache_dtype: str | None = None
    embed_dtype: str = "float32"
    remat_policy: str = "none"

    @nn.compact
    def __call__(
        self,
        observation: Observation,
        *,
        inference_mode: bool = False,
        deterministic: bool = True,
        return_intermediates: bool = False,
        **additional_inputs: jax.Array,
    ) -> Tuple[jax.Array, Dict[str, Any]]:
        """Primary call function for Pi_Zero

        NOTE: ensure the observations have the following dimensions:
        - images: [batch_size, num_images, height, width, 3]
        - text: [batch_size, num_text_tokens]
        - proprio: [batch_size, num_proprio_tokens, num_proprio_features]
        - action: [batch_size, num_actions_conditioned, num_action_features]
        - timesteps: [batch_size]
        - noisy_action: [batch_size, num_actions_to_generate, num_action_features]

        Args:
            observation: Observation
            inference_mode: bool, whether to return the output of the policy
            deterministic: bool, whether to use deterministic operations
            return_intermediates: bool, whether to return the intermediate embeddings
            **additional_inputs: jax.Array, additional inputs for the policy
                - timesteps: [B] timesteps for flow matching
                - noisy_action: [B, A, a] noisy action to use for flow matching
        Returns:
            [B, A, a] output of the policy
        """
        assert "timesteps" in additional_inputs, "timesteps must be provided"
        assert "noisy_action" in additional_inputs, "noisy_action must be provided"

        timesteps = additional_inputs["timesteps"]
        noisy_action = additional_inputs["noisy_action"]
        obs_images = observation["images"]
        obs_text = observation["text"]
        obs_proprio = observation["proprio"]
        obs_action = observation["action"]

        assert obs_proprio is not None or obs_images is not None, "proprio or images must exist"

        if obs_action is not None:
            action = jnp.concatenate([obs_action, noisy_action], axis=1)
        else:
            action = noisy_action

        out: Dict[str, Any] = {}

        # Embed necessary inputs
        action_token_embed = self.embed_action(action, timesteps=timesteps)  # [B, A, D]

        if obs_proprio is not None:
            proprio_token_embed = self.embed_proprio(obs_proprio)  # [B, P, D]
        else:
            proprio_token_embed = None

        if obs_text is not None:
            text_token_embed = self.embed_text(obs_text)  # [B, T, D]
        else:
            text_token_embed = None

        if obs_images is not None:
            image_token_embed = self.embed_images(obs_images)  # [B, I, D]
        else:
            image_token_embed = None

        if return_intermediates:
            out["image_embeddings"] = image_token_embed
            out["text_embeddings"] = text_token_embed
            out["proprio_embeddings"] = proprio_token_embed
            out["action_embeddings"] = action_token_embed

        # Create attention mask and blocks
        attn_mask = self.make_attention_mask(
            obs_images, obs_text, obs_proprio, action
        )  # [B, 1, L, L]
        blocks = self.create_attention_blocks()

        # Run through attention blocks
        x = []
        if image_token_embed is not None:
            x.append((self.input_expert_map["images"], image_token_embed))
        if text_token_embed is not None:
            x.append((self.input_expert_map["text"], text_token_embed))
        if proprio_token_embed is not None:
            x.append((self.input_expert_map["proprio"], proprio_token_embed))
        if action_token_embed is not None:
            x.append((self.input_expert_map["action"], action_token_embed))

        if return_intermediates:
            out["pre_attention"] = x

        for block in blocks:
            x = block(
                x=x,
                attn_mask=attn_mask,
                use_kv_cache=inference_mode,
                deterministic=deterministic,
            )

        if return_intermediates:
            out["post_attention"] = x

        final_norms = {
            mixture_name: RMSNorm(name=f"{mixture_name}_final_norm")
            for mixture_name in self.mixture_specs.keys()
        }

        for i, (mixture_name, x_mixture) in enumerate(x):
            x[i] = (mixture_name, final_norms[mixture_name](x_mixture))

        if return_intermediates:
            out["final_normed_embeddings"] = x

        action_embeddings = x[-1][1][:, -self.unbatched_prediction_shape[0] :, :]
        action_field_pred = nn.Dense(
            features=self.unbatched_prediction_shape[1], name="proj_action_dim"
        )(action_embeddings)
        # [B, A, a] result

        return action_field_pred, out

    def embed_action(self, action: jax.Array, **additional_inputs: jax.Array) -> jax.Array:
        """Embed the action into the action expert

        Args:
            action: [B, A, a] action
            timesteps: [B] timesteps

        Returns:
            [B, A, D] action embeddings
        """
        timesteps = additional_inputs["timesteps"]
        feature_size = self.mixture_specs[self.input_expert_map["action"]]["embed_dim"]
        action_token_embed = ActionEmbedder(embed_dim=feature_size, name="action_embedder")(
            action, timesteps
        )  # [B, A, D]
        return action_token_embed

    def embed_proprio(self, proprio: jax.Array, **additional_inputs: jax.Array) -> jax.Array:
        """Embed the proprioceptive features into the action expert

        Args:
            proprio: [B, P, p] proprioceptive features

        Returns:
            [B, P, D] proprioceptive embeddings
        """
        feature_size = self.mixture_specs[self.input_expert_map["proprio"]]["embed_dim"]
        proprio_token_embed = nn.Dense(features=feature_size, name="proprio_embedder")(
            proprio
        )  # [B, P, D]
        return proprio_token_embed

    def embed_images(self, images: jax.Array, **additional_inputs: jax.Array) -> jax.Array:
        """Embed the images into the gemma expert

        Args:
            images: [B, I, h_i, w_i, 3] images

        Returns:
            [B, I, D] image embeddings
        """
        B, I = images.shape[:2]
        images = images.reshape(B * I, *images.shape[2:])
        vit_config = vit.decode_variant(self.vit_variant)
        embed_dim = self.mixture_specs[self.input_expert_map["images"]]["embed_dim"]
        image_token_embed, aux = vit.ViT(num_classes=embed_dim, name="img", **vit_config)(
            images
        )  # [B * I, D]
        del aux
        images = images.reshape(B, I, *images.shape[1:])
        image_token_embed = image_token_embed.reshape(B, I, embed_dim)
        assert isinstance(image_token_embed, jax.Array)
        return image_token_embed

    def embed_text(self, text: jax.Array, **additional_inputs: jax.Array) -> jax.Array:
        """Embed the text

        Args:
            text: [B, T] text tokens

        Returns:
            [B, T, D] text embeddings
        """
        embed_dim = self.mixture_specs[self.input_expert_map["text"]]["embed_dim"]
        if text.shape[1] > 0:
            embedder = Embedder(
                vocab_size=self.llm_vocab_size,
                embed_dim=embed_dim,
                name="gemma_embedder",
            )
            text_token_embed = embedder.encode(text)  # [B, T, D]
        else:
            text_token_embed = jnp.zeros((text.shape[0], 0, embed_dim))
        return text_token_embed

    def make_attention_mask(
        self,
        images: jax.Array | None,
        text: jax.Array | None,
        proprio: jax.Array | None,
        action: jax.Array,
    ) -> jax.Array:
        """Make the attention mask for the Pi_Zero policy.

        img + txt    prop   act   ]
        [i, i, t, t, p, p, a, a, a]

        [1, 1, 1, 1, 0, 0, 0, 0, 0]
        [1, 1, 1, 1, 0, 0, 0, 0, 0]
        [1, 1, 1, 1, 0, 0, 0, 0, 0]
        [1, 1, 1, 1, 0, 0, 0, 0, 0]
        [1, 1, 1, 1, 1, 1, 0, 0, 0]
        [1, 1, 1, 1, 1, 1, 0, 0, 0]
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
        [1, 1, 1, 1, 1, 1, 1, 1, 1]

        Args:
            images: [B, I, h_i, w_i, 3] images
            text: [B, T] text tokens
            proprio: [B, P, p] proprioceptive features (P is assumed to be 1 in the paper)
            action: [B, A, a] action to predict

        Returns:
            [B, 1, L, L] attention mask
        """
        B = action.shape[0]
        A = action.shape[1]
        I = images.shape[1] if images is not None else 0
        T = text.shape[1] if text is not None else 0
        P = proprio.shape[1] if proprio is not None else 0
        L = I + T + P + A
        input_mask = jnp.ones([B, L])
        # NOTE: if images are missing, assume the dataloader populated them with all 0
        if images is not None:
            img_mask = jnp.any(images, axis=(-3, -2, -1))
            input_mask = input_mask.at[:, :I].set(input_mask[:, :I] * img_mask)
        mask_ar = jnp.zeros([B, L])
        mask_ar = mask_ar.at[:, I + T].set(1)
        mask_ar = mask_ar.at[:, I + T + P].set(1)
        attn_mask = make_attn_mask(input_mask, mask_ar)
        attn_mask = attn_mask[:, None, :, :]
        return attn_mask

    def create_attention_blocks(self) -> List[MoETransformerBlock]:
        """Helper function to create attention blocks

        Returns:
            List of attention blocks
        """
        if self.remat_policy == "none":
            block_cls = MoETransformerBlock
        else:
            block_cls = nn.remat(
                MoETransformerBlock,
                prevent_cse=not self.scan,
                static_argnums=(8, 9),  # 0=self, 8=use_kv_cache, 9=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy),
            )

        block_kw = dict(
            mixture_specs=self.mixture_specs,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            query_pre_attn_norm=self.query_pre_attn_norm,
            attn_logits_softcap=self.attn_logits_softcap,
            post_norms=self.post_norms,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            cache_dtype=self.cache_dtype,
        )
        layers = self.scope.push("layers")  # type: ignore
        if self.scan:
            blocks = [
                nn.scan(
                    block_cls,
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
                block_cls(parent=layers.push(str(layer)), **block_kw)  # type: ignore
                for layer in range(self.depth)
            ]
        return blocks

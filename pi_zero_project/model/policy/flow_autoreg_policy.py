"""Combined Flow Matching and Autoregressive Action Prediction Policy

NOTE: to implement action chunking, predicts as follows:
- Joint_1 Joint_2 Joint_3 ... Joint_1 Joint_2 Joint_3 ... Joint_1 ...

Works by performing attention with the following mask:

    --gemma-- --action expert-- --gemma--
    img + txt    prop  flow act  auto act
    [i, i, t, t, p, p, a, a, a, r, r, r, ...]

    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ...]
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ...]
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ...]
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ...]
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, ...]
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, ...]
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...]
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...]
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...]
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, ...]
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, ...]
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, ...]
"""

from typing import List, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn

from pi_zero_project.model.components.attention import make_attn_mask
from pi_zero_project.model.components.moe_transformer_block import MoETransformerBlock
from pi_zero_project.model.components.norms import RMSNorm
from pi_zero_project.model.components.token_embed import Embedder
from pi_zero_project.model.components.action_embedding import ActionEmbedder
from pi_zero_project.model.vlm.img_model import vit


class FlowAutoregressiveActionPrediction(nn.Module):
    """Combined Flow Matching and Autoregressive Action Prediction Policy"""

    # Initialization parameters
    vit_variant: str
    llm_vocab_size: int

    # Attention parameters
    gemma_mlp_dim: int
    gemma_embed_dim: int
    action_expert_mlp_dim: int
    action_expert_embed_dim: int
    depth: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

    query_pre_attn_norm: str = "rsqrt_head_dim"
    attn_logits_softcap: float = 0.0
    post_norms: bool = False

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    cache_dtype: str | None = None

    embed_dtype: str = "float32"
    scan: bool = False
    remat_policy: str = "none"

    def setup(self) -> None:
        self.embedder = Embedder(
            vocab_size=self.llm_vocab_size,
            embed_dim=self.gemma_embed_dim,
            name="text_embedder",
        )

    @nn.compact
    def __call__(
        self,
        images: jax.Array,
        text: jax.Array,
        proprio: jax.Array,
        flow_matching_action_chunk: jax.Array,
        autoregressive_action_tokens: jax.Array,
        timesteps: jax.Array,
        *,
        inference_mode: bool = False,
        deterministic: bool = True,
        return_intermediates: bool = False,
    ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        """Primary call function for Pi_Zero

        Args:
            images: [B, I, h_i, w_i, 3] images
            text: [B, T] text tokens
            proprio: [B, P, p] proprioceptive features (P is assumed to be 1 in the paper)
            flow_matching_action_chunk: [B, A, a] flow matching action chunk
            autoregressive_action_tokens: [B, R] autoregressive action tokens. R for autoRegressive
            timesteps: [B] timesteps
            inference_mode: bool, whether to return the output of the policy
            return_intermediates: bool, whether to return the intermediate embeddings
        Returns:
            [B, A, a] output of the policy
        """
        assert (
            images.shape[0]
            == text.shape[0]
            == proprio.shape[0]
            == flow_matching_action_chunk.shape[0]
            == autoregressive_action_tokens.shape[0]
        )
        assert len(images.shape) == 5, "images must be [B, I, h_i, w_i, 3]"
        assert len(text.shape) == 2, "text must be [B, T]"
        assert len(proprio.shape) == 3, "proprio must be [B, P, p]"
        assert (
            len(flow_matching_action_chunk.shape) == 3
        ), "flow_matching_action_chunk must be [B, A, a]"
        assert (
            len(autoregressive_action_tokens.shape) == 2
        ), "autoregressive_action_tokens must be [B, R]"
        assert len(timesteps.shape) == 1, "timesteps must be [B]"

        out = {}

        # Embed necessary inputs
        action_token_embed = self.embed_action(flow_matching_action_chunk, timesteps)  # [B, A, D]
        autoregressive_action_token_embed = self.embed_text(
            autoregressive_action_tokens
        )  # [B, R, D] NOTE: simpler if we reuse text model like OpenVLA
        proprio_token_embed = self.embed_proprio(proprio)  # [B, P, D]
        text_token_embed = self.embed_text(text)  # [B, T, D]
        image_token_embed = self.embed_images(images)  # [B, I, D]

        if return_intermediates:
            out["image_embeddings"] = image_token_embed
            out["text_embeddings"] = text_token_embed
            out["proprio_embeddings"] = proprio_token_embed
            out["action_embeddings"] = action_token_embed

        # Create attention mask and blocks
        attn_mask = self.make_attention_mask(
            images, text, proprio, flow_matching_action_chunk, autoregressive_action_tokens
        )  # [B, 1, L, L]
        blocks = self.create_attention_blocks()

        # Run through attention blocks
        x = [
            ("gemma", jnp.concatenate([image_token_embed, text_token_embed], axis=1)),
            ("action_expert", jnp.concatenate([proprio_token_embed, action_token_embed], axis=1)),
            ("gemma", autoregressive_action_token_embed),
        ]

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

        for i, (mixture_name, x_mixture) in enumerate(x):
            x[i] = (mixture_name, RMSNorm(name=f"{mixture_name}_final_norm")(x_mixture))

        if return_intermediates:
            out["final_normed_embeddings"] = x

        action_field_embeddings = x[1][1][:, -flow_matching_action_chunk.shape[1] :, :]
        action_field_pred = nn.Dense(
            features=flow_matching_action_chunk.shape[2], name="proj_action_dim"
        )(action_field_embeddings)

        autoreg_action_embeddings = x[2][1]
        autoreg_action_tokens = self.embedder.decode(autoreg_action_embeddings)

        return action_field_pred, autoreg_action_tokens, out

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
        """Generate an action from the policy.

        Args:
            prng: [B] PRNG key
            images: [B, I, h_i, w_i, 3] images
            text: [B, T] text tokens
            proprio: [B, P, p] proprioceptive features (P is assumed to be 1 in the paper)
            action_shape: [B, A, a] action shape
        Returns:
            [B, A, a] action
        """
        # TODO: maybe some combination of flow matching and autoregressive action prediction
        pass

    def embed_action(self, action: jax.Array, timesteps: jax.Array) -> jax.Array:
        """Embed the action into the action expert

        Args:
            action: [B, A, a] action
            timesteps: [B] timesteps

        Returns:
            [B, A, D] action embeddings
        """
        action_token_embed = ActionEmbedder(
            embed_dim=self.action_expert_embed_dim, name="action_embedder"
        )(
            action, timesteps
        )  # [B, A, D]
        return action_token_embed

    def embed_proprio(self, proprio: jax.Array) -> jax.Array:
        """Embed the proprioceptive features into the action expert

        Args:
            proprio: [B, P, p] proprioceptive features

        Returns:
            [B, P, D] proprioceptive embeddings
        """
        proprio_token_embed = nn.Dense(
            features=self.action_expert_embed_dim, name="proprio_embedder"
        )(
            proprio
        )  # [B, P, D]
        return proprio_token_embed

    def embed_images(self, images: jax.Array) -> jax.Array:
        """Embed the images into the gemma expert

        Args:
            images: [B, I, h_i, w_i, 3] images

        Returns:
            [B, I, D] image embeddings
        """
        B, I = images.shape[:2]
        images = images.reshape(B * I, *images.shape[2:])
        vit_config = vit.decode_variant(self.vit_variant)
        image_token_embed, aux = vit.ViT(
            num_classes=self.gemma_embed_dim, name="img", **vit_config
        )(
            images
        )  # [B * I, D]
        del aux
        images = images.reshape(B, I, *images.shape[1:])
        image_token_embed = image_token_embed.reshape(B, I, self.gemma_embed_dim)
        return image_token_embed

    def embed_text(self, text: jax.Array) -> jax.Array:
        """Embed the text

        Args:
            text: [B, T] text tokens

        Returns:
            [B, T, D] text embeddings
        """
        if text.shape[1] > 0:
            text_token_embed = self.embedder.encode(text)  # [B, T, D]
        else:
            text_token_embed = jnp.zeros((text.shape[0], 0, self.gemma_embed_dim))
        return text_token_embed

    def make_attention_mask(
        self,
        images: jax.Array,
        text: jax.Array,
        proprio: jax.Array,
        flow_matching_action_chunk: jax.Array,
        autoregressive_action_tokens: jax.Array,
    ) -> jax.Array:
        """Make the attention mask for the Pi_Zero policy.

        Creates a mask as follows:
        img + txt    prop  flow act  auto act
        [i, i, t, t, p, p, a, a, a, r, r, r, ...]

        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ...]
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ...]
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ...]
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ...]
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, ...]
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, ...]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...]
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, ...]
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, ...]
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, ...]

        NOTE: during inference, would likely want to run the flow matching and autoregressive
        prediction in entirely different processes.

        Args:
            images: [B, I, h_i, w_i, 3] images
            text: [B, T] text tokens
            proprio: [B, P, p] proprioceptive features (P is assumed to be 1 in the paper)
            flow_matching_action_chunk: [B, A, a] flow matching action chunk
            autoregressive_action_tokens: [B, R] autoregressive action tokens

        Returns:
            [B, 1, L, L] attention mask
        """

        B = images.shape[0]
        I = images.shape[1]
        T = text.shape[1]
        P = proprio.shape[1]
        A = flow_matching_action_chunk.shape[1]
        R = autoregressive_action_tokens.shape[1]
        L = P + A + R + T + I
        input_mask = jnp.ones([B, L])
        # NOTE: if images are missing, assume the dataloader populated them with all 0
        img_mask = jnp.any(images, axis=(-3, -2, -1))
        input_mask = input_mask.at[:, :I].set(input_mask[:, :I] * img_mask)
        mask_ar = jnp.zeros([B, L])
        mask_ar = mask_ar.at[:, I + T].set(1)
        mask_ar = mask_ar.at[:, I + T + P].set(1)

        # setting all autoregressive action tokens to 1
        mask_ar = mask_ar.at[:, I + T + P + A :].set(1)
        attn_mask = make_attn_mask(input_mask, mask_ar)
        # preventing autoregressive action tokens from attending to flow matching action chunk
        attn_mask = attn_mask.at[:, I + T + P :, : I + T + P, : I + T + P + A].set(0)
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
            mixture_specs=[
                {"name": "gemma", "mlp_dim": self.gemma_mlp_dim, "embed_dim": self.gemma_embed_dim},
                {
                    "name": "action_expert",
                    "mlp_dim": self.action_expert_mlp_dim,
                    "embed_dim": self.action_expert_embed_dim,
                },
            ],
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

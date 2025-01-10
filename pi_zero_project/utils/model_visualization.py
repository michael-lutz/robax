"""Misc. Visualization Functions"""

from typing import List, Tuple
import jax


def visualize_moe_attention_masks(x: List[Tuple[str, jax.Array]], attn_mask: jax.Array) -> None:
    """Visualize the overall attention mask + which mixtures are being attended to.

    NOTE: For simplicity, assumes the attention mask is constant across the batch dimension.

    Args:
        x: list of (mixture name, [B, L, D] input embeddings)
        attn_mask: [..., L, S] attention mask
    """
    import matplotlib.pyplot as plt
    import numpy as np

    mixture_names = [mixture_name for mixture_name, _ in x]
    unique_mixtures = list(set(mixture_names))
    colors = plt.cm.get_cmap("viridis", len(unique_mixtures))
    color_map = {name: colors(i) for i, name in enumerate(unique_mixtures)}

    if len(attn_mask.shape) == 4:
        attn_mask = attn_mask[0, 0, :, :]
    elif len(attn_mask.shape) == 3:
        attn_mask = attn_mask[0, :, :]
    elif len(attn_mask.shape) == 2:
        pass
    else:
        raise ValueError(f"Invalid attention mask shape: {attn_mask.shape}")

    grid = np.zeros(attn_mask.shape + (3,))  # Add a color dimension

    start_idx = 0
    for mixture_name, mixture_data in x:
        num_tokens = mixture_data.shape[1]
        end_idx = start_idx + num_tokens

        for i in range(start_idx, end_idx):
            indices = np.where(attn_mask[i, :] == 1)
            grid[i, indices] = color_map[mixture_name][:3]
        start_idx = end_idx

    plt.imshow(grid, interpolation="nearest")
    plt.title("Attention Mask Visualization")
    plt.show()

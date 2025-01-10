"""Base positional embedding functions"""

from typing import Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


def posemb_sincos_2d(
    h: int, w: int, width: int, temperature: float = 10_000.0, dtype: jnp.dtype = jnp.float32
) -> jax.Array:
    """Follows the MoCo v3 logic.

    Args:
        h: height of the grid
        w: width of the grid
        width: width of the positional embedding
        temperature: temperature for the sinusoidal positional embedding
        dtype: dtype of the positional embedding

    Returns:
        [1, L, D] = [1, h * w, width] positional embedding
    """
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(
    self: nn.Module,
    typ: str,
    seqshape: Tuple[int, int],
    width: int,
    name: str,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Get positional embedding

    Args:
        self: flax module
        typ: type of positional embedding
        seqshape: shape of the sequence
        width: width of the positional embedding
        name: name of the positional embedding
        dtype: dtype of the positional embedding

    Returns:
        [1, L, D] = [1, h * w, width] positional embedding
    """
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    elif typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    else:
        raise ValueError(f"Unknown posemb type: {typ}")


def apply_rope(x: jax.Array, *, positions: jax.Array, max_wavelength: float = 10_000) -> jax.Array:
    """Applies RoPE positions [B, L] to x [B, L, H, D].

    Args:
        x: [B, L, H, D]
        positions: [B, L]
        max_wavelength: max wavelength for RoPE

    Returns:
        [B, L, H, D] RoPE applied to x
    """
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return res

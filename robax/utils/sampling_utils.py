"""Sampling utilities"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp


def sample_beta(
    prng: jax.Array, shape: Tuple[int, ...], beta_a: float, beta_b: float, cutoff_value: float
) -> jax.Array:
    """Samples using the 1 - beta distribution as done in the pi_zero paper"""
    beta_sample = jax.random.beta(prng, a=beta_a, b=beta_b, shape=shape)
    return (1 - beta_sample) * cutoff_value


def sample_uniform(
    prng: jax.Array, shape: Tuple[int, ...], minval: float, maxval: float
) -> jax.Array:
    """Samples from a uniform distribution"""
    unif_sample = jax.random.uniform(prng, shape=shape, minval=minval, maxval=maxval)
    unif_sample = jnp.clip(unif_sample, 1e-5, 1.0 - 1e-5)
    return unif_sample


def sample_gaussian_noise(prng: jax.Array, shape: Tuple[int, ...]) -> Tuple[jax.Array, jax.Array]:
    """Samples Gaussian noise from a standard normal distribution"""
    prng, _ = jax.random.split(prng)
    noise = jax.random.normal(prng, shape)
    return prng, noise


def sample_timesteps(
    prng: jax.Array,
    shape: Tuple[int, ...],
    distribution: str,
    **kwargs: Any,
) -> Tuple[jax.Array, jax.Array]:
    """Samples timesteps from a uniform distribution"""
    if distribution == "beta":
        res = sample_beta(prng, shape, **kwargs)
    elif distribution == "uniform":
        res = sample_uniform(prng, shape, **kwargs)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")

    prng, _ = jax.random.split(prng)
    return prng, res

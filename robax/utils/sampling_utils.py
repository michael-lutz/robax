"""Sampling utilities"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp


def sample_complementary_beta(
    prng: jax.Array,
    shape: Tuple[int, ...],
    beta_a: float,
    beta_b: float,
    cutoff_value: float,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Samples using the 1 - beta distribution as done in the pi_zero paper"""
    beta_sample = jax.random.beta(prng, a=beta_a, b=beta_b, shape=shape, dtype=dtype)
    return (1 - beta_sample) * cutoff_value


def sample_uniform(
    prng: jax.Array,
    shape: Tuple[int, ...],
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Samples from a uniform distribution"""
    unif_sample = jax.random.uniform(prng, shape=shape, minval=minval, maxval=maxval, dtype=dtype)
    unif_sample = jnp.clip(unif_sample, 1e-5, 1.0 - 1e-5)
    return unif_sample


def sample_discrete_uniform(
    prng: jax.Array,
    shape: Tuple[int, ...],
    num_timesteps: int,
    dtype: jnp.dtype = jnp.uint32,
) -> jax.Array:
    """Samples from a discrete uniform distribution from 0 to num_timesteps"""
    sampled_ints = jax.random.randint(
        prng, shape=shape, minval=0, maxval=num_timesteps, dtype=dtype
    )
    return jnp.array(sampled_ints, dtype=dtype)


def sample_gaussian_noise(
    prng: jax.Array,
    shape: Tuple[int, ...],
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jax.Array, jax.Array]:
    """Samples Gaussian noise from a standard normal distribution"""
    prng, _ = jax.random.split(prng)
    noise = jax.random.normal(prng, shape, dtype=dtype)
    return prng, noise


def sample_timesteps(
    prng: jax.Array,
    shape: Tuple[int, ...],
    distribution: str,
    dtype: jnp.dtype = jnp.float32,
    **kwargs: Any,
) -> Tuple[jax.Array, jax.Array]:
    """Samples timesteps from a uniform distribution"""
    if distribution == "complementary_beta":
        res = sample_complementary_beta(prng, shape, **kwargs, dtype=dtype)
    elif distribution == "uniform":
        res = sample_uniform(prng, shape, **kwargs, dtype=dtype)
    elif distribution == "discrete_uniform":
        res = sample_discrete_uniform(prng, shape, **kwargs, dtype=dtype)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")

    prng, _ = jax.random.split(prng)
    return prng, res

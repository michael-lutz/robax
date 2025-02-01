"""Image utils."""

from typing import Tuple

import jax
import numpy as np
from scipy.ndimage import zoom


def interpolate_images(images: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    """
    Interpolates a batch of images from shape [..., x, y, 3] to [..., new_x, new_y, 3].

    Args:
        images (np.ndarray): A numpy array of images with shape [..., x, y, 3].
        new_shape (Tuple[int, int]): The new shape to interpolate to.

    Returns:
        np.ndarray: A numpy array of images with shape [..., new_x, new_y, 3].
    """
    assert len(images.shape) >= 3, "Images must have at least 3 dimensions"

    zoom_factors = (
        *[1.0] * (len(images.shape) - 3),
        new_shape[0] / images.shape[-3],
        new_shape[1] / images.shape[-2],
        1.0,
    )

    interpolated_images = zoom(images, zoom_factors, order=1)  # Bilinear interpolation (order=1)
    return interpolated_images


def interpolate_image_jax(image: jax.Array, new_shape: Tuple[int, int]) -> jax.Array:
    """Interpolate an image using JAX.

    Args:
        image (jax.Array): A JAX array of images with shape [..., x, y, 3].
        new_shape (Tuple[int, int]): The new shape to interpolate to.

    Returns:
        jax.Array: A JAX array of images with shape [..., new_x, new_y, 3].
    """
    assert len(image.shape) >= 3, "Images must have at least 3 dimensions"

    def resize_image(image: jax.Array) -> jax.Array:
        resized_image: jax.Array = jax.image.resize(image, (*new_shape, 3), method="bilinear")
        return resized_image.astype(np.uint8)

    if len(image.shape) == 3:
        return resize_image(image)
    else:
        flattened_batch_size = np.prod(image.shape[:-3])
        image_flattened_batch = jax.lax.reshape(image, (flattened_batch_size, *image.shape[-3:]))

        interpolated_images = jax.vmap(resize_image)(image_flattened_batch)

        reshaped_interpolated_images = jax.lax.reshape(
            interpolated_images, (*image.shape[:-3], *new_shape, 3)
        )
        return reshaped_interpolated_images

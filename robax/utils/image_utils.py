"""Image utils."""

from typing import Tuple

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

"""Implements base image model API"""

from abc import ABC, abstractmethod
from typing import Any

import jax


class BaseImageModel(ABC):
    """Base image model API"""

    @abstractmethod
    def __call__(self, image: jax.Array, train: bool = False) -> Any:
        pass

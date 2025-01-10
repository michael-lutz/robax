"""Implements base image model API"""

import jax
from abc import ABC, abstractmethod
from typing import Any


class BaseImageModel(ABC):
    """Base image model API"""

    @abstractmethod
    def __call__(self, image: jax.Array, train: bool = False) -> Any:
        pass

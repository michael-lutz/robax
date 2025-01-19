"""Base inference step."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import flax.linen as nn
import jax

from robax.utils.observation import Observation


class BaseInferenceStep(ABC):
    """Base inference step."""

    @abstractmethod
    def generate_action(
        self,
        prng_key: jax.Array,
        params: Dict[str, Any],
        model: nn.Module,
        observation: Observation,
    ) -> Tuple[jax.Array, jax.Array]:
        """Generate an action from the policy."""
        pass

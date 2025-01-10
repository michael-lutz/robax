"""Implements base llm model API"""

from typing import Any
from abc import ABC, abstractmethod


class BaseLlmModel(ABC):
    """Base llm model API"""

    @abstractmethod
    def __call__(
        self,
        tokens,
        *,
        embedded_prefix=None,
        embed_only=False,
        pre_logits=None,
        positions=None,
        mask=None,
        decode=False,
        deterministic=True,
    ) -> Any:
        pass

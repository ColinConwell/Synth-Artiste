from __future__ import annotations

import abc
from typing import Protocol

from ..config import ImageConfig


class ImageBackend(Protocol):
    name: str

    async def generate(self, prompt: str, config: ImageConfig) -> bytes:
        ...


class AbstractImageBackend(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    async def generate(self, prompt: str, config: ImageConfig) -> bytes:  # pragma: no cover
        raise NotImplementedError



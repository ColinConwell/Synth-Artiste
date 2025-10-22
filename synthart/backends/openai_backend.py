from __future__ import annotations

from ..config import ImageConfig
from ..openai_image import generate_with_retries
from .base import AbstractImageBackend


class OpenAIImageBackend(AbstractImageBackend):
    name = "openai"

    async def generate(self, prompt: str, config: ImageConfig) -> bytes:
        return await generate_with_retries(prompt, config)



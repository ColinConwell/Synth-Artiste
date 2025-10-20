import asyncio
import math
from typing import Optional

from openai import AsyncOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIError

from .config import ImageConfig
from .utils import decode_base64_image


_client: Optional[AsyncOpenAI] = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


async def generate_image_bytes(prompt: str, config: ImageConfig) -> bytes:
    client = get_client()
    kwargs = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "size": config.size,
        "quality": config.quality,
        "timeout": config.request_timeout,
    }
    if config.background:
        kwargs["background"] = config.background

    resp = await client.images.generate(**kwargs)
    b64 = resp.data[0].b64_json
    return decode_base64_image(b64)


async def generate_with_retries(prompt: str, config: ImageConfig) -> bytes:
    delay = config.retry_backoff_base
    last_exc: Optional[Exception] = None
    for attempt in range(config.retries + 1):
        try:
            return await generate_image_bytes(prompt, config)
        except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as exc:
            last_exc = exc
            if attempt >= config.retries:
                break
            await asyncio.sleep(delay)
            delay *= 2.0
        except Exception as exc:  # pragma: no cover - unexpected
            last_exc = exc
            if attempt >= config.retries:
                break
            await asyncio.sleep(delay)
            delay *= 2.0
    assert last_exc is not None
    raise last_exc



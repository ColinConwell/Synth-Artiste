from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image

from openai import AsyncOpenAI


@dataclass
class TextEmbedConfig:
    model: str = "text-embedding-3-large"


async def embed_texts(texts: Iterable[str], model: str = TextEmbedConfig.model) -> np.ndarray:
    client = AsyncOpenAI()
    items = list(texts)
    # OpenAI recommends batching; here we do a single request for simplicity
    resp = await client.embeddings.create(model=model, input=items)
    vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    return np.stack(vecs, axis=0)


class CLIPImageEmbedder:
    def __init__(self, device: str | None = None) -> None:
        from transformers import CLIPModel, CLIPProcessor  # lazy import

        from .images import device_name
        self.device = device_name(device)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()

    @torch.inference_mode()
    def embed_images(self, image_paths: Iterable[Path]) -> np.ndarray:
        from .images import as_image
        images = [as_image(p) for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        image_features = self.model.get_image_features(**inputs)
        # Transformers 5 returns projected features inside a ModelOutput.
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.detach().cpu().numpy().astype(np.float32)



"""Shared image and device conventions; paths and PIL images work everywhere."""
from pathlib import Path
from PIL import Image, ImageOps


def as_image(value):
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB")
    with Image.open(Path(value)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def device_name(device=None):
    if device:
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def release_memory():
    """Call after deleting a large model when moving between notebook sections."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

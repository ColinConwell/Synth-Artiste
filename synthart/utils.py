import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


def slugify(value: str, max_length: int = 60) -> str:
    """Create filesystem-safe slug."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9\-\_\s]", "", value)
    value = re.sub(r"[\s\-]+", "-", value)
    return value[:max_length].strip("-") or "item"


def now_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def decode_base64_image(b64: str) -> bytes:
    return base64.b64decode(b64)


def write_bytes(filepath: Path, data: bytes) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(data)



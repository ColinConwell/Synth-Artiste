from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ImageConfig:
    """Configuration for image generation and orchestration."""

    # OpenAI image params
    size: str = "1024x1024"
    quality: str = "high"  # "low" | "medium" | "high" | "auto"
    background: Optional[str] = None  # e.g., "transparent" for PNGs with alpha

    # Orchestration params
    concurrency: int = 5
    retries: int = 4
    retry_backoff_base: float = 0.8  # seconds, exponential backoff
    request_timeout: float = 60.0  # seconds

    # Output
    output_dir: Path = Path("outputs")
    run_name: Optional[str] = None  # if set, use this exact run directory name under output_dir

    def resolved_output_dir(self) -> Path:
        return self.output_dir.resolve()



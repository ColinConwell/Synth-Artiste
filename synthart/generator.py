import asyncio
from pathlib import Path
from typing import Iterable, List, Tuple

from .artist import SyntheticArtist
from .config import ImageConfig
from .openai_image import generate_with_retries
from .utils import ensure_dir, now_run_id, slugify, write_bytes


async def _bounded_generate_and_save(
    sem: asyncio.Semaphore,
    artist: SyntheticArtist,
    content_idx: int,
    content_prompt: str,
    run_dir: Path,
    config: ImageConfig,
) -> Tuple[Path, str]:
    async with sem:
        prompt = artist.build_image_prompt(content_prompt)
        image_bytes = await generate_with_retries(prompt, config)
        artist_slug = slugify(artist.name)
        content_slug = slugify(content_prompt, max_length=80)
        filename = f"{content_idx:02d}-{content_slug}.png"
        out_path = run_dir / artist_slug / filename
        write_bytes(out_path, image_bytes)
        return out_path, content_prompt


async def generate_dataset(
    artists: Iterable[SyntheticArtist],
    content_prompts: Iterable[str],
    config: ImageConfig,
) -> List[Tuple[Path, str]]:
    """Generate images for all (artist x content) combinations and save to disk.

    Returns list of (output_path, content_prompt) for all completed generations.
    """
    run_dir = config.resolved_output_dir() / f"run-{now_run_id()}"
    ensure_dir(run_dir)

    artists = list(artists)
    content_prompts = list(content_prompts)

    sem = asyncio.Semaphore(config.concurrency)
    tasks: List[asyncio.Task] = []

    for artist in artists:
        artist_dir = run_dir / slugify(artist.name)
        ensure_dir(artist_dir)
        for idx, content in enumerate(content_prompts, start=1):
            tasks.append(
                asyncio.create_task(
                    _bounded_generate_and_save(sem, artist, idx, content, run_dir, config)
                )
            )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    output: List[Tuple[Path, str]] = []
    for res in results:
        if isinstance(res, Exception):
            # Surface the first error loudly; in practice, you might collect/report.
            raise res
        output.append(res)
    return output



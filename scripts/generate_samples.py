import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthart.artist import SyntheticArtist
from synthart.config import ImageConfig
from synthart.generator import generate_dataset


def load_env() -> None:
    # Prefer .env.local if present, else fallback to .env
    cwd = Path(__file__).resolve().parent.parent
    env_local = cwd / ".env.local"
    env_default = cwd / ".env"
    if env_local.exists():
        load_dotenv(dotenv_path=env_local)
    elif env_default.exists():
        load_dotenv(dotenv_path=env_default)
    else:
        load_dotenv()


def build_artists() -> list[SyntheticArtist]:
    return [
        SyntheticArtist(
            name="Neon Noir Minimalist",
            system_prompt=(
                "A visionary who paints urban nightscapes with restrained geometry,"
                " high-contrast neon accents, deep shadows, and negative space."
            ),
            style_tags=["noir", "neon", "minimal", "urban", "geometric"],
        ),
        SyntheticArtist(
            name="Analog Botanical Surrealist",
            system_prompt=(
                "A surreal botanist who composes dreamlike film-photography scenes,"
                " soft grain, shallow depth of field, muted palettes, and organic textures."
            ),
            style_tags=["film", "botanical", "surreal", "muted", "grain"],
        ),
    ]


def build_content_prompts() -> list[str]:
    return [
        "A lone telephone booth under a flickering streetlight",
        "An empty crosswalk with rain-slick reflections",
        "A rooftop with satellite dishes and a glowing skyline",
        "A cyclist passing by an alley of neon signs",
        "A vintage sedan parked beside a shuttered diner",
        "A wilted rose encased in melting ice",
        "A fern unfurling inside a glass terrarium",
        "A moth hovering over a glowing bulb",
        "A cluster of mushrooms in morning haze",
        "A bouquet suspended in zero gravity",
    ]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a matched artist sample set")
    parser.add_argument("--num-per-artist", type=int, default=10)
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="high")
    args = parser.parse_args()
    if args.num_per_artist < 1:
        parser.error("--num-per-artist must be positive")
    load_env()
    # Verify API key presence early for clearer errors
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")

    artists = build_artists()
    content_prompts = build_content_prompts()[:args.num_per_artist]

    config = ImageConfig(
        size="1024x1024",
        quality=args.quality,
        concurrency=6,
        retries=4,
        request_timeout=120.0,
        output_dir=ROOT / "outputs",
    )

    results = await generate_dataset(artists=artists, content_prompts=content_prompts, config=config)

    # Simple report
    for path, content in results:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    asyncio.run(main())



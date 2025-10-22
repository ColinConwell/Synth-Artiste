import argparse
import asyncio
import json
import multiprocessing
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Fix for macOS mutex lock error with asyncio + multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthart.artist import SyntheticArtist
from synthart.config import ImageConfig
from synthart.backends.openai_backend import OpenAIImageBackend
from synthart.embeddings import embed_texts, CLIPImageEmbedder
from synthart.sampler import select_prompts_by_embeddings
from synthart.generator import generate_dataset
from synthart.evaluator import diversity_stats, write_manifest
from synthart.utils import now_run_id, ensure_dir


def load_env() -> None:
    env_local = ROOT / ".env.local"
    env_default = ROOT / ".env"
    if env_local.exists():
        load_dotenv(dotenv_path=env_local)
    elif env_default.exists():
        load_dotenv(dotenv_path=env_default)
    else:
        load_dotenv()


def default_artists() -> List[SyntheticArtist]:
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


def candidate_prompts() -> List[str]:
    return [
        "A lone telephone booth under a flickering streetlight",
        "An empty crosswalk with rain-slick reflections",
        "A rooftop with satellite dishes and a glowing skyline",
        "A cyclist passing by an alley of neon signs",
        "A vintage sedan parked beside a shuttered diner",
    ]


async def plan_prompts(num_per_artist: int) -> List[str]:
    prompts = candidate_prompts()
    try:
        X = await embed_texts(prompts)
        selected = select_prompts_by_embeddings(prompts, X, num_per_artist)
    except Exception:
        # Fallback: simple head selection if embeddings are unavailable
        selected = prompts[:num_per_artist]
    return selected


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_per_artist", type=int, default=3)
    parser.add_argument("--backend", type=str, default="openai", choices=["openai"])
    parser.add_argument("--plan_only", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    load_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")

    artists = default_artists()
    selected_prompts = await plan_prompts(args.num_per_artist)
    print(f"Planning selected {len(selected_prompts)} prompts per artist...")

    if args.plan_only:
        print(json.dumps({"selected_prompts": selected_prompts}, indent=2))
        return

    run_name = args.run_name or f"run-{now_run_id()}"
    config = ImageConfig(
        size="1024x1024",
        quality="high",
        concurrency=3,
        retries=2,
        output_dir=ROOT / "test_outputs",
        run_name=run_name
    )
    run_dir = ROOT / "test_outputs" / run_name
    ensure_dir(run_dir)
    
    print("Starting generation...")
    results = await generate_dataset(artists=artists, content_prompts=selected_prompts, config=config)

    print(f"Generated {len(results)} images")
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(main())


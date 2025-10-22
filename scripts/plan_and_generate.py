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
    # Set PyTorch threading to avoid mutex errors on macOS
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parent.parent
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
        "A wilted rose encased in melting ice",
        "A fern unfurling inside a glass terrarium",
        "A moth hovering over a glowing bulb",
        "A cluster of mushrooms in morning haze",
        "A bouquet suspended in zero gravity",
        "A deserted train platform with a single bench",
        "A lighthouse cutting through coastal fog",
        "An abandoned greenhouse overrun with vines",
        "A mirrored room filled with floating lilies",
        "A glass of water refracting sunlight onto leaves",
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
    # Set event loop policy for macOS compatibility
    if sys.platform == "darwin":
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        except ImportError:
            # Fall back to asyncio's default but with selector event loop
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_per_artist", type=int, default=10)
    parser.add_argument("--backend", type=str, default="openai", choices=["openai"])  # future: diffusers
    parser.add_argument("--plan_only", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--skip_clip", action="store_true", help="Skip CLIP evaluation (useful on macOS with threading issues)")
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
    config = ImageConfig(size="1024x1024", quality="high", concurrency=6, retries=4, output_dir=ROOT / "outputs", run_name=run_name)
    run_dir = (ROOT / "outputs" / run_name)
    ensure_dir(run_dir)
    # Early pointer and pre-manifest stub so the run dir is visible even if generation fails
    with open(ROOT / "outputs" / "last_run.txt", "w") as f:
        f.write(str(run_dir))
    pre_manifest = {
        "backend": args.backend,
        "selected_prompts": selected_prompts,
        "config": {
            "size": config.size,
            "quality": config.quality,
            "concurrency": config.concurrency,
        },
        "status": "planning_complete",
    }
    write_manifest(run_dir / "manifest.json", pre_manifest)

    # Currently only OpenAI backend is implemented; the generator uses it implicitly
    print("Starting generation...")
    results = await generate_dataset(artists=artists, content_prompts=selected_prompts, config=config)

    # Prepare manifest and attempt CLIP embeddings; always write manifest
    image_paths = [p for (p, _c) in results]
    metrics = None
    metrics_error = None
    
    if not args.skip_clip:
        # Run CLIP evaluation in a subprocess to avoid macOS threading issues
        print("Evaluating diversity with CLIP embeddings...")
        try:
            import subprocess
            import tempfile
            
            # Create a temporary script to run CLIP in isolation
            clip_script = f'''
import sys
from pathlib import Path
sys.path.insert(0, "{ROOT}")

from synthart.embeddings import CLIPImageEmbedder
from synthart.evaluator import diversity_stats
import json

image_paths = {[str(p) for p in image_paths]}
clip = CLIPImageEmbedder()
img_X = clip.embed_images([Path(p) for p in image_paths])
metrics = diversity_stats(img_X)
print(json.dumps(metrics))
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(clip_script)
                temp_script = f.name
            
            try:
                result = subprocess.run(
                    [sys.executable, temp_script],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    metrics = json.loads(result.stdout.strip())
                    print(f"CLIP evaluation successful")
                else:
                    metrics_error = f"CLIP subprocess error (use --skip_clip to disable)"
            finally:
                Path(temp_script).unlink(missing_ok=True)
        except Exception as e:
            metrics_error = f"{e.__class__.__name__}: {e}"
    else:
        print("Skipping CLIP evaluation (--skip_clip flag set)")

    # Write manifest inside the run directory (outputs/run-YYYYMMDD-HHMMSS)
    manifest_path = run_dir / "manifest.json"
    print(f"Writing manifest to: {manifest_path}")
    payload = {
        "backend": args.backend,
        "selected_prompts": selected_prompts,
        "images": [str(p) for p in image_paths],
        "metrics": metrics,
        "metrics_error": metrics_error,
        "config": {
            "size": config.size,
            "quality": config.quality,
            "concurrency": config.concurrency,
        },
    }
    write_manifest(manifest_path, payload)
    print(json.dumps({"manifest": str(manifest_path), "metrics": metrics, "metrics_error": metrics_error}, indent=2))
    # Write last_run pointer
    with open(ROOT / "outputs" / "last_run.txt", "w") as f:
        f.write(str(run_dir))


if __name__ == "__main__":
    asyncio.run(main())



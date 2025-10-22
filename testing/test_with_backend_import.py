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


async def test():
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set")
        return
    
    # Step 1: Plan prompts with embeddings
    print("Step 1: Planning prompts...")
    prompts = ["A test image 1", "A test image 2", "A test image 3", "A test image 4"]
    X = await embed_texts(prompts)
    selected = select_prompts_by_embeddings(prompts, X, 2)
    print(f"Selected prompts: {selected}")
    
    # Step 2: Create artists
    print("Step 2: Creating artists...")
    artists = [
        SyntheticArtist(
            name="Test Artist",
            system_prompt="A test artist",
            style_tags=["test"],
        )
    ]
    
    # Step 3: Generate dataset
    print("Step 3: Generating dataset...")
    config = ImageConfig(
        size="1024x1024",
        quality="high",
        concurrency=2,
        retries=2,
        output_dir=ROOT / "test_outputs",
        run_name="test-backend-import"
    )
    
    results = await generate_dataset(artists, selected, config)
    
    print(f"Generated {len(results)} images")
    print("Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test())


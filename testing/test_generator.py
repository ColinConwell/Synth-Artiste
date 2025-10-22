import asyncio
import multiprocessing
import sys
import os
from pathlib import Path

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

async def test_generator():
    from dotenv import load_dotenv
    from synthart.artist import SyntheticArtist
    from synthart.config import ImageConfig
    from synthart.generator import generate_dataset
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set")
        return
    
    print("Creating test artist...")
    artists = [
        SyntheticArtist(
            name="Test Artist",
            system_prompt="A test artist",
            style_tags=["test"],
        )
    ]
    
    content_prompts = ["A test image 1", "A test image 2"]
    
    config = ImageConfig(
        size="1024x1024",
        quality="high",
        concurrency=2,
        retries=2,
        output_dir=ROOT / "test_outputs",
        run_name="test-run"
    )
    
    print("Starting generation...")
    results = await generate_dataset(artists, content_prompts, config)
    
    print(f"Generated {len(results)} images")
    for path, prompt in results:
        print(f"  - {path}")
    
    print("Test completed")

if __name__ == "__main__":
    asyncio.run(test_generator())


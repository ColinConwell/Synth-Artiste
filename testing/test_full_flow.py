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

async def test_full_flow():
    from dotenv import load_dotenv
    from synthart.artist import SyntheticArtist
    from synthart.config import ImageConfig
    from synthart.embeddings import embed_texts
    from synthart.sampler import select_prompts_by_embeddings
    from synthart.generator import generate_dataset
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set")
        return
    
    # Step 1: Plan prompts with embeddings
    print("Step 1: Planning prompts...")
    prompts = [
        "A test image 1",
        "A test image 2",
        "A test image 3",
        "A test image 4",
    ]
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
        run_name="test-full-flow"
    )
    
    results = await generate_dataset(artists, selected, config)
    
    print(f"Generated {len(results)} images")
    for path, prompt in results:
        print(f"  - {path}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_full_flow())


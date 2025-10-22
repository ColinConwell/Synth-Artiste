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

async def test_with_embeddings():
    from dotenv import load_dotenv
    from synthart.embeddings import embed_texts
    from openai import AsyncOpenAI
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set")
        return
    
    # Test embeddings first
    print("Testing embeddings...")
    texts = ["test 1", "test 2", "test 3"]
    X = await embed_texts(texts)
    print(f"Embeddings created: shape {X.shape}")
    
    # Now test image generation after embeddings
    print("Testing image generation after embeddings...")
    client = AsyncOpenAI()
    resp = await client.images.generate(
        model="gpt-image-1",
        prompt="Test image",
        size="1024x1024",
        quality="high",
        timeout=60.0
    )
    print(f"Image generation successful: {resp.data[0].b64_json[:50]}...")
    
    print("Test completed")

if __name__ == "__main__":
    asyncio.run(test_with_embeddings())


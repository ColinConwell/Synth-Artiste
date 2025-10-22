import asyncio
import multiprocessing
import sys
import os
from pathlib import Path

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

async def test_image_generation():
    from openai import AsyncOpenAI
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set")
        return
    
    client = AsyncOpenAI()
    print("Client created successfully")
    
    # Try generating an image
    try:
        print("Generating image...")
        resp = await client.images.generate(
            model="gpt-image-1",
            prompt="A simple test image of a red circle",
            size="1024x1024",
            quality="high",
            timeout=60.0
        )
        print(f"Image generation successful: {resp.data[0].b64_json[:50]}...")
    except Exception as e:
        print(f"Image generation error: {type(e).__name__}: {e}")
    
    print("Test completed")

if __name__ == "__main__":
    asyncio.run(test_image_generation())


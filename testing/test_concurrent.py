import asyncio
import multiprocessing
import sys
import os

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

async def generate_one(idx: int):
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI()
    print(f"Task {idx}: Generating...")
    resp = await client.images.generate(
        model="gpt-image-1",
        prompt=f"Test image {idx}",
        size="1024x1024",
        quality="high",
        timeout=60.0
    )
    print(f"Task {idx}: Complete")
    return idx, resp.data[0].b64_json[:50]

async def test_concurrent():
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set")
        return
    
    print("Starting concurrent image generation...")
    
    # Test with 3 concurrent tasks
    sem = asyncio.Semaphore(3)
    
    async def bounded_task(idx):
        async with sem:
            return await generate_one(idx)
    
    tasks = [asyncio.create_task(bounded_task(i)) for i in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for res in results:
        if isinstance(res, Exception):
            print(f"Error: {type(res).__name__}: {res}")
        else:
            print(f"Success: Task {res[0]}")
    
    print("Test completed")

if __name__ == "__main__":
    asyncio.run(test_concurrent())


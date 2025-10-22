import asyncio
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

async def test_openai():
    from openai import AsyncOpenAI
    from dotenv import load_dotenv
    
    load_dotenv()
    
    client = AsyncOpenAI()
    print("Client created successfully")
    
    # Try a simple embedding call
    try:
        resp = await client.embeddings.create(
            model="text-embedding-3-large",
            input=["test"]
        )
        print(f"Embedding call successful: {len(resp.data)} embeddings")
    except Exception as e:
        print(f"Embedding error: {e}")
    
    print("Test completed")

if __name__ == "__main__":
    asyncio.run(test_openai())


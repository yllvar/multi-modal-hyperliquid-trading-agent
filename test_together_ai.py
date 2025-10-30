"""
Test script to verify Together.ai API key and connection.
"""
import os
import asyncio
from dotenv import load_dotenv
import aiohttp

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
TOGETHER_AI_API_KEY = os.getenv('TOGETHER_AI_API_KEY')

async def test_together_ai():
    """Test the Together.ai API connection."""
    if not TOGETHER_AI_API_KEY:
        print("❌ Error: TOGETHER_AI_API_KEY not found in environment variables")
        print("Please add it to your .env file or set it as an environment variable")
        return

    url = "https://api.together.xyz/api/inference"
    headers = {
        "Authorization": f"Bearer {TOGETHER_AI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "prompt": "Hello, world! This is a test of the Together.ai API. ",
        "max_tokens": 10,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": ["</s>"],
        "echo": False
    }

    try:
        async with aiohttp.ClientSession() as session:
            print("🔍 Testing Together.ai API connection...")
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    print("✅ Successfully connected to Together.ai API!")
                    print("\nResponse:")
                    print(f"Model: {result.get('model')}")
                    print(f"Output: {result.get('output', {}).get('choices', [{}])[0].get('text', 'No text generated')}")
                else:
                    error = await response.text()
                    print(f"❌ Error: {response.status} - {error}")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    print("🔑 Found API Key:", "*" * 20 + TOGETHER_AI_API_KEY[-4:] if TOGETHER_AI_API_KEY else "Not found")
    asyncio.run(test_together_ai())

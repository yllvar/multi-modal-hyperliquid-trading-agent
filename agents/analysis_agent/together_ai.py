"""
Together.ai integration for the AI Trading System.

This module provides a client for interacting with the Together.ai API for
natural language processing and other AI tasks.
"""

import os
import json
import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
import aiohttp
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Then import config
try:
    from config import config
except ImportError:
    # Fallback if config module is not available
    config = {}

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types."""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"

@dataclass
class AIMessage:
    """A message in a conversation with the AI."""
    role: str  # 'system', 'user', or 'assistant'
    content: str
    name: Optional[str] = None

class TogetherAIClient:
    """Client for interacting with the Together.ai API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the Together.ai client.
        
        Args:
            api_key: Together.ai API key. If not provided, will be read from config.
            base_url: Base URL for the Together.ai API.
        """
        # Update the default model to use a serverless model
        self.api_key = api_key or config.get('together_ai.api_key') or os.getenv('TOGETHER_AI_API_KEY')
        self.base_url = base_url or config.get('together_ai.base_url', 'https://api.together.xyz')
        self.timeout = config.get('together_ai.timeout', 30.0)
        self.max_retries = config.get('together_ai.max_retries', 3)
        self.retry_delay = config.get('together_ai.retry_delay', 1.0)
        self.default_model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'  # Updated to a serverless model
        
        if not self.api_key:
            raise ValueError("Together.ai API key is required")
        
        self._session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp ClientSession."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    'Authorization': f"Bearer {self.api_key}",
                    'Content-Type': 'application/json'
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session
    
    async def close(self):
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Any:
        """Make an HTTP request to the Together.ai API."""
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        session = await self._get_session()
        
        for attempt in range(self.max_retries + 1):
            try:
                async with session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers={
                        'Authorization': f"Bearer {self.api_key}",
                        'Content-Type': 'application/json'
                    }
                ) as response:
                    if response.status == 429:
                        retry_after = float(response.headers.get('Retry-After', self.retry_delay))
                        logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                        await asyncio.sleep(retry_after)
                        continue
                        
                    response.raise_for_status()
                    
                    if stream:
                        return response
                    
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        return await response.json()
                    return await response.text()
                    
            except aiohttp.ClientError as e:
                if attempt == self.max_retries:
                    logger.error(f"API request failed after {self.max_retries} attempts: {e}")
                    raise
                
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                await asyncio.sleep(self.retry_delay * (attempt + 1))
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Generate a chat completion using Together.ai.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: The model to use. If not provided, uses the default model.
            temperature: Controls randomness (0.0 to 1.0).
            max_tokens: Maximum number of tokens to generate.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            repetition_penalty: Penalty for repeating tokens.
            stop: List of strings that will stop generation when encountered.
            stream: Whether to stream the response.
            
        Returns:
            The completion response or an async generator for streaming.
        """
        data = {
            'model': model or self.default_model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'top_k': top_k,
            'repetition_penalty': repetition_penalty,
            'stream': stream
        }
        
        if stop:
            data['stop'] = stop
        
        if stream:
            return self._stream_response('chat/completions', data)
            
        response = await self._request('POST', 'chat/completions', data=data)
        return response
    
    async def _stream_response(self, endpoint: str, data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming responses from the API."""
        data['stream'] = True
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/{endpoint}",
                json=data,
                headers={
                    'Authorization': f"Bearer {self.api_key}",
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                
                async for line in response.content:
                    if line.startswith(b'data: '):
                        chunk = line[6:].strip()
                        if chunk == b'[DONE]':
                            break
                            
                        try:
                            data = json.loads(chunk)
                            yield data
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse chunk: {e}")
                            continue
    
    async def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings for the input texts.
        
        Args:
            texts: A single text or a list of texts to embed.
            model: The embedding model to use.
            
        Returns:
            A list of embedding vectors.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        data = {
            'model': model or 'togethercomputer/m2-bert-80M-8k-retrieval',
            'inputs': texts
        }
        
        response = await self._request('POST', 'embeddings', data=data)
        return [item['embedding'] for item in response['data']]
    
    async def analyze_sentiment(
        self,
        text: str,
        context: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze the sentiment of a given text.
        
        Args:
            text: The text to analyze.
            context: Additional context for the analysis.
            model: The model to use for sentiment analysis.
            
        Returns:
            A dictionary containing sentiment analysis results.
        """
        context_str = f"Context: {context}\n" if context else ""
        prompt = (
            "Analyze the sentiment of the following text and provide a score from -1 (very negative) "
            "to 1 (very positive), along with a confidence score and key phrases.\n\n"
            f"{context_str}"
            f"Text: {text}\n\n"
            "Respond in JSON format with the following structure:\n"
            "{\n"
            '    "sentiment_score": float,\n'
            '    "confidence": float,\n'
            '    "key_phrases": List[str],\n'
            '    "explanation": str\n'
            '}'
        )
        
        messages = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant that analyzes sentiment of financial texts.'
            },
            {
                'role': 'user',
                'content': prompt.strip()
            }
        ]
        
        try:
            response = await self.chat_completion(
                messages=messages,
                model=model or self.default_model,
                temperature=0.2,
                max_tokens=500
            )
            
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            if not content:
                raise ValueError("Empty response from model")
                
            # Extract JSON from the response
            if '```json' in content:
                json_str = content.strip().split('```json')[1].split('```')[0].strip()
            else:
                # Try to parse the entire content as JSON if no code block is found
                json_str = content.strip()
                
            return json.loads(json_str)
            
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse sentiment analysis response: {e}")
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'key_phrases': [],
                'explanation': f'Failed to analyze sentiment: {str(e)}'
            }

def get_together_ai() -> 'TogetherAIClient':
    """Get the global TogetherAIClient instance with lazy initialization."""
    global _together_ai_instance
    if _together_ai_instance is None:
        _together_ai_instance = TogetherAIClient()
    return _together_ai_instance

# Initialize on first use
_together_ai_instance = None

# For backward compatibility
try:
    together_ai = get_together_ai()
except ValueError as e:
    # Don't fail on import, only when actually used
    together_ai = None

# Example usage:
# result = await together_ai.analyze_sentiment("The market is looking bullish today!")
# print(result['sentiment_score'])  # e.g., 0.8

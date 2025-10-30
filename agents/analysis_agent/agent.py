"""
Analysis Agent implementation for the AI Trading System.

This module provides the AnalysisAgent class that handles sentiment analysis,
news analysis, and other AI-powered analysis tasks using Together.ai.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import json

from ..base_agent import BaseAgent, Message, MessageType
from .together_ai import TogetherAIClient, ModelType

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Container for analysis results."""
    symbol: str
    timestamp: datetime
    sentiment_score: float
    confidence: float
    key_phrases: List[str]
    explanation: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the analysis result to a dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'sentiment_score': self.sentiment_score,
            'confidence': self.confidence,
            'key_phrases': self.key_phrases,
            'explanation': self.explanation,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create an AnalysisResult instance from a dictionary."""
        return cls(
            symbol=data['symbol'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            sentiment_score=data['sentiment_score'],
            confidence=data['confidence'],
            key_phrases=data.get('key_phrases', []),
            explanation=data['explanation'],
            metadata=data.get('metadata')
        )

class AnalysisAgent(BaseAgent):
    """Agent responsible for performing analysis using AI models."""
    
    def __init__(
        self,
        agent_id: str = 'analysis_agent',
        model_name: str = 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        temperature: float = 0.3,
        max_tokens: int = 1000,
        api_key: str = None,
        **kwargs
    ):
        """Initialize the AnalysisAgent.
        
        Args:
            agent_id: Unique identifier for the agent
            model_name: Name of the model to use for analysis
            temperature: Temperature for model sampling
            max_tokens: Maximum number of tokens to generate
            api_key: API key for the AI service
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(agent_id=agent_id, **kwargs)
        self.model = model_name  # Using model_name for consistency with the rest of the code
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Initialize the AI client with the provided API key
        self.ai_client = TogetherAIClient(api_key=api_key)
        self._analysis_cache = {}
    
    async def start(self):
        """Start the agent and subscribe to relevant message types."""
        await super().start()
        # Subscribe to messages this agent cares about
        self.message_bus.subscribe(
            self.agent_id,
            [
                MessageType.ANALYZE_NEWS,
                MessageType.ANALYZE_SENTIMENT,
                MessageType.GENERATE_INDICATORS,
                MessageType.PROCESS_MARKET_DATA,
                MessageType.MARKET_DATA  # Add subscription to MARKET_DATA
            ]
        )
        logger.info(f"{self.agent_id} started and subscribed to messages")
    
    async def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data and generate insights.
        
        Args:
            market_data: Dictionary containing market data with keys like 'symbol', 'price', 'volume', etc.
            
        Returns:
            Dict containing analysis results
        """
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')
            logger.info(f"Analyzing market data for {symbol}")
            
            # Generate a simple analysis based on price and volume
            price = float(market_data.get('price', 0))
            volume = float(market_data.get('volume', 0))
            
            # Simple trend analysis
            price_change = price - float(market_data.get('open', price))
            price_change_pct = (price_change / float(market_data.get('open', price))) * 100 if price > 0 else 0
            
            # Volume analysis
            avg_volume = float(market_data.get('average_volume', volume))
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            
            # Generate a simple sentiment based on price and volume
            if price_change > 0 and volume_ratio > 1.5:
                sentiment = "bullish"
                confidence = min(0.9, 0.6 + (volume_ratio * 0.1))
            elif price_change < 0 and volume_ratio > 1.5:
                sentiment = "bearish"
                confidence = min(0.9, 0.6 + (volume_ratio * 0.1))
            else:
                sentiment = "neutral"
                confidence = 0.7
            
            # Generate key insights
            insights = []
            if abs(price_change_pct) > 2.0:
                direction = "up" if price_change > 0 else "down"
                insights.append(f"Significant price movement: {abs(price_change_pct):.2f}% {direction}")
            
            if volume_ratio > 2.0:
                insights.append(f"High trading volume: {volume_ratio:.1f}x average")
            
            if 'rsi' in market_data.get('indicators', {}):
                rsi = market_data['indicators']['rsi']
                if rsi > 70:
                    insights.append("RSI indicates overbought conditions")
                elif rsi < 30:
                    insights.append("RSI indicates oversold conditions")
            
            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'price': price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'volume': volume,
                'volume_ratio': volume_ratio,
                'sentiment': sentiment,
                'confidence': confidence,
                'insights': insights,
                'indicators': market_data.get('indicators', {})
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market data: {e}", exc_info=True)
            return {
                'error': str(e),
                'symbol': market_data.get('symbol', 'UNKNOWN'),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def handle_message(self, message: Message):
        """Process incoming messages."""
        try:
            if message.msg_type == MessageType.MARKET_DATA:
                # Process market data
                market_data = message.payload
                if market_data:
                    result = await self.analyze_market_data(market_data)
                    # Send the analysis result back to the sender
                    await self.send_message(
                        Message(
                            msg_type=MessageType.ANALYSIS_RESULT,
                            sender=self.agent_id,
                            recipients=[message.sender],  # Send back to the original sender
                            payload=result
                        )
                    )
            
            elif message.msg_type == MessageType.ANALYZE_NEWS:
                result = await self.analyze_news(
                    text=message.payload.get('text'),
                    symbol=message.payload.get('symbol'),
                    context=message.payload.get('context')
                )
                await self._send_analysis_result(
                    result,
                    message.sender,
                    message.msg_id
                )
                
            elif message.msg_type == MessageType.ANALYZE_SENTIMENT:
                result = await self.analyze_sentiment(
                    text=message.payload.get('text'),
                    symbol=message.payload.get('symbol'),
                    context=message.payload.get('context')
                )
                await self._send_analysis_result(
                    result,
                    message.sender,
                    message.msg_id
                )
                
            elif message.msg_type == MessageType.PROCESS_MARKET_DATA:
                market_data = message.payload.get('data')
                if market_data:
                    result = await self.process_market_data(market_data)
                    await self._send_analysis_result(
                        result,
                        message.sender,
                        message.msg_id
                    )
        
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await self._send_error(
                str(e),
                message.sender,
                message.msg_id
            )
    
    async def _send_analysis_result(
        self,
        result: AnalysisResult,
        recipient_id: str,
        original_message_id: str = None
    ):
        """Send analysis result to the recipient."""
        await self.send_message(
            Message(
                sender_id=self.agent_id,
                recipient_id=recipient_id,
                message_type=MessageType.ANALYSIS_RESULT,
                payload=result.to_dict(),
                in_reply_to=original_message_id
            )
        )
    
    async def _send_error(
        self,
        error: str,
        recipient_id: str,
        original_message_id: str = None
    ):
        """Send an error message to the recipient."""
        await self.send_message(
            Message(
                sender_id=self.agent_id,
                recipient_id=recipient_id,
                message_type=MessageType.ERROR,
                payload={'error': error},
                in_reply_to=original_message_id
            )
        )
    
    async def analyze_sentiment(
        self,
        text: str,
        symbol: str = None,
        context: str = None,
        model: str = None
    ) -> AnalysisResult:
        """Analyze the sentiment of the given text.
        
        Args:
            text: The text to analyze
            symbol: Optional symbol this text is related to
            context: Additional context for the analysis
            model: Model to use for analysis (overrides default)
            
        Returns:
            AnalysisResult containing sentiment analysis
        """
        try:
            result = await self.ai_client.analyze_sentiment(
                text=text,
                context=context,
                model=model or self.model
            )
            
            return AnalysisResult(
                symbol=symbol or 'N/A',
                timestamp=datetime.utcnow(),
                sentiment_score=result.get('sentiment_score', 0.0),
                confidence=result.get('confidence', 0.0),
                key_phrases=result.get('key_phrases', []),
                explanation=result.get('explanation', ''),
                metadata={
                    'model': model or self.model,
                    'input_length': len(text)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}", exc_info=True)
            # Return a neutral result on error
            return AnalysisResult(
                symbol=symbol or 'N/A',
                timestamp=datetime.utcnow(),
                sentiment_score=0.0,
                confidence=0.0,
                key_phrases=[],
                explanation=f"Error during analysis: {str(e)}",
                metadata={'error': True}
            )
    
    async def analyze_news(
        self,
        text: str,
        symbol: str = None,
        context: str = None,
        model: str = None
    ) -> AnalysisResult:
        """Analyze news text for trading signals.
        
        Args:
            text: News article or text to analyze
            symbol: Optional symbol this news is about
            context: Additional context for the analysis
            model: Model to use for analysis (overrides default)
            
        Returns:
            AnalysisResult containing news analysis
        """
        try:
            # First get sentiment
            sentiment = await self.analyze_sentiment(text, symbol, context, model)
            
            # Then get additional insights
            prompt_parts = [
                "Analyze the following financial news and provide insights:",
                "",
                *([f"Symbol: {symbol}"] if symbol else []),
                *([f"Context: {context}"] if context else []),
                f"News: {text}",
                "",
                "Provide analysis in this JSON format:",
                '{',
                '    "impact": "high/medium/low",',
                '    "potential_effect": "bullish/bearish/neutral",',
                '    "key_points": ["list", "of", "key", "points"],',
                '    "trading_implications": "brief explanation of trading implications"',
                '}'
            ]
            prompt = "\n".join(prompt_parts)
            
            response = await self.ai_client.chat_completion(
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a financial news analyst. Analyze the following news and provide trading insights.'
                    },
                    {
                        'role': 'user',
                        'content': prompt.strip()
                    }
                ],
                model=model or self.model,
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse the response
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            insights = json.loads(content)
            
            # Combine with sentiment analysis
            return AnalysisResult(
                symbol=symbol or 'N/A',
                timestamp=datetime.utcnow(),
                sentiment_score=sentiment.sentiment_score,
                confidence=sentiment.confidence,
                key_phrases=sentiment.key_phrases + insights.get('key_points', []),
                explanation=(
                    f"Sentiment: {sentiment.explanation}\n"
                    f"Impact: {insights.get('impact', 'unknown')}\n"
                    f"Potential Effect: {insights.get('potential_effect', 'neutral')}\n"
                    f"Trading Implications: {insights.get('trading_implications', 'None provided')}"
                ),
                metadata={
                    'model': model or self.model,
                    'impact': insights.get('impact'),
                    'potential_effect': insights.get('potential_effect'),
                    'input_length': len(text)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in news analysis: {e}", exc_info=True)
            # Fall back to just sentiment analysis if detailed analysis fails
            return await self.analyze_sentiment(text, symbol, context, model)
    
    async def process_market_data(
        self,
        market_data: Dict[str, Any],
        model: str = None
    ) -> AnalysisResult:
        """Process market data and generate analysis.
        
        Args:
            market_data: Dictionary containing market data
            model: Model to use for analysis (overrides default)
            
        Returns:
            AnalysisResult containing market analysis
        """
        try:
            # Convert market data to a readable format
            data_str = json.dumps(market_data, indent=2)
            
            prompt = f"""
            Analyze the following market data and provide trading insights:
            
            {data_str}
            
            Provide analysis in this JSON format:
            {{
                "market_condition": "trending/bullish/bearish/range-bound/volatile",
                "key_levels": {{
                    "support": [list, of, support, levels],
                    "resistance": [list, of, resistance, levels]
                }},
                "momentum": "strong/weak/none",
                "volume_analysis": "high/medium/low volume",
                "trading_implications": "brief explanation of trading implications"
            }}
            """
            
            response = await self.ai_client.chat_completion(
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a technical market analyst. Analyze the following market data and provide trading insights.'
                    },
                    {
                        'role': 'user',
                        'content': prompt.strip()
                    }
                ],
                model=model or self.model,
                temperature=0.2,
                max_tokens=500
            )
            
            # Parse the response
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            analysis = json.loads(content)
            
            return AnalysisResult(
                symbol=market_data.get('symbol', 'N/A'),
                timestamp=datetime.utcnow(),
                sentiment_score=0.0,  # Neutral for technical analysis
                confidence=0.9,  # High confidence for technical analysis
                key_phrases=[
                    analysis.get('market_condition', ''),
                    analysis.get('momentum', ''),
                    analysis.get('volume_analysis', '')
                ],
                explanation=(
                    f"Market Condition: {analysis.get('market_condition', 'N/A')}\n"
                    f"Momentum: {analysis.get('momentum', 'N/A')}\n"
                    f"Volume Analysis: {analysis.get('volume_analysis', 'N/A')}\n"
                    f"Trading Implications: {analysis.get('trading_implications', 'None provided')}"
                ),
                metadata={
                    'model': model or self.model,
                    'market_condition': analysis.get('market_condition'),
                    'key_levels': analysis.get('key_levels', {}),
                    'momentum': analysis.get('momentum')
                }
            )
            
        except Exception as e:
            logger.error(f"Error in market data analysis: {e}", exc_info=True)
            return AnalysisResult(
                symbol=market_data.get('symbol', 'N/A'),
                timestamp=datetime.utcnow(),
                sentiment_score=0.0,
                confidence=0.0,
                key_phrases=[],
                explanation=f"Error during market data analysis: {str(e)}",
                metadata={'error': True}
            )
    
    async def generate_indicators(
        self,
        market_data: Dict[str, Any],
        indicator_config: Dict[str, Any],
        model: str = None
    ) -> Dict[str, Any]:
        """Generate technical indicators based on market data.
        
        Args:
            market_data: Dictionary containing market data
            indicator_config: Configuration for indicators to generate
            model: Model to use for analysis (overrides default)
            
        Returns:
            Dictionary containing generated indicators
        """
        try:
            # Convert market data to a readable format
            data_str = json.dumps(market_data, indent=2)
            config_str = json.dumps(indicator_config, indent=2)
            
            prompt = f"""
            Generate technical indicators based on the following market data and configuration:
            
            Market Data:
            {data_str}
            
            Indicator Configuration:
            {config_str}
            
            Provide the indicators in this JSON format:
            {{
                "indicators": {{
                    "indicator_name": {{
                        "value": value,
                        "signal": "buy/sell/neutral",
                        "confidence": 0.0-1.0,
                        "metadata": {{}}
                    }}
                }},
                "composite_signal": "strong_buy/buy/neutral/sell/strong_sell",
                "confidence": 0.0-1.0,
                "explanation": "brief explanation of the indicators and signals"
            }}
            """
            
            response = await self.ai_client.chat_completion(
                messages=[
                    {
                        'role': 'system',
                        'content': (
                            'You are a technical analysis expert. Generate technical indicators '
                            'based on the provided market data and configuration.'
                        )
                    },
                    {
                        'role': 'user',
                        'content': prompt.strip()
                    }
                ],
                model=model or self.model,
                temperature=0.1,  # Lower temperature for more consistent results
                max_tokens=1000
            )
            
            # Parse the response
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            result = json.loads(content)
            
            # Add metadata
            result['metadata'] = {
                'model': model or self.model,
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': market_data.get('symbol', 'N/A')
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating indicators: {e}", exc_info=True)
            return {
                'error': str(e),
                'indicators': {},
                'composite_signal': 'neutral',
                'confidence': 0.0,
                'explanation': 'Error generating indicators: ' + str(e).replace('\\', '\\\\')
            }

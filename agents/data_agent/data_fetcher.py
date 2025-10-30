"""
Hyperliquid Data Fetcher for the Data Agent.

This module provides functionality to fetch and process market data from Hyperliquid.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
import aiohttp
import pandas as pd
from config import config

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Container for Hyperliquid market data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the market data to a dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'funding_rate': self.funding_rate,
            'open_interest': self.open_interest,
            'metadata': self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create a MarketData instance from a dictionary."""
        return cls(
            symbol=data['symbol'],
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) 
                     else datetime.fromtimestamp(data['timestamp'] / 1000, tz=timezone.utc),
            open=float(data['open']),
            high=float(data['high']),
            low=float(data['low']),
            close=float(data['close']),
            volume=float(data.get('volume', 0)),
            funding_rate=float(data.get('funding_rate', 0)),
            open_interest=float(data.get('open_interest', 0)) if data.get('open_interest') is not None else None,
            metadata=data.get('metadata', {})
        )

class DataFetcher:
    """Fetches market data from Hyperliquid exchange."""
    
    def __init__(self, base_url: str = None, ws_url: str = None, message_bus = None):
        """Initialize the DataFetcher for Hyperliquid.
        
        Args:
            base_url: Base URL for the REST API (default: from config)
            ws_url: WebSocket URL for real-time data (default: from config)
            message_bus: Message bus for inter-agent communication
        """
        self.base_url = base_url or "https://api.hyperliquid.xyz"
        self.ws_url = ws_url or "wss://api.hyperliquid.xyz/ws"
        self.message_bus = message_bus
        self.session = None
        self.ws = None
        self._initialized = False
        self._subscriptions = set()
        
        # Cache for market data
        self.market_data_cache = {}
        self.last_update = {}
        self._meta_cache = None
    
    async def start(self):
        """Initialize the HTTP session and WebSocket connection."""
        if not self._initialized:
            self.session = aiohttp.ClientSession()
            self._initialized = True
            logger.info("Hyperliquid DataFetcher started")
    
    async def stop(self):
        """Close all connections and clean up resources."""
        if self._initialized:
            if self.ws and not self.ws.closed:
                await self.ws.close()
            if self.session and not self.session.closed:
                await self.session.close()
            self._initialized = False
            logger.info("Hyperliquid DataFetcher stopped")
    
    async def close(self):
        """Alias for stop() for context manager compatibility."""
        await self.stop()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _make_request(self, endpoint: str, method: str = 'GET', params: dict = None, data: dict = None) -> dict:
        """Make an HTTP request to the Hyperliquid API."""
        url = f"{self.base_url}{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=10
            ) as response:
                response.raise_for_status()
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {str(e)}")
            raise

    async def _get_meta(self) -> dict:
        """Get and cache the Hyperliquid market metadata."""
        if not self._meta_cache:
            endpoint = "/info"
            params = {'type': 'meta'}
            self._meta_cache = await self._make_request(endpoint, method='POST', data=params)
        return self._meta_cache

    async def fetch_available_symbols(self) -> List[str]:
        """Fetch all available trading symbols from Hyperliquid.
        
        Returns:
            List of available trading symbols in uppercase (e.g., ['BTC', 'ETH', 'SOL'])
        """
        await self.start()
        
        try:
            # Hyperliquid API requires a POST request with a specific payload
            params = {
                'type': 'meta'
            }
            
            data = await self._make_request("/info", method='POST', data=params)
            
            # The response should have a 'universe' key with asset information
            if isinstance(data, dict) and 'universe' in data:
                # Extract symbol names from the universe data and convert to uppercase
                symbols = [asset['name'].upper() for asset in data['universe'] if 'name' in asset]
                return sorted(list(set(symbols)))  # Ensure unique and sorted symbols
            
            logger.warning("Unexpected response format when fetching available symbols")
            return []
            
        except Exception as e:
            logger.error(f"Error fetching available symbols: {str(e)}", exc_info=True)
            raise
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = '1h',
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[MarketData]:
        """Fetch OHLCV (Open, High, Low, Close, Volume) data from Hyperliquid."""
        await self.start()
        
        try:
            # Hyperliquid API expects the request in this format
            params = {
                'type': 'candleSnapshot',
                'req': {
                    'coin': symbol.upper(),
                    'interval': interval,
                    'startTime': start_time or (int(time.time() * 1000) - (limit * 60 * 60 * 1000)),
                    'endTime': end_time or int(time.time() * 1000)
                }
            }
            
            logger.info(f"Sending request to Hyperliquid API with params: {params}")
            data = await self._make_request("/info", method='POST', data=params)
            logger.info(f"Received response from Hyperliquid API. Type: {type(data)}, Content: {data}")

            # Convert to MarketData objects
            candles = []

            # Handle both list and dict responses
            if isinstance(data, list):
                logger.info("Response is a list")
                candle_data = data
            elif isinstance(data, dict):
                logger.info(f"Response is a dict with keys: {data.keys()}")
                candle_data = data.get('candles', [])
            else:
                logger.warning(f"Unexpected response type: {type(data)}")
                candle_data = []
                
            logger.info(f"Processing {len(candle_data)} candles")
                
            for i, candle in enumerate(candle_data):
                try:
                    logger.debug(f"Processing candle {i}: {candle}")
                    # Try to handle both dict and list formats
                    if isinstance(candle, dict):
                        # Handle dictionary format
                        candles.append(MarketData(
                            symbol=symbol.upper(),
                            timestamp=datetime.fromtimestamp(candle.get('t', candle.get('time', 0)) / 1000, tz=timezone.utc),
                            open=float(candle.get('o', candle.get('open', 0))),
                            high=float(candle.get('h', candle.get('high', 0))),
                            low=float(candle.get('l', candle.get('low', 0))),
                            close=float(candle.get('c', candle.get('close', 0))),
                            volume=float(candle.get('v', candle.get('volume', 0))),
                            metadata={
                                'interval': interval,
                                'taker_buy_volume': candle.get('tbv', candle.get('taker_buy_volume')),
                                'number_of_trades': candle.get('n', candle.get('number_of_trades'))
                            }
                        ))
                    else:
                        # Handle list format
                        candles.append(MarketData(
                            symbol=symbol.upper(),
                            timestamp=datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc),
                            open=float(candle[1]),
                            high=float(candle[2]),
                            low=float(candle[3]),
                            close=float(candle[4]),
                            volume=float(candle[5]),
                            metadata={
                                'interval': interval,
                                'taker_buy_volume': float(candle[6]) if len(candle) > 6 else None,
                                'number_of_trades': int(candle[7]) if len(candle) > 7 else None
                            }
                        ))
                except (IndexError, ValueError, KeyError) as e:
                    logger.warning(f"Error parsing candle data at index {i}: {e}. Candle data: {candle}")
                    continue
            
            logger.info(f"Successfully parsed {len(candles)} candles")
            return candles[-limit:]  # Return only the requested number of candles
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {str(e)}", exc_info=True)
            raise

    async def stream_market_data(
        self,
        symbols: List[str],
        on_data: callable = None
    ) -> None:  # Changed return type to None since we're using a callback
        """Stream real-time market data from Hyperliquid."""
        await self.start()
        
        # Normalize symbols
        symbols = [s.upper() for s in symbols]
        
        # If no WebSocket connection exists, create one
        if self.ws is None or self.ws.closed:
            self.ws = await self.session.ws_connect(self.ws_url)
        
        try:
            # Subscribe to the order book and ticker streams for each symbol
            for symbol in symbols:
                # Subscribe to order book updates
                await self.ws.send_json({
                    'method': 'subscribe',
                    'subscription': {
                        'type': 'l2Book',
                        'coin': symbol
                    }
                })
                
                # Subscribe to ticker updates
                await self.ws.send_json({
                    'method': 'subscribe',
                    'subscription': {
                        'type': 'ticker',
                        'coin': symbol
                    }
                })
                
                self._subscriptions.add(symbol)
            
            logger.info(f"Subscribed to {len(symbols)} symbols")
            
            # Process incoming messages
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        
                        # Process different message types
                        if 'channel' in data:
                            if data['channel'] == 'ticker':
                                market_data = self._parse_ticker_data(data)
                                if market_data and on_data:
                                    await on_data(market_data)
                                    
                            elif data['channel'] == 'l2book':
                                # Handle order book updates
                                pass
                                
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse WebSocket message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self.ws.exception()}")
                    break
                
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning("WebSocket connection closed")
                    break
                    
        except Exception as e:
            logger.error(f"Error in market data stream: {str(e)}")
            raise
        finally:
            # Unsubscribe from all symbols
            await self._unsubscribe_all()
    
    def _parse_ticker_data(self, data: dict) -> Optional[MarketData]:
        """Parse ticker data from WebSocket message."""
        try:
            ticker = data.get('data', {})
            if not ticker:
                return None
                
            symbol = ticker.get('symbol', '').upper()
            if not symbol:
                return None
            
            return MarketData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(ticker.get('time', 0) / 1000, tz=timezone.utc),
                open=float(ticker.get('open', 0)),
                high=float(ticker.get('high', 0)),
                low=float(ticker.get('low', 0)),
                close=float(ticker.get('close', 0)),
                volume=float(ticker.get('volume', 0)),
                funding_rate=float(ticker.get('fundingRate', 0)),
                open_interest=float(ticker.get('openInterest', 0)),
                metadata={
                    'bid': float(ticker.get('bidPrice', 0)),
                    'ask': float(ticker.get('askPrice', 0)),
                    'bid_size': float(ticker.get('bidQty', 0)),
                    'ask_size': float(ticker.get('askQty', 0))
                }
            )
            
        except Exception as e:
            logger.error(f"Error parsing ticker data: {e}")
            return None
    
    async def _unsubscribe_all(self):
        """Unsubscribe from all active subscriptions."""
        if not self.ws or self.ws.closed:
            return
            
        try:
            for symbol in list(self._subscriptions):
                # Unsubscribe from order book
                await self.ws.send_json({
                    'method': 'unsubscribe',
                    'subscription': {
                        'type': 'l2Book',
                        'coin': symbol
                    }
                })
                
                # Unsubscribe from ticker
                await self.ws.send_json({
                    'method': 'unsubscribe',
                    'subscription': {
                        'type': 'ticker',
                        'coin': symbol
                    }
                })
                
                self._subscriptions.remove(symbol)
                
            logger.info("Unsubscribed from all data streams")
            
        except Exception as e:
            logger.error(f"Error unsubscribing from data streams: {e}")
    
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, list]:
        """Fetch the order book for a symbol from Hyperliquid.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC')
            limit: Maximum number of bids/asks to return (max: 50)
            
        Returns:
            Dictionary containing 'bids' and 'asks' lists
        """
        await self.start()
        
        try:
            # First get the meta data to find the asset index
            meta = await self._get_meta()
            asset_idx = next((i for i, asset in enumerate(meta['universe']) 
                           if asset['name'] == symbol.upper()), None)
            
            if asset_idx is None:
                raise ValueError(f"Asset {symbol} not found in Hyperliquid universe")
            
            # Get order book using the asset index
            endpoint = "/info"
            params = {
                'type': 'l2Book',
                'coin': symbol.upper()
            }
            
            data = await self._make_request(endpoint, method='POST', data=params)
            
            # Process the order book data
            bids = [[float(price), float(size)] for price, size in data.get('bids', [])[:limit]]
            asks = [[float(price), float(size)] for price, size in data.get('asks', [])[:limit]]
            
            return {
                'bids': bids,
                'asks': asks,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbol': symbol.upper()
            }
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {str(e)}")
            raise
    
    async def fetch_trades(
        self,
        symbol: str,
        limit: int = 100,
        from_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fetch recent trades for a symbol from Hyperliquid.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC')
            limit: Maximum number of trades to return (max: 1000)
            from_id: Trade ID to fetch from (not currently supported)
            
        Returns:
            List of trade dictionaries
        """
        await self.start()
        
        try:
            # First get the meta data to find the asset index
            meta = await self._get_meta()
            asset_idx = next((i for i, asset in enumerate(meta['universe']) 
                           if asset['name'] == symbol.upper()), None)
            
            if asset_idx is None:
                raise ValueError(f"Asset {symbol} not found in Hyperliquid universe")
            
            # Get recent trades
            endpoint = "/info"
            params = {
                'type': 'trades',
                'coin': symbol.upper(),
                'limit': min(limit, 1000)
            }
            
            data = await self._make_request(endpoint, method='POST', data=params)
            
            # Process the trades
            trades = []
            for trade in data.get('trades', []):
                try:
                    trades.append({
                        'id': trade.get('tid'),
                        'price': float(trade.get('px', 0)),
                        'qty': float(trade.get('sz', 0)),
                        'quote_qty': float(trade.get('px', 0)) * float(trade.get('sz', 0)),
                        'time': datetime.fromtimestamp(trade.get('time', 0) / 1000, tz=timezone.utc),
                        'is_buyer_maker': trade.get('side', '').lower() == 'sell',
                        'side': trade.get('side', '').lower(),
                        'symbol': symbol.upper()
                    })
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing trade data: {e}")
                    continue
            
            return trades
            
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {str(e)}")
            raise

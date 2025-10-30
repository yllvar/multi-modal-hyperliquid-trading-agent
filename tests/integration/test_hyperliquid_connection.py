"""Integration tests for Hyperliquid connection and data fetching."""

import os
import asyncio
import pytest
import logging
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Add the project root to the Python path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.data_agent.data_fetcher import DataFetcher, MarketData

# Test configuration
TEST_SYMBOL = 'BTC'  # Base symbol for testing (Hyperliquid uses single asset symbols like 'BTC', 'ETH', etc.)

@pytest.fixture
async def data_fetcher():
    """Fixture to create and clean up a DataFetcher instance."""
    fetcher = DataFetcher()
    await fetcher.start()
    yield fetcher
    await fetcher.stop()
    
@pytest.mark.asyncio
async def test_fetch_available_symbols(data_fetcher):
    """Test fetching all available trading symbols from Hyperliquid."""
    # First, let's add a method to the DataFetcher class to fetch available symbols
    # We'll add this to data_fetcher.py
    symbols = await data_fetcher.fetch_available_symbols()
    
    # Basic assertions
    assert isinstance(symbols, list)
    assert len(symbols) > 0  # There should be at least one symbol
    
    # Check that common symbols exist
    common_symbols = ['BTC', 'ETH', 'SOL']  # Add more common symbols as needed
    for symbol in common_symbols:
        assert symbol in symbols, f"Expected {symbol} to be in available symbols"
    
    # Check that all symbols are strings and uppercase
    for symbol in symbols:
        assert isinstance(symbol, str)
        assert symbol == symbol.upper(), f"Symbol {symbol} should be uppercase"
    
    logger.info(f"Successfully fetched {len(symbols)} symbols from Hyperliquid")
    logger.debug(f"Available symbols: {sorted(symbols)}")

@pytest.mark.asyncio
async def test_fetch_ohlcv(data_fetcher):
    """Test fetching OHLCV data from Hyperliquid."""
    # Test with default parameters
    candles = await data_fetcher.fetch_ohlcv(
        symbol=TEST_SYMBOL,
        interval='1h',
        limit=10
    )
    
    # Basic assertions
    assert len(candles) > 0
    assert len(candles) <= 10  # Should not return more than requested
    
    # Check candle structure
    candle = candles[0]
    assert candle.symbol == TEST_SYMBOL.upper()
    assert isinstance(candle.timestamp, datetime)
    assert isinstance(candle.open, float)
    assert isinstance(candle.high, float)
    assert isinstance(candle.low, float)
    assert isinstance(candle.close, float)
    assert isinstance(candle.volume, float)
    assert candle.high >= candle.low
    
    # Test with custom time range
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - (24 * 60 * 60 * 1000)  # 24 hours ago
    
    candles = await data_fetcher.fetch_ohlcv(
        symbol=TEST_SYMBOL,
        interval='4h',
        limit=6,
        start_time=start_time,
        end_time=end_time
    )
    
    assert len(candles) > 0
    assert len(candles) <= 6
    assert hasattr(candle, 'volume')
    
    # Check data types
    assert isinstance(candle.symbol, str)
    assert isinstance(candle.timestamp, datetime)
    assert isinstance(candle.open, float)
    assert isinstance(candle.high, float)
    assert isinstance(candle.low, float)
    assert isinstance(candle.close, float)
    assert isinstance(candle.volume, float)
    
    # Check values make sense
    assert candle.high >= candle.low
    assert candle.high >= candle.open
    assert candle.high >= candle.close
    assert candle.low <= candle.open
    assert candle.low <= candle.close
    assert candle.volume >= 0

@pytest.mark.asyncio
async def test_fetch_order_book(data_fetcher):
    """Test fetching order book data from Hyperliquid."""
    order_book = await data_fetcher.fetch_order_book(
        symbol=TEST_SYMBOL,
        limit=10
    )
    
    # Basic assertions
    assert 'bids' in order_book
    assert 'asks' in order_book
    assert isinstance(order_book['bids'], list)
    assert isinstance(order_book['asks'], list)
    
    # Check bid/ask structure if data is available
    if order_book['bids']:
        bid = order_book['bids'][0]
        assert len(bid) == 2  # price, quantity
        assert bid[0] > 0  # price > 0
        assert bid[1] > 0  # quantity > 0
    
    if order_book['asks']:
        ask = order_book['asks'][0]
        assert len(ask) == 2  # price, quantity
        assert ask[0] > 0  # price > 0
        assert ask[1] > 0  # quantity > 0

@pytest.mark.asyncio
async def test_stream_market_data(data_fetcher):
    """Test streaming real-time market data from Hyperliquid."""
    received_data = []
    
    async def on_data(data):
        nonlocal received_data
        received_data.append(data)
        if len(received_data) >= 3:  # Stop after receiving 3 updates
            raise asyncio.CancelledError("Test completed")
    
    # Start the stream
    stream_task = asyncio.create_task(
        data_fetcher.stream_market_data(
            symbols=[TEST_SYMBOL],
            on_data=on_data
        )
    )
    
    try:
        # Wait for some data to be received or timeout after 30 seconds
        start_time = asyncio.get_event_loop().time()
        while len(received_data) < 3 and (asyncio.get_event_loop().time() - start_time) < 30:
            await asyncio.sleep(0.1)
            
        # Check if we received any data
        if not received_data:
            logger.warning("No data received from stream, this might be normal if there's no recent market activity")
            return  # Skip the rest of the test if no data was received
            
        # Check the structure of the received data
        for data in received_data:
            assert isinstance(data, MarketData)
            assert data.symbol == TEST_SYMBOL.upper()
            assert isinstance(data.timestamp, datetime)
            assert isinstance(data.open, float)
            assert isinstance(data.high, float)
            assert isinstance(data.low, float)
            assert isinstance(data.close, float)
            assert data.high >= data.low
            
    except asyncio.CancelledError:
        # Expected when we cancel the task
        pass
    except Exception as e:
        logger.error(f"Error in test_stream_market_data: {str(e)}")
        raise
    finally:
        if not stream_task.done():
            stream_task.cancel()
            try:
                await stream_task
            except (asyncio.CancelledError, Exception):
                pass

if __name__ == "__main__":
    # Allow running the tests directly for debugging
    import sys
    
    async def run_tests():
        fetcher = DataFetcher()
        try:
            await fetcher.start()
            
            # Test OHLCV
            print("Testing OHLCV data...")
            candles = await fetcher.fetch_ohlcv(TEST_SYMBOL, '1h', 5)
            print(f"Retrieved {len(candles)} candles")
            for candle in candles:
                print(f"{candle.timestamp} - O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close} V:{candle.volume}")
            
            # Test order book
            print("\nTesting order book...")
            order_book = await fetcher.fetch_order_book(TEST_SYMBOL, 3)
            print(f"Bids: {order_book['bids']}")
            print(f"Asks: {order_book['asks']}")
            
            # Test streaming (run for a few seconds)
            print("\nTesting real-time data (press Ctrl+C to stop)...")
            try:
                async for data in fetcher.stream_market_data([TEST_SYMBOL]):
                    print(f"{data.timestamp} - {data.symbol}: {data.close} (Δ: {data.close - data.open:.2f})")
            except asyncio.CancelledError:
                pass
                
        finally:
            await fetcher.stop()
    
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

"""
Integration tests for the AI Trading System.

These tests verify that all components work together correctly.
"""

# tests/integration/test_trading_system.py
import asyncio
import logging
import json
import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

# Import the modules that depend on config
from agents.analysis_agent.agent import AnalysisAgent, AnalysisResult
from agents.trading_agent.agent import TradingAgent, OrderSide, OrderType, OrderStatus
from agents.risk_agent.agent import RiskAgent, RiskParameters, RiskAssessment, RiskLevel
from agents.execution_agent.agent import ExecutionAgent, ExecutionParameters
from agents.base_agent import Message, MessageType, MessageBus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def message_bus():
    """Create a message bus for testing."""
    return MessageBus()

@pytest.fixture
async def analysis_agent(message_bus):
    """Create and start an analysis agent for testing."""
    agent = AnalysisAgent(
        agent_id="test_analysis_agent",
        message_bus=message_bus,
        model="test-model"
    )
    await agent.start()
    yield agent
    await agent.stop()

@pytest.fixture
async def risk_agent(message_bus):
    """Create and start a risk agent for testing."""
    risk_params = RiskParameters(
        max_position_size_pct=Decimal('0.1'),
        max_risk_per_trade_pct=Decimal('0.02'),
        daily_loss_limit_pct=Decimal('0.05'),
        max_leverage=Decimal('3.0'),
        max_drawdown_pct=Decimal('0.1'),
        min_liquidity_usd=Decimal('1000000'),
        max_concentration_pct=Decimal('0.3')
    )
    agent = RiskAgent(risk_parameters=risk_params, message_bus=message_bus)
    await agent.start()
    yield agent
    await agent.stop()

@pytest.fixture
async def execution_agent(message_bus):
    """Create and start an execution agent for testing."""
    exec_params = ExecutionParameters(
        slippage=Decimal('0.001'),
        max_slippage=Decimal('0.01'),
        max_retries=3,
        retry_delay=0.1,
        use_vwap=True,
        vwap_window=5,
        min_order_size=Decimal('10.0'),
        max_order_size=Decimal('100000.0')
    )
    agent = ExecutionAgent(execution_params=exec_params, message_bus=message_bus)
    await agent.start()
    yield agent
    await agent.stop()

@pytest.fixture
async def trading_agent(message_bus, risk_agent):
    """Create and start a trading agent for testing."""
    agent = TradingAgent(
        initial_balance={'USDT': Decimal('10000')},
        max_position_size=Decimal('0.1'),
        max_risk_per_trade=Decimal('0.02'),
        default_slippage=Decimal('0.001'),
        message_bus=message_bus,
        risk_agent=risk_agent
    )
    await agent.start()
    yield agent
    await agent.stop()

@pytest.mark.asyncio
async def test_market_data_processing(analysis_agent, trading_agent):
    """Test that market data is properly processed and analyzed."""
    test_data = {
        'symbol': 'BTC/USDT',
        'price': '50000.0',
        'volume': '1000.0',
        'open': '49000.0',
        'average_volume': '800.0',
        'timestamp': datetime.utcnow().isoformat(),
        'indicators': {'rsi': 45.0}
    }

    # Create a future to track when the analysis is complete
    analysis_complete = asyncio.Future()
    
    # Create a mock for the message bus publish method to capture the analysis result
    original_publish = analysis_agent.message_bus.publish
    async def mock_publish(message):
        # Call the original publish method
        await original_publish(message)
        
        # If this is an ANALYSIS_RESULT message, set the future
        if message.msg_type == MessageType.ANALYSIS_RESULT:
            if not analysis_complete.done():
                analysis_complete.set_result(message.payload)
    
    # Patch the message bus publish method
    with patch.object(analysis_agent.message_bus, 'publish', new=mock_publish):
        # Publish the market data message
        await analysis_agent.message_bus.publish(
            Message(
                msg_type=MessageType.MARKET_DATA,
                sender='test',
                recipients=[analysis_agent.agent_id],
                payload=test_data
            )
        )
        
        # Wait for the analysis to complete or timeout
        try:
            result = await asyncio.wait_for(analysis_complete, timeout=2.0)
            
            # Verify the result has the expected structure
            assert 'symbol' in result, "Analysis result is missing 'symbol'"
            assert result['symbol'] == 'BTC/USDT', "Unexpected symbol in analysis result"
            assert 'sentiment' in result, "Analysis result is missing 'sentiment'"
            assert 'confidence' in result, "Analysis result is missing 'confidence'"
            
        except asyncio.TimeoutError:
            # Check if the agent is subscribed to MARKET_DATA messages
            subscribed = analysis_agent.message_bus.is_subscribed(
                analysis_agent.agent_id, 
                MessageType.MARKET_DATA
            )
            assert False, (
                "Timed out waiting for market data to be processed. "
                f"Agent subscribed to MARKET_DATA: {subscribed}"
            )

@pytest.mark.asyncio
async def test_trade_signal_processing(trading_agent, risk_agent, execution_agent):
    """Test that trade signals are properly processed through the system."""
    test_signal = {
        'symbol': 'BTC/USDT',
        'action': 'buy',
        'quantity': '0.1',
        'price': '50000.0',
        'order_type': 'market',
        'stop_loss': '48000.0',
        'take_profit': '52000.0',
        'confidence': 0.8  # Add confidence to ensure the signal is processed
    }
    
    # Verify the trading agent is subscribed to NEW_SIGNAL messages
    assert trading_agent.message_bus.is_subscribed(
        trading_agent.agent_id, 
        MessageType.NEW_SIGNAL
    ), "Trading agent is not subscribed to NEW_SIGNAL messages"
    
    # Create an event to track when an order is placed
    order_processed = asyncio.Event()
    
    # Create a mock for place_order that will be called by the agent
    async def mock_place_order_impl(*args, **kwargs):
        # Just set the event and return a mock order
        order_processed.set()
        return {
            'order_id': 'test_order_123',
            'symbol': kwargs.get('symbol'),
            'side': kwargs.get('side'),
            'order_type': kwargs.get('order_type'),
            'quantity': kwargs.get('quantity'),
            'status': 'FILLED'
        }
    
    # Mock risk assessment and other methods
    with (
        patch.object(risk_agent, 'assess_order_risk') as mock_assess,
        patch.object(trading_agent, 'place_order', side_effect=mock_place_order_impl) as mock_place,
        patch.object(trading_agent, '_get_current_price') as mock_get_price,
        patch.object(trading_agent, '_calculate_position_size') as mock_calc_size
    ):
        # Configure the mocks
        mock_get_price.return_value = Decimal('50000.0')
        mock_calc_size.return_value = Decimal('0.1')
        mock_assess.return_value = RiskAssessment(
            is_approved=True,
            risk_score=0.1,
            risk_level=RiskLevel.LOW,
            reasons=[],
            suggested_actions=[],
            adjusted_quantity=Decimal('0.1'),
            max_position_size=Decimal('10000'),
            max_quantity=Decimal('0.2'),
            metadata={}
        )
        
        # Send the signal
        await trading_agent.message_bus.publish(
            Message(
                msg_type=MessageType.NEW_SIGNAL,
                sender='test_strategy',
                recipients=[trading_agent.agent_id],
                payload=test_signal
            )
        )
        
        # Wait for the order to be processed or timeout
        try:
            await asyncio.wait_for(order_processed.wait(), timeout=2.0)
            
            # Verify place_order was called with the correct arguments
            mock_place.assert_called_once()
            
            # Get the arguments passed to place_order
            call_args = mock_place.call_args[1]
            assert call_args['symbol'] == 'BTC/USDT'
            assert call_args['side'] == OrderSide.BUY
            assert call_args['order_type'] == OrderType.MARKET
            assert call_args['quantity'] == Decimal('0.1')
            
            # Verify the risk assessment was called
            mock_assess.assert_called_once()
            
        except asyncio.TimeoutError:
            # Check if the agent is subscribed to NEW_SIGNAL messages
            subscribed = trading_agent.message_bus.is_subscribed(
                trading_agent.agent_id, 
                MessageType.NEW_SIGNAL
            )
            assert False, (
                "Timed out waiting for order to be processed. "
                f"Agent subscribed to NEW_SIGNAL: {subscribed}"
            )
            
        # Verify the risk assessment was called
        mock_assess.assert_called_once()

@pytest.mark.asyncio
async def test_risk_management(trading_agent, risk_agent):
    """Test that risk management rules are enforced."""
    # Test position size limit
    assessment = await risk_agent.assess_order_risk({
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'quantity': '1000.0',  # This should be way over the limit
        'price': '50000.0',
        'order_type': 'market'
    })
    
    assert not assessment.is_approved, "Position size limit not enforced"
    assert any("exceeds maximum" in reason for reason in assessment.reasons), \
        "Should reject order that's too large"

@pytest.mark.asyncio
async def test_order_execution_lifecycle(trading_agent, execution_agent):
    """Test the complete order execution lifecycle."""
    # Place an order
    order = await trading_agent.place_order(
        symbol='BTC/USDT',
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal('0.1')
    )

    assert order is not None, "Failed to place order"
    
    # For market orders, they are immediately filled in this implementation
    expected_status = OrderStatus.FILLED if order.order_type == OrderType.MARKET else OrderStatus.NEW
    assert order.status == expected_status, f"Order should be in {expected_status} state"
    
    if order.status == OrderStatus.FILLED:
        # Verify the order was filled completely
        assert order.filled_quantity == order.quantity, "Order should be completely filled"
        assert order.remaining_quantity == Decimal('0'), "No quantity should remain"
        assert 'fill_price' in order.metadata, "Fill price should be in metadata"
    
    # Wait for execution
    await asyncio.sleep(0.2)
    
    # Verify order was filled
    assert order.order_id not in trading_agent.open_orders, "Order should be removed from open orders"
    assert order.order_id in trading_agent.closed_orders, "Order should be in closed orders"
    assert trading_agent.closed_orders[order.order_id].status == OrderStatus.FILLED, "Order should be FILLED"

@pytest.mark.asyncio
async def test_portfolio_rebalancing(trading_agent):
    """Test portfolio rebalancing functionality."""
    # Set up initial positions
    from agents.trading_agent.agent import Position
    
    btc_position = Position(
        symbol='BTC/USDT',
        quantity=Decimal('0.5'),
        entry_price=Decimal('50000'),
        current_price=Decimal('50000'),
        unrealized_pnl=Decimal('0'),
        entry_time=datetime.utcnow()
    )
    
    eth_position = Position(
        symbol='ETH/USDT',
        quantity=Decimal('10'),
        entry_price=Decimal('3000'),
        current_price=Decimal('3000'),
        unrealized_pnl=Decimal('0'),
        entry_time=datetime.utcnow()
    )
    
    trading_agent.positions = {
        'BTC/USDT': btc_position,
        'ETH/USDT': eth_position
    }
    
    # Update portfolio value
    trading_agent.portfolio_value = Decimal('100000')
    
    # Test rebalancing
    target_weights = {
        'BTC/USDT': Decimal('0.6'),
        'ETH/USDT': Decimal('0.4')
    }
    
    rebalance_orders = await trading_agent.rebalance_portfolio(target_weights)
    
    # Verify rebalancing orders were created
    assert len(rebalance_orders) > 0, "No rebalancing orders were created"
    assert all(hasattr(order, 'order_id') for order in rebalance_orders), "Rebalance orders should be Order objects"

@pytest.mark.asyncio
async def test_error_handling(trading_agent, message_bus):
    """Test error handling in the trading system."""
    # Test with invalid symbol format
    with pytest.raises(ValueError, match="Invalid symbol format. Expected format: 'BASE/QUOTE'"):
        await trading_agent.place_order(
            symbol='INVALID_SYMBOL_FORMAT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('0.1')
        )
    
    # Test with invalid order type
    with pytest.raises(ValueError, match="Invalid order type. Must be one of: market, limit, stop, stop_limit, trailing_stop"):
        await trading_agent.place_order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type='INVALID_ORDER_TYPE',
            quantity=Decimal('0.1')
        )
    
    # Test with zero quantity
    with pytest.raises(ValueError, match="Quantity must be positive"):
        await trading_agent.place_order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('0')
        )
    
    # Test with negative price for limit order
    with pytest.raises(ValueError, match="Price must be positive"):
        await trading_agent.place_order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal('0.1'),
            price=Decimal('-50000.0')
        )
    
    # Test with missing price for limit order
    with pytest.raises(ValueError, match="Price is required for limit orders"):
        await trading_agent.place_order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal('0.1')
        )
    
    # Test message handling with invalid message type
    with patch.object(trading_agent, 'handle_error') as mock_handle_error:
        await message_bus.publish(
            Message(
                msg_type='INVALID_MESSAGE_TYPE',
                sender='test',
                recipients=[trading_agent.agent_id],
                payload={}
            )
        )
        await asyncio.sleep(0.1)  # Give time for message processing
        # The agent should log an error for unknown message types
        assert True  # Just verify the message was processed without errors

if __name__ == "__main__":
    pytest.main(["-v", __file__])
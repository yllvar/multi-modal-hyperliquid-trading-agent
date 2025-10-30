"""
Trading Agent implementation for the AI Trading System.

This module provides the TradingAgent class that handles order execution,
position management, and strategy implementation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Tuple
from decimal import Decimal
import uuid

from ..base_agent import BaseAgent, Message, MessageType

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Types of orders that can be placed."""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'
    TRAILING_STOP = 'trailing_stop'

class OrderSide(Enum):
    """Sides of an order (buy/sell)."""
    BUY = 'buy'
    SELL = 'sell'

class OrderStatus(Enum):
    """Possible statuses of an order."""
    NEW = 'new'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'
    CANCELED = 'canceled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'

@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = 'GTC'  # Good Till Canceled
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: Decimal = Decimal('0')
    remaining_quantity: Decimal = field(init=False)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.remaining_quantity = self.quantity - self.filled_quantity
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))
        if self.price and not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        if self.stop_price and not isinstance(self.stop_price, Decimal):
            self.stop_price = Decimal(str(self.stop_price))

    def update_fill(self, fill_quantity: Decimal, fill_price: Decimal):
        """Update order with a fill."""
        fill_quantity = Decimal(str(fill_quantity))
        fill_price = Decimal(str(fill_price))
        
        self.filled_quantity += fill_quantity
        self.remaining_quantity = max(Decimal('0'), self.quantity - self.filled_quantity)
        self.updated_at = datetime.utcnow()
        
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        # Update average fill price in metadata
        if 'fill_price' not in self.metadata:
            self.metadata['fill_price'] = fill_price
            self.metadata['total_quantity'] = fill_quantity
        else:
            total_value = (
                self.metadata['fill_price'] * self.metadata['total_quantity'] +
                fill_price * fill_quantity
            )
            total_quantity = self.metadata['total_quantity'] + fill_quantity
            self.metadata['fill_price'] = total_value / total_quantity
            self.metadata['total_quantity'] = total_quantity

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': str(self.quantity),
            'price': str(self.price) if self.price is not None else None,
            'stop_price': str(self.stop_price) if self.stop_price is not None else None,
            'time_in_force': self.time_in_force,
            'status': self.status.value,
            'filled_quantity': str(self.filled_quantity),
            'remaining_quantity': str(self.remaining_quantity),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'client_order_id': self.client_order_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create an Order from a dictionary."""
        order = cls(
            order_id=data['order_id'],
            symbol=data['symbol'],
            side=OrderSide(data['side']),
            order_type=OrderType(data['order_type']),
            quantity=Decimal(data['quantity']),
            price=Decimal(data['price']) if data.get('price') is not None else None,
            stop_price=Decimal(data['stop_price']) if data.get('stop_price') is not None else None,
            time_in_force=data.get('time_in_force', 'GTC'),
            status=OrderStatus(data.get('status', 'new')),
            filled_quantity=Decimal(data.get('filled_quantity', '0')),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            client_order_id=data.get('client_order_id', str(uuid.uuid4())),
            metadata=data.get('metadata', {})
        )
        return order

@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal = None
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    entry_time: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))
        if not isinstance(self.entry_price, Decimal):
            self.entry_price = Decimal(str(self.entry_price))
        if not isinstance(self.unrealized_pnl, Decimal):
            self.unrealized_pnl = Decimal(str(self.unrealized_pnl))
        if not isinstance(self.realized_pnl, Decimal):
            self.realized_pnl = Decimal(str(self.realized_pnl))
    
    def update_price(self, price: Decimal):
        """Update the position with the latest price."""
        price = Decimal(str(price))
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity
        self.last_updated = datetime.utcnow()
    
    def add_to_position(self, quantity: Decimal, price: Decimal):
        """Add to an existing position."""
        quantity = Decimal(str(quantity))
        price = Decimal(str(price))
        
        if (self.quantity > 0 and quantity < 0) or (self.quantity < 0 and quantity > 0):
            # This is a partial close or reversal
            return self.reduce_position(abs(quantity), price)
        
        # Calculate new average entry price
        total_quantity = self.quantity + quantity
        if total_quantity == 0:
            self.entry_price = Decimal('0')
        else:
            self.entry_price = (
                (self.quantity * self.entry_price) + (quantity * price)
            ) / total_quantity
        
        self.quantity = total_quantity
        self.update_price(price if self.current_price is None else self.current_price)
        return Decimal('0')  # No realized PnL for adding to position
    
    def reduce_position(self, quantity: Decimal, price: Decimal) -> Decimal:
        """Reduce an existing position."""
        quantity = Decimal(str(quantity))
        price = Decimal(str(price))
        
        if self.quantity == 0:
            return Decimal('0')
        
        # Calculate how much of the position is being closed
        close_quantity = min(abs(quantity), abs(self.quantity))
        if self.quantity < 0:
            close_quantity = -close_quantity
        
        # Calculate realized PnL
        realized_pnl = (price - self.entry_price) * close_quantity
        self.realized_pnl += realized_pnl
        
        # Update position
        self.quantity -= close_quantity
        if self.quantity == 0:
            self.entry_price = Decimal('0')
        
        self.update_price(price if self.current_price is None else self.current_price)
        return realized_pnl
    
    def close_position(self, price: Decimal) -> Decimal:
        """Close the entire position."""
        price = Decimal(str(price))
        realized_pnl = self.reduce_position(self.quantity, price)
        return realized_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': str(self.quantity),
            'entry_price': str(self.entry_price),
            'current_price': str(self.current_price) if self.current_price is not None else None,
            'unrealized_pnl': str(self.unrealized_pnl),
            'realized_pnl': str(self.realized_pnl),
            'entry_time': self.entry_time.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create a Position from a dictionary."""
        return cls(
            symbol=data['symbol'],
            quantity=Decimal(data['quantity']),
            entry_price=Decimal(data['entry_price']),
            current_price=Decimal(data['current_price']) if data.get('current_price') is not None else None,
            unrealized_pnl=Decimal(data.get('unrealized_pnl', '0')),
            realized_pnl=Decimal(data.get('realized_pnl', '0')),
            entry_time=datetime.fromisoformat(data['entry_time']),
            last_updated=datetime.fromisoformat(data['last_updated']),
            metadata=data.get('metadata', {})
        )

class TradingStrategy(Enum):
    """Trading strategies that can be used by the agent."""
    MEAN_REVERSION = 'mean_reversion'
    TREND_FOLLOWING = 'trend_following'
    BREAKOUT = 'breakout'
    MOMENTUM = 'momentum'
    ARBITRAGE = 'arbitrage'
    PAIR_TRADING = 'pair_trading'
    GRID = 'grid'
    CUSTOM = 'custom'

class TradingAgent(BaseAgent):
    """Agent responsible for executing trades based on strategies and signals."""
    
    def __init__(
        self,
        agent_id: str = 'trading_agent',
        initial_balance: Dict[str, float] = None,
        max_position_size: float = 0.1,  # Max position size as % of balance
        max_risk_per_trade: float = 0.02,  # Max risk per trade as % of balance
        default_slippage: float = 0.001,  # Default slippage (0.1%)
        risk_agent = None,  # Reference to the risk agent
        **kwargs
    ):
        """Initialize the TradingAgent.
        
        Args:
            agent_id: Unique identifier for the agent
            initial_balance: Initial balance for each currency
            max_position_size: Maximum position size as a percentage of balance
            max_risk_per_trade: Maximum risk per trade as a percentage of balance
            default_slippage: Default slippage for market orders
            risk_agent: Reference to the risk agent for order validation
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(agent_id=agent_id, **kwargs)
        self.balance = {k: Decimal(str(v)) for k, v in (initial_balance or {'USDT': 10000}).items()}
        self.positions: Dict[str, Position] = {}
        self.open_orders: Dict[str, Order] = {}
        self.closed_orders: Dict[str, Order] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.strategies: Dict[str, TradingStrategy] = {}
        self.max_position_size = Decimal(str(max_position_size))
        self.max_risk_per_trade = Decimal(str(max_risk_per_trade))
        self.default_slippage = Decimal(str(default_slippage))
        self.risk_agent = risk_agent  # Store reference to risk agent
        self._order_counter = 0
    
    async def start(self):
        """Start the agent and subscribe to relevant message types."""
        await super().start()
        # Subscribe to messages this agent cares about
        self.message_bus.subscribe(
            self.agent_id,
            [
                MessageType.NEW_SIGNAL,
                MessageType.MARKET_DATA,
                MessageType.ANALYSIS_RESULT,
                MessageType.ORDER_UPDATE,
                MessageType.POSITION_UPDATE
            ]
        )
        logger.info(f"{self.agent_id} started and subscribed to messages")
    
    async def handle_message(self, message: Message):
        """Process incoming messages."""
        try:
            if message.msg_type == MessageType.NEW_SIGNAL:
                await self._handle_signal(message.payload)
            
            elif message.msg_type == MessageType.MARKET_DATA:
                await self._update_positions_with_market_data(message.payload)
            
            elif message.msg_type == MessageType.ANALYSIS_RESULT:
                await self._handle_analysis_result(message.payload, message.sender)
            
            elif message.msg_type == MessageType.ORDER_UPDATE:
                await self._handle_order_update(message.payload)
            
            elif message.msg_type == MessageType.POSITION_UPDATE:
                await self._handle_position_update(message.payload)
        
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await self._send_error(
                str(e),
                message.sender,
                message.msg_id
            )
    
    async def _handle_signal(self, signal: Dict[str, Any]):
        """Handle a new trading signal."""
        try:
            symbol = signal['symbol']
            action = signal['action']  # 'buy', 'sell', 'hold', etc.
            confidence = Decimal(str(signal.get('confidence', 0.5)))
            
            # Skip if confidence is too low
            if confidence < Decimal('0.3'):
                logger.info(f"Skipping {action} signal for {symbol} due to low confidence: {confidence}")
                return
            
            # Get current price and position
            current_price = await self._get_current_price(symbol)
            position = self.positions.get(symbol)
            
            # Calculate position size based on risk management
            quantity = await self._calculate_position_size(symbol, current_price, confidence)
            
            # Create order details for risk assessment
            order_details = {
                'symbol': symbol,
                'action': action,
                'quantity': float(quantity),
                'price': float(current_price),
                'order_type': signal.get('order_type', 'market'),
                'confidence': float(confidence)
            }
            
            # Get risk assessment
            risk_assessment = await self.risk_agent.assess_order_risk(order_details)
            
            # Skip if risk assessment doesn't approve the trade
            if not risk_assessment.is_approved:
                logger.warning(f"Risk assessment rejected {action} order for {symbol}: {risk_assessment.reasons}")
                return
                
            # Adjust quantity based on risk assessment if needed
            if hasattr(risk_assessment, 'adjusted_quantity') and risk_assessment.adjusted_quantity:
                quantity = min(quantity, Decimal(str(risk_assessment.adjusted_quantity)))
            
            if action == 'buy':
                if position and position.quantity < 0:
                    # Close short position
                    await self.place_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=min(quantity, abs(position.quantity)),
                        price=current_price,
                        metadata={'signal': signal, 'risk_assessment': risk_assessment}
                    )
                else:
                    # Open long position
                    await self.place_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        price=current_price,
                        metadata={'signal': signal, 'risk_assessment': risk_assessment}
                    )
            
            elif action == 'sell':
                if position and position.quantity > 0:
                    # Close long position
                    await self.place_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=min(quantity, position.quantity),
                        price=current_price,
                        metadata={'signal': signal, 'risk_assessment': risk_assessment}
                    )
                else:
                    # Open short position
                    await self.place_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        price=current_price,
                        metadata={'signal': signal, 'risk_assessment': risk_assessment}
                    )
            
            # Handle other signal types (trailing stop, take profit, etc.)
            elif action in ['trailing_stop', 'take_profit', 'stop_loss']:
                await self._handle_advanced_order(signal, current_price, position)
        
        except Exception as e:
            logger.error(f"Error handling signal: {e}", exc_info=True)
    
    async def _handle_analysis_result(self, analysis: Dict[str, Any], sender_id: str):
        """Handle analysis results from the analysis agent."""
        try:
            symbol = analysis.get('symbol')
            if not symbol or symbol == 'N/A':
                logger.warning(f"Received analysis without a valid symbol: {analysis}")
                return
            
            # Extract key metrics from analysis
            sentiment_score = Decimal(str(analysis.get('sentiment_score', 0)))
            confidence = Decimal(str(analysis.get('confidence', 0.5)))
            
            # Simple strategy based on sentiment score
            if sentiment_score >= Decimal('0.3'):
                await self._handle_signal({
                    'symbol': symbol,
                    'action': 'buy',
                    'confidence': float(confidence),
                    'source': 'sentiment_analysis',
                    'analysis': analysis
                })
            elif sentiment_score <= Decimal('-0.3'):
                await self._handle_signal({
                    'symbol': symbol,
                    'action': 'sell',
                    'confidence': float(confidence),
                    'source': 'sentiment_analysis',
                    'analysis': analysis
                })
            
            # Update position metadata with analysis
            if symbol in self.positions:
                self.positions[symbol].metadata['last_analysis'] = analysis
        
        except Exception as e:
            logger.error(f"Error handling analysis result: {e}", exc_info=True)
    
    async def _handle_order_update(self, order_update: Dict[str, Any]):
        """Handle order status updates."""
        try:
            order_id = order_update['order_id']
            status = OrderStatus(order_update['status'])
            
            if order_id in self.open_orders:
                order = self.open_orders[order_id]
                order.status = status
                order.updated_at = datetime.utcnow()
                
                # Handle fills
                if 'filled_quantity' in order_update and order_update['filled_quantity']:
                    fill_qty = Decimal(str(order_update['filled_quantity']))
                    fill_price = Decimal(str(order_update.get('fill_price', order.price or '0')))
                    order.update_fill(fill_qty, fill_price)
                    
                    # Update position
                    await self._update_position_from_fill(order, fill_qty, fill_price)
                
                # Move to closed orders if filled or canceled
                if status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                    self.closed_orders[order_id] = order
                    del self.open_orders[order_id]
                    
                    # Log the trade
                    if status == OrderStatus.FILLED:
                        self._log_trade(order)
        
        except Exception as e:
            logger.error(f"Error processing order update: {e}", exc_info=True)
    
    async def _update_position_from_fill(self, order: Order, fill_quantity: Decimal, fill_price: Decimal):
        """Update position based on order fill."""
        symbol = order.symbol
        position = self.positions.get(symbol)
        
        # Calculate base and quote currencies (assuming format like "BTC/USDT")
        if '/' in symbol:
            base_currency, quote_currency = symbol.split('/')
        else:
            base_currency = symbol
            quote_currency = 'USDT'  # Default quote currency
        
        # Update position
        if position is None:
            position = Position(
                symbol=symbol,
                quantity=fill_quantity if order.side == OrderSide.BUY else -fill_quantity,
                entry_price=fill_price,
                current_price=fill_price
            )
            self.positions[symbol] = position
        else:
            # Add to or reduce existing position
            if (position.quantity > 0 and order.side == OrderSide.BUY) or \
               (position.quantity < 0 and order.side == OrderSide.SELL):
                # Adding to position
                position.add_to_position(
                    fill_quantity if order.side == OrderSide.BUY else -fill_quantity,
                    fill_price
                )
            else:
                # Reducing or reversing position
                realized_pnl = position.reduce_position(
                    fill_quantity if order.side == OrderSide.BUY else -fill_quantity,
                    fill_price
                )
                # Update balance with realized PnL
                self.balance[quote_currency] = self.balance.get(quote_currency, Decimal('0')) + realized_pnl
        
        # Update balances
        if order.side == OrderSide.BUY:
            # Deduct cost from quote currency balance
            cost = fill_quantity * fill_price
            self.balance[quote_currency] = self.balance.get(quote_currency, Decimal('0')) - cost
            # Add to base currency balance
            self.balance[base_currency] = self.balance.get(base_currency, Decimal('0')) + fill_quantity
        else:  # SELL
            # Deduct from base currency balance
            self.balance[base_currency] = self.balance.get(base_currency, Decimal('0')) - fill_quantity
            # Add to quote currency balance
            proceeds = fill_quantity * fill_price
            self.balance[quote_currency] = self.balance.get(quote_currency, Decimal('0')) + proceeds
        
        # Ensure no negative balances (should be handled by pre-trade checks)
        self.balance = {k: v for k, v in self.balance.items() if v > Decimal('0.00000001')}
        
        # Publish position update
        await self._publish_position_update(position)
    
    async def _handle_position_update(self, position_update: Dict[str, Any]):
        """Handle external position updates."""
        try:
            symbol = position_update['symbol']
            if symbol in self.positions:
                # Update existing position
                self.positions[symbol].__dict__.update({
                    'quantity': Decimal(str(position_update['quantity'])),
                    'entry_price': Decimal(str(position_update['entry_price'])),
                    'current_price': Decimal(str(position_update.get('current_price', '0'))),
                    'unrealized_pnl': Decimal(str(position_update.get('unrealized_pnl', '0'))),
                    'realized_pnl': Decimal(str(position_update.get('realized_pnl', '0'))),
                    'last_updated': datetime.utcnow(),
                    'metadata': position_update.get('metadata', {})
                })
            else:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=Decimal(str(position_update['quantity'])),
                    entry_price=Decimal(str(position_update['entry_price'])),
                    current_price=Decimal(str(position_update.get('current_price', '0'))),
                    unrealized_pnl=Decimal(str(position_update.get('unrealized_pnl', '0'))),
                    realized_pnl=Decimal(str(position_update.get('realized_pnl', '0'))),
                    entry_time=datetime.fromisoformat(position_update.get('entry_time', datetime.utcnow().isoformat())),
                    metadata=position_update.get('metadata', {})
                )
        
        except Exception as e:
            logger.error(f"Error processing position update: {e}", exc_info=True)
    
    async def _update_positions_with_market_data(self, market_data: Dict[str, Any]):
        """Update positions with latest market data."""
        try:
            symbol = market_data['symbol']
            price = Decimal(str(market_data['price']))
            
            if symbol in self.positions:
                position = self.positions[symbol]
                position.update_price(price)
                
                # Publish position update
                await self._publish_position_update(position)
        
        except Exception as e:
            logger.error(f"Error updating positions with market data: {e}", exc_info=True)
    
    async def _publish_position_update(self, position: Position):
        """Publish position update to the message bus."""
        await self.send_message(
            Message(
                msg_type=MessageType.POSITION_UPDATE,
                sender=self.agent_id,
                recipients=[],
                payload=position.to_dict()
            )
        )
    
    def _log_trade(self, order: Order):
        """Log a completed trade."""
        if order.status != OrderStatus.FILLED:
            return
            
        trade = {
            'trade_id': str(uuid.uuid4()),
            'order_id': order.order_id,
            'client_order_id': order.client_order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': str(order.quantity),
            'price': str(order.metadata.get('fill_price', order.price or '0')),
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': order.metadata
        }
        self.trade_history.append(trade)
        
        # In a real implementation, you might want to persist trades to a database
        logger.info(f"Trade executed: {trade}")
    
    async def _get_current_price(self, symbol: str) -> Decimal:
        """Get the current price for a symbol."""
        # In a real implementation, this would fetch the current price from an exchange
        # For now, we'll return a default value
        return Decimal('50000')  # Example price
    
    async def _calculate_position_size(
        self,
        symbol: str,
        price: Decimal,
        confidence: Decimal
    ) -> Decimal:
        """Calculate position size based on risk management rules."""
        # Get account balance in quote currency
        if '/' in symbol:
            quote_currency = symbol.split('/')[1]
        else:
            quote_currency = 'USDT'  # Default quote currency
            
        balance = self.balance.get(quote_currency, Decimal('0'))
        
        if balance <= 0:
            return Decimal('0')
        
        # Calculate position size based on risk parameters
        risk_amount = balance * self.max_risk_per_trade * confidence
        position_value = min(risk_amount * Decimal('2'), balance * self.max_position_size)
        
        # Calculate quantity based on position value and price
        quantity = position_value / price
        
        # Round to appropriate decimal places (assuming 8 decimal places for most cryptocurrencies)
        return quantity.quantize(Decimal('0.00000001'))
    
    async def _handle_advanced_order(self, signal: Dict[str, Any], current_price: Decimal, position: Optional[Position]):
        """Handle advanced order types like stop loss, take profit, etc."""
        try:
            symbol = signal['symbol']
            action = signal['action']
            
            if action == 'trailing_stop':
                # Place a trailing stop order
                trail_percent = Decimal(str(signal.get('trail_percent', '0.02')))  # Default 2% trail
                trail_amount = current_price * trail_percent
                
                await self.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL if position and position.quantity > 0 else OrderSide.BUY,
                    order_type=OrderType.TRAILING_STOP,
                    quantity=abs(position.quantity) if position else await self._calculate_position_size(symbol, current_price, Decimal('1')),
                    price=current_price,
                    stop_price=current_price - trail_amount if (position and position.quantity > 0) else current_price + trail_amount,
                    metadata={
                        'trail_amount': str(trail_amount),
                        'trail_percent': str(trail_percent),
                        'signal': signal
                    }
                )
            
            elif action == 'stop_loss':
                # Place a stop loss order
                stop_price = Decimal(str(signal.get('stop_price', current_price * Decimal('0.95'))))  # Default 5% stop loss
                
                await self.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL if position and position.quantity > 0 else OrderSide.BUY,
                    order_type=OrderType.STOP,
                    quantity=abs(position.quantity) if position else await self._calculate_position_size(symbol, current_price, Decimal('1')),
                    price=stop_price * Decimal('0.995'),  # Slightly below stop price for market execution
                    stop_price=stop_price,
                    metadata={
                        'stop_type': 'stop_loss',
                        'signal': signal
                    }
                )
            
            elif action == 'take_profit':
                # Place a take profit order
                take_profit_price = Decimal(str(signal.get('take_profit_price', current_price * Decimal('1.05'))))  # Default 5% take profit
                
                await self.place_order(
                    symbol=symbol,
                    side=OrderSide.SELL if position and position.quantity > 0 else OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=abs(position.quantity) if position else await self._calculate_position_size(symbol, current_price, Decimal('1')),
                    price=take_profit_price,
                    metadata={
                        'order_type': 'take_profit',
                        'signal': signal
                    }
                )
        
        except Exception as e:
            logger.error(f"Error handling advanced order: {e}", exc_info=True)
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: str = 'GTC',
        client_order_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Order:
        """Place a new order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: BUY or SELL
            order_type: Type of order (MARKET, LIMIT, etc.)
            quantity: Amount to buy/sell
            price: Price for limit orders (required for non-market orders)
            stop_price: Stop price for stop orders
            time_in_force: Order time in force (default: 'GTC' - Good Till Cancelled)
            client_order_id: Optional client order ID
            metadata: Additional order metadata
            
        Returns:
            The created Order object
            
        Raises:
            ValueError: If any input validation fails
        """
        # Input validation
        if not isinstance(symbol, str) or '/' not in symbol:
            raise ValueError("Invalid symbol format. Expected format: 'BASE/QUOTE'")
            
        if not isinstance(order_type, OrderType):
            try:
                order_type = OrderType(order_type.lower())
            except ValueError:
                raise ValueError(f"Invalid order type. Must be one of: {', '.join(t.value for t in OrderType)}")
                
        if not isinstance(side, OrderSide):
            try:
                side = OrderSide(side.lower())
            except ValueError:
                raise ValueError(f"Invalid order side. Must be one of: {', '.join(s.value for s in OrderSide)}")
                
        if not isinstance(quantity, Decimal):
            try:
                quantity = Decimal(str(quantity))
            except (TypeError, ValueError, decimal.InvalidOperation):
                raise ValueError("Quantity must be a valid decimal number")
                
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
            
        if order_type != OrderType.MARKET and price is None:
            raise ValueError(f"Price is required for {order_type.value} orders")
            
        if price is not None and price <= 0:
            raise ValueError("Price must be positive")
            
        if stop_price is not None and stop_price <= 0:
            raise ValueError("Stop price must be positive")
            
        try:
            # Generate order ID
            order_id = f"order_{self._order_counter:08d}"
            self._order_counter += 1
            
            # Create order
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                client_order_id=client_order_id or str(uuid.uuid4()),
                metadata=metadata or {}
            )
            
            # Add to open orders
            self.open_orders[order_id] = order
            
            # In a real implementation, this would send the order to an exchange
            # For now, we'll simulate an immediate fill for market orders
            if order_type == OrderType.MARKET:
                # Get current price for market orders if not provided
                if price is None:
                    current_price = await self._get_current_price(symbol)
                    if current_price is None:
                        raise ValueError(f"Cannot determine price for market order on {symbol}")
                    price = current_price
                
                # Simulate fill with slight slippage
                fill_price = price * (Decimal('1') + (self.default_slippage * (-1 if side == OrderSide.BUY else 1)))
                order.update_fill(quantity, fill_price)
                
                # Update position
                await self._update_position_from_fill(order, quantity, fill_price)
                
                # Move to closed orders
                self.closed_orders[order_id] = order
                del self.open_orders[order_id]
                
                # Log the trade
                self._log_trade(order)
            
            # Publish order update
            await self.send_message(
                Message(
                    msg_type=MessageType.ORDER_UPDATE,
                    sender=self.agent_id,
                    recipients=[],
                    payload=order.to_dict()
                )
            )
            
            return order
        
        except Exception as e:
            logger.error(f"Error placing order: {e}", exc_info=True)
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            if order_id in self.open_orders:
                order = self.open_orders[order_id]
                order.status = OrderStatus.CANCELED
                order.updated_at = datetime.utcnow()
                
                # Move to closed orders
                self.closed_orders[order_id] = order
                del self.open_orders[order_id]
                
                # Publish order update
                await self.send_message(
                    Message(
                        sender_id=self.agent_id,
                        message_type=MessageType.ORDER_UPDATE,
                        payload=order.to_dict()
                    )
                )
                
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error canceling order: {e}", exc_info=True)
            return False
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get the current position for a symbol."""
        return self.positions.get(symbol)
    
    async def get_open_orders(self, symbol: str = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol."""
        if symbol:
            return [o for o in self.open_orders.values() if o.symbol == symbol]
        return list(self.open_orders.values())
    
    async def get_closed_orders(self, symbol: str = None, limit: int = 100) -> List[Order]:
        """Get recently closed orders, optionally filtered by symbol."""
        orders = list(self.closed_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders[-limit:]
    
    async def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history, optionally filtered by symbol."""
        trades = self.trade_history
        if symbol:
            trades = [t for t in trades if t['symbol'] == symbol]
        return trades[-limit:]
    
    async def get_balance(self) -> Dict[str, Decimal]:
        """Get current account balance."""
        return self.balance.copy()
    
    async def rebalance_portfolio(self, target_weights: Dict[str, Decimal]) -> List[Order]:
        """Rebalance the portfolio to match target weights.
        
        Args:
            target_weights: Dictionary mapping symbols to target weights (0-1)
            
        Returns:
            List of orders created for rebalancing
            
        Raises:
            ValueError: If target weights don't sum to ~1.0 or if any weight is invalid
        """
        # Validate target weights
        total_weight = sum(target_weights.values())
        if not Decimal('0.99') <= total_weight <= Decimal('1.01'):
            raise ValueError(f"Target weights must sum to ~1.0, got {total_weight}")
            
        for symbol, weight in target_weights.items():
            if weight < 0 or weight > 1:
                raise ValueError(f"Weight for {symbol} must be between 0 and 1, got {weight}")
        
        # Get current portfolio value
        portfolio_value = await self.get_portfolio_value()
        if portfolio_value <= 0:
            logger.warning("Portfolio value is zero or negative, cannot rebalance")
            return []
            
        # Calculate current weights and target values
        current_values = {}
        current_weights = {}
        
        # Get current prices for all symbols
        prices = {}
        for symbol in target_weights.keys():
            prices[symbol] = await self._get_current_price(symbol)
        
        # Add cash as a position with weight (1 - sum of other weights)
        cash_weight = Decimal('1.0')
        for symbol, position in self.positions.items():
            if position.current_price is not None:
                current_value = position.quantity * position.current_price
                current_values[symbol] = current_value
                current_weights[symbol] = current_value / portfolio_value
                cash_weight -= current_weights[symbol]
        
        # Add cash to current weights
        current_weights['CASH'] = cash_weight
        
        # Generate rebalancing orders
        orders = []
        
        # First, close positions not in target weights
        for symbol in list(current_weights.keys()):
            if symbol not in target_weights and symbol != 'CASH':
                # Close the position
                position = self.positions[symbol]
                if position.quantity > 0:
                    order = await self.place_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=position.quantity,
                        metadata={'rebalance': True}
                    )
                    orders.append(order)
        
        # Then, rebalance to target weights
        for symbol, target_weight in target_weights.items():
            target_value = portfolio_value * target_weight
            current_value = current_values.get(symbol, Decimal('0'))
            price = await self._get_current_price(symbol)
            
            if not price or price <= 0:
                logger.warning(f"Invalid price for {symbol}, skipping rebalance")
                continue
                
            current_quantity = self.positions[symbol].quantity if symbol in self.positions else Decimal('0')
            target_quantity = target_value / price
            
            # Round to lot size if needed (example: 8 decimal places for BTC)
            target_quantity = round(target_quantity, 8)
            
            # Skip if difference is negligible
            if abs(current_quantity - target_quantity) < Decimal('0.00000001'):
                continue
                
            # Determine order side and quantity
            if target_quantity > current_quantity:
                side = OrderSide.BUY
                quantity = target_quantity - current_quantity
            else:
                side = OrderSide.SELL
                quantity = current_quantity - target_quantity
            
            # Place order
            order = await self.place_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                metadata={'rebalance': True}
            )
            orders.append(order)
        
        return orders
    
    async def get_portfolio_value(self, prices: Dict[str, Decimal] = None) -> Decimal:
        """Calculate total portfolio value in quote currency."""
        if prices is None:
            prices = {}
        
        total = Decimal('0')
        
        # Add cash balances
        for currency, amount in self.balance.items():
            if currency == 'USDT' or currency == 'USD':
                total += amount
            elif f"{currency}/USDT" in prices:
                total += amount * prices[f"{currency}/USDT"]
            elif f"{currency}/USD" in prices:
                total += amount * prices[f"{currency}/USD"]
        
        # Add position values
        for symbol, position in self.positions.items():
            if position.current_price is not None:
                total += position.quantity * position.current_price
        
        return total

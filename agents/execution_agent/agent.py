"""
Execution Agent implementation for the AI Trading System.

This module provides the ExecutionAgent class that handles order routing,
optimization, and backtesting of trading strategies.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import json
import uuid
import pandas as pd
import numpy as np

from ..base_agent import BaseAgent, Message, MessageType

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Status of an order."""
    NEW = 'new'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'
    CANCELED = 'canceled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'

class OrderSide(Enum):
    """Side of an order."""
    BUY = 'buy'
    SELL = 'sell'

class OrderType(Enum):
    """Type of an order."""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'
    TRAILING_STOP = 'trailing_stop'

@dataclass
class ExecutionParameters:
    """Parameters for order execution."""
    slippage: float = 0.001  # Default slippage (0.1%)
    max_slippage: float = 0.01  # Maximum allowed slippage (1%)
    max_retries: int = 3  # Maximum number of retries for failed orders
    retry_delay: float = 0.1  # Delay between retries in seconds
    use_vwap: bool = True  # Whether to use VWAP for execution
    vwap_window: int = 5  # Window size for VWAP calculation in minutes
    min_order_size: float = 10.0  # Minimum order size in quote currency
    max_order_size: float = 100000.0  # Maximum order size in quote currency
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'slippage': self.slippage,
            'max_slippage': self.max_slippage,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'use_vwap': self.use_vwap,
            'vwap_window': self.vwap_window,
            'min_order_size': self.min_order_size,
            'max_order_size': self.max_order_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionParameters':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class ExecutionResult:
    """Result of an order execution."""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    filled_quantity: Decimal
    price: Decimal
    average_fill_price: Decimal
    status: OrderStatus
    fees: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': str(self.quantity),
            'filled_quantity': str(self.filled_quantity),
            'price': str(self.price),
            'average_fill_price': str(self.average_fill_price),
            'status': self.status.value,
            'fees': str(self.fees),
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        """Create from dictionary."""
        return cls(
            order_id=data['order_id'],
            client_order_id=data['client_order_id'],
            symbol=data['symbol'],
            side=OrderSide(data['side']),
            order_type=OrderType(data['order_type']),
            quantity=Decimal(data['quantity']),
            filled_quantity=Decimal(data.get('filled_quantity', '0')),
            price=Decimal(data['price']),
            average_fill_price=Decimal(data.get('average_fill_price', data['price'])),
            status=OrderStatus(data['status']),
            fees=Decimal(data.get('fees', '0')),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )

@dataclass
class BacktestResult:
    """Result of a backtest."""
    strategy_name: str
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    initial_balance: Decimal
    final_balance: Decimal
    total_return: Decimal
    annualized_return: Decimal
    max_drawdown: Decimal
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: Decimal
    avg_loss: Decimal
    max_win: Decimal
    max_loss: Decimal
    metrics: Dict[str, Any] = field(default_factory=dict)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: List[Dict[datetime, Decimal]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'initial_balance': str(self.initial_balance),
            'final_balance': str(self.final_balance),
            'total_return': float(self.total_return),
            'annualized_return': float(self.annualized_return),
            'max_drawdown': float(self.max_drawdown),
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': float(self.avg_win),
            'avg_loss': float(self.avg_loss),
            'max_win': float(self.max_win),
            'max_loss': float(self.max_loss),
            'metrics': self.metrics,
            'trades': self.trades,
            'equity_curve': [
                {'timestamp': ts.isoformat(), 'balance': float(bal)}
                for ts, bal in self.equity_curve
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestResult':
        """Create from dictionary."""
        return cls(
            strategy_name=data['strategy_name'],
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']),
            initial_balance=Decimal(str(data['initial_balance'])),
            final_balance=Decimal(str(data['final_balance'])),
            total_return=Decimal(str(data['total_return'])),
            annualized_return=Decimal(str(data['annualized_return'])),
            max_drawdown=Decimal(str(data['max_drawdown'])),
            sharpe_ratio=data['sharpe_ratio'],
            sortino_ratio=data['sortino_ratio'],
            win_rate=data['win_rate'],
            profit_factor=data['profit_factor'],
            total_trades=data['total_trades'],
            winning_trades=data['winning_trades'],
            losing_trades=data['losing_trades'],
            avg_win=Decimal(str(data['avg_win'])),
            avg_loss=Decimal(str(data['avg_loss'])),
            max_win=Decimal(str(data['max_win'])),
            max_loss=Decimal(str(data['max_loss'])),
            metrics=data.get('metrics', {}),
            trades=data.get('trades', []),
            equity_curve=[
                (datetime.fromisoformat(item['timestamp']), Decimal(str(item['balance'])))
                for item in data.get('equity_curve', [])
            ]
        )

class ExecutionAgent(BaseAgent):
    """Agent responsible for order execution and backtesting."""
    
    def __init__(
        self,
        agent_id: str = 'execution_agent',
        execution_params: Optional[ExecutionParameters] = None,
        **kwargs
    ):
        """Initialize the ExecutionAgent.
        
        Args:
            agent_id: Unique identifier for the agent
            execution_params: Execution parameters
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(agent_id=agent_id, **kwargs)
        self.execution_params = execution_params or ExecutionParameters()
        self.active_orders: Dict[str, Dict] = {}
        self.order_history: List[Dict] = []
        self.exchange_connections: Dict[str, Any] = {}
        self._setup_exchange_connections()
    
    def _setup_exchange_connections(self):
        """Set up connections to exchanges."""
        # In a real implementation, this would initialize API clients for each exchange
        self.exchange_connections = {
            'binance': {
                'connected': False,
                'client': None,
                'markets': {}
            },
            'ftx': {
                'connected': False,
                'client': None,
                'markets': {}
            },
            # Add more exchanges as needed
        }
    
    async def start(self):
        """Start the agent and subscribe to relevant message types."""
        await super().start()
        # Subscribe to messages this agent cares about
        self.message_bus.subscribe(
            self.agent_id,
            [
                MessageType.ORDER_APPROVED,
                MessageType.CANCEL_ORDER,
                MessageType.MARKET_DATA,
                MessageType.BACKTEST_REQUEST,
                MessageType.ORDER_STATUS_REQUEST
            ]
        )
        
        # Connect to exchanges
        await self._connect_to_exchanges()
        
        logger.info(f"{self.agent_id} started and subscribed to messages")
    
    async def _process_message(self, message: Message):
        """Process incoming messages."""
        try:
            if message.message_type == MessageType.ORDER_APPROVED:
                await self._handle_order_approved(message)
            
            elif message.message_type == MessageType.CANCEL_ORDER:
                await self._handle_cancel_order(message)
            
            elif message.message_type == MessageType.MARKET_DATA:
                await self._handle_market_data(message.payload)
            
            elif message.message_type == MessageType.BACKTEST_REQUEST:
                await self._handle_backtest_request(message)
            
            elif message.message_type == MessageType.ORDER_STATUS_REQUEST:
                await self._handle_order_status_request(message)
        
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await self._send_error(
                str(e),
                message.sender_id,
                message.message_id
            )
    
    async def _connect_to_exchanges(self):
        """Connect to all configured exchanges."""
        # In a real implementation, this would establish connections to each exchange
        # and load market information
        for exchange_id in self.exchange_connections:
            try:
                logger.info(f"Connecting to {exchange_id}...")
                # Simulate connection delay
                await asyncio.sleep(0.1)
                self.exchange_connections[exchange_id]['connected'] = True
                logger.info(f"Connected to {exchange_id}")
            except Exception as e:
                logger.error(f"Failed to connect to {exchange_id}: {e}")
    
    async def _handle_order_approved(self, message: Message):
        """Handle an approved order request."""
        order_request = message.payload
        order_id = order_request.get('order_id', str(uuid.uuid4()))
        client_order_id = order_request.get('client_order_id', str(uuid.uuid4()))
        symbol = order_request['symbol']
        side = OrderSide(order_request['side'])
        order_type = OrderType(order_request.get('order_type', 'market'))
        quantity = Decimal(str(order_request['quantity']))
        price = Decimal(str(order_request.get('price', '0'))) if order_request.get('price') else None
        
        # Log the order
        logger.info(f"Executing {order_type.value} {side.value} order for {quantity} {symbol} at {price or 'market'}")
        
        try:
            # Execute the order
            execution = await self.execute_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                client_order_id=client_order_id,
                metadata={
                    'request_id': message.message_id,
                    'requester': message.sender_id,
                    **order_request.get('metadata', {})
                }
            )
            
            # Store the order
            self.active_orders[order_id] = {
                'order_id': order_id,
                'client_order_id': client_order_id,
                'symbol': symbol,
                'side': side,
                'order_type': order_type,
                'quantity': quantity,
                'price': price,
                'status': execution.status,
                'filled_quantity': execution.filled_quantity,
                'average_fill_price': execution.average_fill_price,
                'fees': execution.fees,
                'timestamp': execution.timestamp,
                'metadata': execution.metadata
            }
            
            # Add to order history
            self.order_history.append(self.active_orders[order_id])
            
            # Send execution report
            await self._send_execution_report(execution, message.sender_id, message.message_id)
            
            # If order is not fully filled, schedule a check
            if execution.status != OrderStatus.FILLED:
                asyncio.create_task(self._monitor_order(execution, message.sender_id))
        
        except Exception as e:
            logger.error(f"Error executing order: {e}", exc_info=True)
            await self._send_error(
                f"Failed to execute order: {str(e)}",
                message.sender_id,
                message.message_id
            )
    
    async def _handle_cancel_order(self, message: Message):
        """Handle an order cancellation request."""
        order_id = message.payload.get('order_id')
        if not order_id:
            await self._send_error(
                "Missing order_id in cancel request",
                message.sender_id,
                message.message_id
            )
            return
        
        if order_id not in self.active_orders:
            await self._send_error(
                f"Order {order_id} not found or already completed",
                message.sender_id,
                message.message_id
            )
            return
        
        try:
            # Cancel the order
            await self.cancel_order(order_id)
            
            # Update order status
            self.active_orders[order_id]['status'] = OrderStatus.CANCELED
            self.active_orders[order_id]['updated_at'] = datetime.utcnow()
            
            # Send confirmation
            await self.send_message(
                Message(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type=MessageType.ORDER_CANCELED,
                    payload={
                        'order_id': order_id,
                        'client_order_id': self.active_orders[order_id]['client_order_id'],
                        'status': 'canceled',
                        'timestamp': datetime.utcnow().isoformat()
                    },
                    in_reply_to=message.message_id
                )
            )
            
            # Remove from active orders
            del self.active_orders[order_id]
        
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}", exc_info=True)
            await self._send_error(
                f"Failed to cancel order: {str(e)}",
                message.sender_id,
                message.message_id
            )
    
    async def _handle_market_data(self, market_data: Dict[str, Any]):
        """Handle incoming market data."""
        # Update order execution logic based on market data
        # For example, update VWAP, check for stop/take-profit levels, etc.
        pass
    
    async def _handle_backtest_request(self, message: Message):
        """Handle a backtest request."""
        try:
            request = message.payload
            strategy = request['strategy']
            symbol = request['symbol']
            timeframe = request.get('timeframe', '1d')
            start_time = datetime.fromisoformat(request['start_time'])
            end_time = datetime.fromisoformat(request.get('end_time', datetime.utcnow().isoformat()))
            initial_balance = Decimal(str(request.get('initial_balance', 10000)))
            
            logger.info(f"Starting backtest for {strategy} on {symbol} ({timeframe}) from {start_time} to {end_time}")
            
            # Run the backtest
            result = await self.run_backtest(
                strategy=strategy,
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                initial_balance=initial_balance,
                parameters=request.get('parameters', {})
            )
            
            # Send backtest results
            await self.send_message(
                Message(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type=MessageType.BACKTEST_RESULT,
                    payload=result.to_dict(),
                    in_reply_to=message.message_id
                )
            )
        
        except Exception as e:
            logger.error(f"Error running backtest: {e}", exc_info=True)
            await self._send_error(
                f"Backtest failed: {str(e)}",
                message.sender_id,
                message.message_id
            )
    
    async def _handle_order_status_request(self, message: Message):
        """Handle an order status request."""
        order_id = message.payload.get('order_id')
        client_order_id = message.payload.get('client_order_id')
        
        if not order_id and not client_order_id:
            await self._send_error(
                "Must provide either order_id or client_order_id",
                message.sender_id,
                message.message_id
            )
            return
        
        # Find the order
        order = None
        if order_id and order_id in self.active_orders:
            order = self.active_orders[order_id]
        elif client_order_id:
            for o in self.active_orders.values():
                if o['client_order_id'] == client_order_id:
                    order = o
                    break
        
        if not order:
            # Check historical orders
            historical_orders = [o for o in self.order_history 
                               if (order_id and o['order_id'] == order_id) or 
                               (client_order_id and o['client_order_id'] == client_order_id)]
            if historical_orders:
                order = historical_orders[-1]  # Most recent
        
        if not order:
            await self._send_error(
                f"Order not found: {order_id or client_order_id}",
                message.sender_id,
                message.message_id
            )
            return
        
        # Send order status
        await self.send_message(
            Message(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.ORDER_STATUS_UPDATE,
                payload={
                    'order_id': order['order_id'],
                    'client_order_id': order['client_order_id'],
                    'symbol': order['symbol'],
                    'side': order['side'].value,
                    'order_type': order['order_type'].value,
                    'quantity': str(order['quantity']),
                    'filled_quantity': str(order.get('filled_quantity', '0')),
                    'price': str(order.get('price', '0')),
                    'average_fill_price': str(order.get('average_fill_price', '0')),
                    'status': order['status'].value,
                    'fees': str(order.get('fees', '0')),
                    'timestamp': order['timestamp'].isoformat(),
                    'metadata': order.get('metadata', {})
                },
                in_reply_to=message.message_id
            )
        )
    
    async def _monitor_order(self, execution: ExecutionResult, requester_id: str):
        """Monitor an order for fills and updates."""
        # In a real implementation, this would poll the exchange for updates
        # For now, we'll simulate a fill after a short delay
        await asyncio.sleep(1)
        
        if execution.order_id in self.active_orders:
            order = self.active_orders[execution.order_id]
            
            # Simulate a fill
            if order['status'] == OrderStatus.NEW:
                order['status'] = OrderStatus.FILLED
                order['filled_quantity'] = order['quantity']
                order['average_fill_price'] = order.get('price', Decimal('50000'))
                order['fees'] = order['quantity'] * order['average_fill_price'] * Decimal('0.001')  # 0.1% fee
                order['updated_at'] = datetime.utcnow()
                
                # Send update
                await self._send_execution_report(
                    ExecutionResult(
                        order_id=order['order_id'],
                        client_order_id=order['client_order_id'],
                        symbol=order['symbol'],
                        side=order['side'],
                        order_type=order['order_type'],
                        quantity=order['quantity'],
                        filled_quantity=order['filled_quantity'],
                        price=order.get('price', Decimal('0')),
                        average_fill_price=order['average_fill_price'],
                        status=order['status'],
                        fees=order['fees'],
                        timestamp=order['updated_at'],
                        metadata=order.get('metadata', {})
                    ),
                    requester_id,
                    order.get('metadata', {}).get('request_id')
                )
                
                # Move to order history
                self.order_history.append(order)
                del self.active_orders[execution.order_id]
    
    async def _send_execution_report(
        self,
        execution: ExecutionResult,
        recipient_id: str,
        original_message_id: str = None
    ):
        """Send an execution report to the requester."""
        await self.send_message(
            Message(
                sender_id=self.agent_id,
                recipient_id=recipient_id,
                message_type=MessageType.EXECUTION_REPORT,
                payload=execution.to_dict(),
                in_reply_to=original_message_id
            )
        )
    
    async def _send_error(
        self,
        error: str,
        recipient_id: str,
        original_message_id: str = None
    ):
        """Send an error message."""
        await self.send_message(
            Message(
                sender_id=self.agent_id,
                recipient_id=recipient_id,
                message_type=MessageType.ERROR,
                payload={'error': error},
                in_reply_to=original_message_id
            )
        )
    
    async def execute_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        client_order_id: Optional[str] = None,
        exchange_id: str = 'binance',  # Default exchange
        **kwargs
    ) -> ExecutionResult:
        """Execute an order on the specified exchange.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side (buy/sell)
            order_type: Order type (market/limit/stop/etc.)
            quantity: Order quantity in base currency
            price: Limit/stop price (required for limit/stop orders)
            client_order_id: Optional client-specified order ID
            exchange_id: Exchange ID to execute on
            **kwargs: Additional order parameters
            
        Returns:
            ExecutionResult with order execution details
        """
        # Generate order ID if not provided
        order_id = str(uuid.uuid4())
        client_order_id = client_order_id or f"{exchange_id}_{int(datetime.utcnow().timestamp() * 1000)}"
        
        # Validate order parameters
        if quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        if order_type in [OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT] and price is None:
            raise ValueError(f"Price is required for {order_type.value} orders")
        
        # Check exchange connection
        if exchange_id not in self.exchange_connections or not self.exchange_connections[exchange_id]['connected']:
            raise ConnectionError(f"Not connected to exchange: {exchange_id}")
        
        # In a real implementation, this would call the exchange API
        # For now, we'll simulate order execution
        logger.info(
            f"Executing {order_type.value} {side.value} order for {quantity} {symbol} "
            f"at {price or 'market'} on {exchange_id}"
        )
        
        # Simulate execution with slippage
        execution_price = await self._simulate_execution_price(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price
        )
        
        # Calculate fees (simplified)
        fees = quantity * execution_price * Decimal('0.001')  # 0.1% fee
        
        # Create execution result
        execution = ExecutionResult(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            filled_quantity=quantity,  # Assume immediate fill for simulation
            price=price or execution_price,
            average_fill_price=execution_price,
            status=OrderStatus.FILLED,
            fees=fees,
            timestamp=datetime.utcnow(),
            metadata={
                'exchange': exchange_id,
                'simulated': True,
                **kwargs.get('metadata', {})
            }
        )
        
        return execution
    
    async def cancel_order(
        self,
        order_id: str,
        symbol: Optional[str] = None,
        exchange_id: str = 'binance'
    ) -> bool:
        """Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Optional symbol for validation
            exchange_id: Exchange ID where the order was placed
            
        Returns:
            True if cancellation was successful, False otherwise
        """
        # In a real implementation, this would call the exchange API
        logger.info(f"Canceling order {order_id} on {exchange_id}")
        
        # Simulate cancellation
        await asyncio.sleep(0.1)
        
        return True
    
    async def run_backtest(
        self,
        strategy: str,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        initial_balance: Decimal = Decimal('10000'),
        parameters: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """Run a backtest for the specified strategy and parameters.
        
        Args:
            strategy: Strategy name or configuration
            symbol: Trading pair symbol
            timeframe: Timeframe for the backtest (e.g., '1d', '1h')
            start_time: Backtest start time
            end_time: Backtest end time
            initial_balance: Initial account balance in quote currency
            parameters: Strategy parameters
            
        Returns:
            BacktestResult with performance metrics and trades
        """
        logger.info(
            f"Running backtest for {strategy} on {symbol} ({timeframe}) from {start_time} to {end_time} "
            f"with initial balance {initial_balance}"
        )
        
        # In a real implementation, this would:
        # 1. Load historical data for the symbol and timeframe
        # 2. Initialize the strategy with the given parameters
        # 3. Run the strategy on the historical data
        # 4. Track trades, P&L, and other metrics
        # 5. Generate a performance report
        
        # For now, we'll simulate a simple backtest with random trades
        np.random.seed(42)  # For reproducible results
        
        # Generate random trades
        num_trades = 100
        trade_returns = np.random.normal(0.001, 0.01, num_trades)  # Random returns with mean 0.1% and std 1%
        trade_pnls = initial_balance * Decimal('0.01') * np.array([Decimal(str(r)) for r in trade_returns])
        
        # Calculate cumulative P&L
        cumulative_pnl = np.cumsum(trade_pnls)
        running_balance = [initial_balance + pnl for pnl in cumulative_pnl]
        
        # Calculate metrics
        total_return = (running_balance[-1] / initial_balance - 1) * 100
        max_drawdown = self._calculate_max_drawdown(running_balance)
        sharpe_ratio = self._calculate_sharpe_ratio(trade_returns)
        sortino_ratio = self._calculate_sortino_ratio(trade_returns)
        
        # Count winning and losing trades
        winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
        losing_trades = num_trades - winning_trades
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        # Calculate average win/loss
        avg_win = np.mean([pnl for pnl in trade_pnls if pnl > 0]) if winning_trades > 0 else Decimal('0')
        avg_loss = abs(np.mean([pnl for pnl in trade_pnls if pnl < 0])) if losing_trades > 0 else Decimal('0')
        profit_factor = (winning_trades * avg_win) / (losing_trades * avg_loss) if losing_trades > 0 else float('inf')
        
        # Generate trade history
        trades = []
        for i in range(num_trades):
            trade_time = start_time + (end_time - start_time) * (i / num_trades)
            trades.append({
                'trade_id': str(uuid.uuid4()),
                'symbol': symbol,
                'side': 'buy' if trade_returns[i] > 0 else 'sell',
                'quantity': Decimal('1.0'),
                'price': Decimal('50000'),
                'cost': Decimal('50000'),
                'fee': Decimal('5.0'),
                'pnl': float(trade_pnls[i]),
                'return_pct': float(trade_returns[i] * 100),
                'entry_time': trade_time.isoformat(),
                'exit_time': (trade_time + timedelta(hours=1)).isoformat(),
                'metadata': {
                    'strategy': strategy,
                    'parameters': parameters or {}
                }
            })
        
        # Generate equity curve (hourly)
        equity_curve = []
        num_points = 100
        for i in range(num_points):
            point_time = start_time + (end_time - start_time) * (i / num_points)
            idx = min(int(i / num_points * num_trades), num_trades - 1)
            equity_curve.append((point_time, running_balance[idx]))
        
        # Calculate annualized return
        days = (end_time - start_time).days
        years = days / 365.25
        annualized_return = ((running_balance[-1] / initial_balance) ** (1 / years) - 1) * 100 if years > 0 else Decimal('0')
        
        # Create backtest result
        result = BacktestResult(
            strategy_name=strategy,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            initial_balance=initial_balance,
            final_balance=running_balance[-1],
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown * 100,  # Convert to percentage
            sharpe_ratio=float(sharpe_ratio),
            sortino_ratio=float(sortino_ratio),
            win_rate=win_rate,
            profit_factor=float(profit_factor),
            total_trades=num_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_win=max(trade_pnls) if trade_pnls else Decimal('0'),
            max_loss=min(trade_pnls) if trade_pnls else Decimal('0'),
            metrics={
                'volatility': float(np.std(trade_returns) * np.sqrt(252)),  # Annualized
                'calmar_ratio': float(annualized_return / (max_drawdown * 100)) if max_drawdown > 0 else 0.0,
                'expectancy': float((win_rate * avg_win) - ((1 - win_rate) * avg_loss)),
                'avg_trade_duration': '1h',
                'profit_per_day': float((running_balance[-1] - initial_balance) / max(1, days)),
                'risk_free_rate': 0.0
            },
            trades=trades,
            equity_curve=equity_curve
        )
        
        return result
    
    async def _simulate_execution_price(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None
    ) -> Decimal:
        """Simulate order execution with slippage and market impact."""
        # In a real implementation, this would use order book data
        # For now, we'll use a simple slippage model
        base_slippage = self.execution_params.slippage
        
        # Market orders have higher slippage than limit orders
        if order_type == OrderType.MARKET:
            slippage = base_slippage * 2
        else:
            slippage = base_slippage
        
        # Larger orders have more slippage
        size_factor = min(1.0, float(quantity) / 10.0)  # Normalize by some order size
        slippage *= (1.0 + size_factor)
        
        # Get current price (simulated)
        current_price = Decimal('50000')  # Example price
        
        # Calculate execution price with slippage
        if side == OrderSide.BUY:
            execution_price = current_price * (1 + Decimal(str(slippage)))
        else:  # SELL
            execution_price = current_price * (1 - Decimal(str(slippage)))
        
        # For limit orders, ensure we don't get a better price than requested
        if order_type == OrderType.LIMIT and price is not None:
            if side == OrderSide.BUY:
                execution_price = min(execution_price, price)
            else:  # SELL
                execution_price = max(execution_price, price)
        
        return execution_price.quantize(Decimal('0.01'))
    
    def _calculate_max_drawdown(self, equity_curve: List[Decimal]) -> float:
        """Calculate maximum drawdown from an equity curve."""
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = Decimal('0')
        
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return float(max_dd)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from a series of returns."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return float(np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns))
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio from a series of returns."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        downside_returns = np.minimum(0, excess_returns)
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return float('inf' if np.mean(excess_returns) > 0 else 0.0)
        
        return float(np.sqrt(252) * np.mean(excess_returns) / downside_std)

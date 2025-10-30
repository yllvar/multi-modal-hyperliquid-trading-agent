"""
Risk Agent implementation for the AI Trading System.

This module provides the RiskAgent class that handles risk management,
position sizing, and compliance monitoring.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import json
import uuid

from ..base_agent import BaseAgent, Message, MessageType

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk levels for position sizing and trading decisions."""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5

class RiskLimitType(Enum):
    """Types of risk limits that can be set."""
    POSITION_SIZE = 'position_size'
    DAILY_LOSS = 'daily_loss'
    DAILY_TRADES = 'daily_trades'
    LEVERAGE = 'leverage'
    DRAWDOWN = 'drawdown'
    CONCENTRATION = 'concentration'
    LIQUIDITY = 'liquidity'

@dataclass
class RiskLimit:
    """Defines a risk limit with optional time-based constraints."""
    limit_type: RiskLimitType
    value: Decimal
    time_window: Optional[timedelta] = None  # None means no time-based limit
    scope: str = 'all'  # 'all', 'symbol', 'strategy', etc.
    scope_value: Optional[str] = None  # Specific symbol or strategy
    
    def __post_init__(self):
        if not isinstance(self.value, Decimal):
            self.value = Decimal(str(self.value))

@dataclass
class ComplianceRule:
    """Defines a compliance rule for trading."""
    rule_id: str
    name: str
    description: str
    condition: str  # Python expression that evaluates to True/False
    action: str  # 'warn', 'block', 'reduce', etc.
    severity: str  # 'low', 'medium', 'high', 'critical'
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskAssessment:
    """Result of a risk assessment for a trading decision."""
    is_approved: bool
    risk_score: float  # 0.0 (no risk) to 1.0 (maximum risk)
    risk_level: RiskLevel
    reasons: List[str]
    suggested_actions: List[str]
    max_position_size: Optional[Decimal] = None
    max_quantity: Optional[Decimal] = None
    adjusted_quantity: Optional[Decimal] = None  # The quantity after risk adjustment
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskParameters:
    """Container for risk management parameters."""
    max_position_size_pct: float = 0.1  # Max position size as % of portfolio
    max_risk_per_trade_pct: float = 0.02  # Max risk per trade as % of portfolio
    daily_loss_limit_pct: float = 0.05  # Max daily loss as % of portfolio
    max_leverage: float = 3.0  # Maximum allowed leverage
    max_drawdown_pct: float = 0.1  # Max drawdown before reducing position sizes
    min_liquidity_usd: float = 1000000  # Minimum liquidity for a symbol to be traded
    max_concentration_pct: float = 0.3  # Max % of portfolio in a single position
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'max_position_size_pct': self.max_position_size_pct,
            'max_risk_per_trade_pct': self.max_risk_per_trade_pct,
            'daily_loss_limit_pct': self.daily_loss_limit_pct,
            'max_leverage': self.max_leverage,
            'max_drawdown_pct': self.max_drawdown_pct,
            'min_liquidity_usd': self.min_liquidity_usd,
            'max_concentration_pct': self.max_concentration_pct
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskParameters':
        """Create from dictionary."""
        return cls(**data)

class RiskAgent(BaseAgent):
    """Agent responsible for risk management and compliance."""
    
    def __init__(
        self,
        agent_id: str = 'risk_agent',
        risk_parameters: Optional[RiskParameters] = None,
        **kwargs
    ):
        """Initialize the RiskAgent.
        
        Args:
            agent_id: Unique identifier for the agent
            risk_parameters: Risk management parameters
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(agent_id=agent_id, **kwargs)
        self.risk_params = risk_parameters or RiskParameters()
        self.risk_limits: List[RiskLimit] = []
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.portfolio_value: Decimal = Decimal('0')
        self.positions: Dict[str, Dict] = {}
        self.today_pnl: Decimal = Decimal('0')
        self.today_trades: int = 0
        self.historical_pnl: Dict[datetime, Decimal] = {}
        self._setup_default_limits()
        self._setup_default_rules()
    
    def _setup_default_limits(self):
        """Set up default risk limits."""
        self.risk_limits = [
            # Position size limits
            RiskLimit(
                limit_type=RiskLimitType.POSITION_SIZE,
                value=Decimal(str(self.risk_params.max_position_size_pct)),
                scope='all'
            ),
            # Daily loss limit
            RiskLimit(
                limit_type=RiskLimitType.DAILY_LOSS,
                value=Decimal(str(self.risk_params.daily_loss_limit_pct)),
                time_window=timedelta(days=1),
                scope='all'
            ),
            # Leverage limit
            RiskLimit(
                limit_type=RiskLimitType.LEVERAGE,
                value=Decimal(str(self.risk_params.max_leverage)),
                scope='all'
            ),
            # Drawdown limit
            RiskLimit(
                limit_type=RiskLimitType.DRAWDOWN,
                value=Decimal(str(self.risk_params.max_drawdown_pct)),
                scope='all'
            ),
            # Concentration limit
            RiskLimit(
                limit_type=RiskLimitType.CONCENTRATION,
                value=Decimal(str(self.risk_params.max_concentration_pct)),
                scope='all'
            )
        ]
    
    def _setup_default_rules(self):
        """Set up default compliance rules."""
        self.compliance_rules = {
            'min_liquidity': ComplianceRule(
                rule_id='min_liquidity',
                name='Minimum Liquidity Requirement',
                description='Ensure sufficient liquidity before trading',
                condition='liquidity_24h >= min_liquidity',
                action='block',
                severity='high',
                metadata={
                    'min_liquidity': self.risk_params.min_liquidity_usd
                }
            ),
            'max_position_size': ComplianceRule(
                rule_id='max_position_size',
                name='Maximum Position Size',
                description='Limit position size to a percentage of portfolio',
                condition='position_value <= portfolio_value * max_position_size_pct',
                action='block',
                severity='high',
                metadata={
                    'max_position_size_pct': self.risk_params.max_position_size_pct
                }
            ),
            'daily_loss_limit': ComplianceRule(
                rule_id='daily_loss_limit',
                name='Daily Loss Limit',
                description='Stop trading if daily loss exceeds limit',
                condition='abs(daily_pnl) <= portfolio_value * daily_loss_limit_pct',
                action='reduce',
                severity='critical',
                metadata={
                    'daily_loss_limit_pct': self.risk_params.daily_loss_limit_pct
                }
            )
        }
    
    async def start(self):
        """Start the agent and subscribe to relevant message types."""
        await super().start()
        # Subscribe to messages this agent cares about
        self.message_bus.subscribe(
            self.agent_id,
            [
                MessageType.ORDER_REQUEST,
                MessageType.POSITION_UPDATE,
                MessageType.PORTFOLIO_UPDATE,
                MessageType.RISK_CHECK,
                MessageType.MARKET_DATA
            ]
        )
        logger.info(f"{self.agent_id} started and subscribed to messages")
    
    async def _process_message(self, message: Message):
        """Process incoming messages."""
        try:
            if message.message_type == MessageType.ORDER_REQUEST:
                await self._handle_order_request(message)
            
            elif message.message_type == MessageType.POSITION_UPDATE:
                await self._handle_position_update(message.payload)
            
            elif message.message_type == MessageType.PORTFOLIO_UPDATE:
                await self._handle_portfolio_update(message.payload)
            
            elif message.message_type == MessageType.RISK_CHECK:
                await self._handle_risk_check(message)
            
            elif message.message_type == MessageType.MARKET_DATA:
                await self._handle_market_data(message.payload)
        
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await self._send_error(
                str(e),
                message.sender_id,
                message.message_id
            )
    
    async def _handle_order_request(self, message: Message):
        """Handle an order request and perform risk checks."""
        order_request = message.payload
        symbol = order_request.get('symbol')
        quantity = Decimal(str(order_request.get('quantity', '0')))
        order_type = order_request.get('order_type', 'market')
        side = order_request.get('side')
        
        # Check if trading is allowed (e.g., market hours, system status)
        if not await self._is_trading_allowed():
            await self._reject_order(
                order_request,
                message.sender_id,
                message.message_id,
                'Trading is currently not allowed'
            )
            return
        
        # Perform risk assessment
        assessment = await self.assess_order_risk(order_request)
        
        if assessment.is_approved:
            # Forward approved order to execution agent
            await self.send_message(
                Message(
                    sender_id=self.agent_id,
                    recipient_id='execution_agent',
                    message_type=MessageType.ORDER_APPROVED,
                    payload=order_request,
                    in_reply_to=message.message_id
                )
            )
        else:
            # Reject order with risk assessment details
            await self._reject_order(
                order_request,
                message.sender_id,
                message.message_id,
                'Order rejected by risk management',
                assessment
            )
    
    async def _handle_position_update(self, position_data: Dict[str, Any]):
        """Update internal position state."""
        symbol = position_data['symbol']
        self.positions[symbol] = position_data
        
        # Update PnL tracking
        if 'unrealized_pnl' in position_data:
            self._update_pnl(position_data['unrealized_pnl'])
    
    async def _handle_portfolio_update(self, portfolio_data: Dict[str, Any]):
        """Update portfolio information."""
        if 'portfolio_value' in portfolio_data:
            self.portfolio_value = Decimal(str(portfolio_data['portfolio_value']))
        
        if 'positions' in portfolio_data:
            self.positions = {
                p['symbol']: p for p in portfolio_data['positions']
            }
        
        if 'today_pnl' in portfolio_data:
            self.today_pnl = Decimal(str(portfolio_data['today_pnl']))
    
    async def _handle_risk_check(self, message: Message):
        """Handle a risk check request."""
        try:
            check_type = message.payload.get('check_type')
            data = message.payload.get('data', {})
            
            if check_type == 'order':
                assessment = await self.assess_order_risk(data)
            elif check_type == 'position':
                assessment = await self.assess_position_risk(data)
            elif check_type == 'portfolio':
                assessment = await self.assess_portfolio_risk()
            else:
                assessment = RiskAssessment(
                    is_approved=False,
                    risk_score=1.0,
                    risk_level=RiskLevel.VERY_HIGH,
                    reasons=[f'Unknown risk check type: {check_type}'],
                    suggested_actions=['Contact support']
                )
            
            # Send response
            await self.send_message(
                Message(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type=MessageType.RISK_ASSESSMENT,
                    payload={
                        'request_id': message.payload.get('request_id'),
                        'assessment': {
                            'is_approved': assessment.is_approved,
                            'risk_score': float(assessment.risk_score),
                            'risk_level': assessment.risk_level.name,
                            'reasons': assessment.reasons,
                            'suggested_actions': assessment.suggested_actions,
                            'max_position_size': float(assessment.max_position_size) if assessment.max_position_size else None,
                            'max_quantity': float(assessment.max_quantity) if assessment.max_quantity else None,
                            'metadata': assessment.metadata
                        }
                    },
                    in_reply_to=message.message_id
                )
            )
        
        except Exception as e:
            logger.error(f"Error processing risk check: {e}", exc_info=True)
            await self._send_error(
                f"Error processing risk check: {str(e)}",
                message.sender_id,
                message.message_id
            )
    
    async def _handle_market_data(self, market_data: Dict[str, Any]):
        """Handle incoming market data updates."""
        # In a real implementation, this would update internal market state
        # used for risk calculations (e.g., liquidity, volatility)
        pass
    
    async def assess_order_risk(self, order: Dict[str, Any]) -> RiskAssessment:
        """Assess the risk of a potential order."""
        symbol = order.get('symbol')
        quantity = Decimal(str(order.get('quantity', '0')))
        side = order.get('side')
        order_type = order.get('order_type', 'market')
        
        # Initialize assessment
        reasons = []
        suggested_actions = []
        risk_score = 0.0
        is_approved = True
        
        # Get current price for calculations
        current_price = await self._get_current_price(symbol)
        if current_price is None:
            return RiskAssessment(
                is_approved=False,
                risk_score=1.0,
                risk_level=RiskLevel.VERY_HIGH,
                reasons=['Unable to get current price for risk assessment'],
                suggested_actions=['Check market data service']
            )
        
        # Calculate position value
        position_value = quantity * current_price
        
        # Check position size limit
        max_position_value = self.portfolio_value * Decimal(str(self.risk_params.max_position_size_pct))
        if position_value > max_position_value:
            is_approved = False
            risk_score = max(risk_score, 0.8)
            reasons.append(
                f"Position size {position_value:.2f} exceeds maximum allowed {max_position_value:.2f} "
                f"({self.risk_params.max_position_size_pct*100:.1f}% of portfolio)"
            )
            suggested_actions.append(
                f"Reduce position size to {max_position_value/current_price:.8f} {symbol} or less"
            )
        
        # Check daily loss limit
        if self.today_pnl < -self.portfolio_value * Decimal(str(self.risk_params.daily_loss_limit_pct)):
            is_approved = False
            risk_score = max(risk_score, 0.9)
            reasons.append(
                f"Daily PnL {self.today_pnl:.2f} exceeds daily loss limit "
                f"{self.portfolio_value * Decimal(str(self.risk_params.daily_loss_limit_pct)):.2f}"
            )
            suggested_actions.append("Wait until next trading day or adjust risk parameters")
        
        # Check liquidity (simplified example)
        liquidity = await self._get_market_liquidity(symbol)
        if liquidity and position_value > liquidity * Decimal('0.1'):  # Don't take more than 10% of daily volume
            risk_score = max(risk_score, 0.7)
            reasons.append(
                f"Order size {position_value:.2f} is large relative to market liquidity {liquidity:.2f}"
            )
            suggested_actions.append("Consider splitting order into smaller chunks")
        
        # Check concentration risk
        position = self.positions.get(symbol, {'quantity': 0})
        new_position_value = (abs(Decimal(str(position.get('quantity', 0))) + quantity) * current_price)
        if new_position_value > self.portfolio_value * Decimal(str(self.risk_params.max_concentration_pct)):
            risk_score = max(risk_score, 0.8)
            reasons.append(
                f"Position would exceed maximum concentration of "
                f"{self.risk_params.max_concentration_pct*100:.1f}% of portfolio"
            )
            suggested_actions.append("Reduce position size or increase portfolio diversification")
        
        # Determine risk level based on score
        risk_level = self._calculate_risk_level(risk_score)
        
        # Calculate maximum allowed position size
        max_position_size = min(
            max_position_value,
            self.portfolio_value * Decimal(str(self.risk_params.max_concentration_pct))
        )
        
        max_qty = max_position_size / current_price if current_price > 0 else None
        return RiskAssessment(
            is_approved=is_approved,
            risk_score=risk_score,
            risk_level=risk_level,
            reasons=reasons,
            suggested_actions=suggested_actions,
            max_position_size=max_position_size,
            max_quantity=max_qty,
            adjusted_quantity=min(quantity, max_qty) if max_qty and not is_approved else quantity,
            metadata={
                'position_value': float(position_value),
                'current_price': float(current_price),
                'portfolio_value': float(self.portfolio_value)
            }
        )
    
    async def assess_position_risk(self, position: Dict[str, Any]) -> RiskAssessment:
        """Assess the risk of an existing position."""
        # Similar to assess_order_risk but for existing positions
        # Implementation would check drawdown, margin requirements, etc.
        return RiskAssessment(
            is_approved=True,
            risk_score=0.0,
            risk_level=RiskLevel.VERY_LOW,
            reasons=[],
            suggested_actions=[]
        )
    
    async def assess_portfolio_risk(self) -> RiskAssessment:
        """Assess overall portfolio risk."""
        # Check portfolio-level risks like correlation, beta, drawdown, etc.
        return RiskAssessment(
            is_approved=True,
            risk_score=0.0,
            risk_level=RiskLevel.VERY_LOW,
            reasons=[],
            suggested_actions=[]
        )
    
    async def _is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        # Implement trading hours, system status checks, etc.
        return True
    
    async def _get_current_price(self, symbol: str) -> Optional[Decimal]:
        """Get the current price for a symbol."""
        # In a real implementation, this would fetch from market data
        return Decimal('50000')  # Example price
    
    async def _get_market_liquidity(self, symbol: str) -> Optional[Decimal]:
        """Get the current market liquidity for a symbol."""
        # In a real implementation, this would fetch from market data
        return Decimal('1000000')  # Example liquidity
    
    def _calculate_risk_level(self, risk_score: float) -> RiskLevel:
        """Convert a risk score to a RiskLevel."""
        if risk_score >= 0.8:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MODERATE
        elif risk_score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def _update_pnl(self, pnl: Decimal):
        """Update PnL tracking."""
        self.today_pnl += pnl
        self.historical_pnl[datetime.utcnow()] = self.today_pnl
        
        # Keep only the last 24 hours of PnL data
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.historical_pnl = {
            k: v for k, v in self.historical_pnl.items() if k >= cutoff
        }
    
    async def _reject_order(
        self,
        order: Dict[str, Any],
        recipient_id: str,
        original_message_id: str,
        reason: str,
        assessment: Optional[RiskAssessment] = None
    ):
        """Send an order rejection message."""
        rejection = {
            'order': order,
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if assessment:
            rejection['assessment'] = {
                'risk_score': float(assessment.risk_score),
                'risk_level': assessment.risk_level.name,
                'reasons': assessment.reasons,
                'suggested_actions': assessment.suggested_actions
            }
        
        await self.send_message(
            Message(
                sender_id=self.agent_id,
                recipient_id=recipient_id,
                message_type=MessageType.ORDER_REJECTED,
                payload=rejection,
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

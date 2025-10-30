"""Base agent class for the multi-agent trading system.

This module provides the BaseAgent class that all other agents will inherit from,
and the Message class for inter-agent communication.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Generic, Type, Callable, Awaitable
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_system.log')
    ]
)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages that can be sent between agents."""
    DATA_UPDATE = "data_update"
    MARKET_DATA = "market_data"
    MARKET_SIGNAL = "market_signal"
    NEW_SIGNAL = "new_signal"
    TRADE_SIGNAL = "trade_signal"
    RISK_ASSESSMENT = "risk_assessment"
    ORDER_REQUEST = "order_request"
    ORDER_EXECUTED = "order_executed"
    ORDER_REJECTED = "order_rejected"
    ORDER_UPDATE = "order_update"
    PORTFOLIO_UPDATE = "portfolio_update"
    ANALYZE_NEWS = "analyze_news"
    ANALYZE_SENTIMENT = "analyze_sentiment"
    GENERATE_INDICATORS = "generate_indicators"
    ANALYSIS_RESULT = "analysis_result"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    CONFIG_UPDATE = "config_update"
    POSITION_UPDATE = "position_update"
    PROCESS_MARKET_DATA = "process_market_data"
    RISK_CHECK = "risk_check"
    ORDER_APPROVED = "order_approved"
    CANCEL_ORDER = "cancel_order"
    BACKTEST_REQUEST = "backtest_request"
    ORDER_STATUS_REQUEST = "order_status_request"

@dataclass
class Message:
    """Message class for inter-agent communication."""
    msg_type: MessageType
    sender: str
    recipients: List[str]
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            'msg_type': self.msg_type.value,
            'sender': self.sender,
            'recipients': self.recipients,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat(),
            'msg_id': self.msg_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a Message from a dictionary."""
        return cls(
            msg_type=MessageType(data['msg_type']),
            sender=data['sender'],
            recipients=data['recipients'],
            payload=data['payload'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            msg_id=data.get('msg_id', str(uuid.uuid4()))
        )

class AgentStatus(Enum):
    """Agent status enum."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

class BaseAgent:
    """Base class for all agents in the trading system.
    
    This class provides common functionality for all agents, including message
    handling, lifecycle management, and logging.
    """
    
    def __init__(self, agent_id: str, message_bus: 'MessageBus' = None):
        """Initialize the agent.
        
        Args:
            agent_id: Unique identifier for this agent
            message_bus: Message bus for inter-agent communication
        """
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.status = AgentStatus.STOPPED
        self._handlers = {}
        self._stop_event = asyncio.Event()
        self._tasks = set()
        
        # Register default handlers
        self.register_handler(MessageType.HEARTBEAT, self._handle_heartbeat)
        self.register_handler(MessageType.ERROR, self._handle_error)
    
    def register_handler(self, msg_type: MessageType, handler: Callable[['Message'], Awaitable[None]]):
        """Register a message handler for a specific message type.
        
        Args:
            msg_type: The type of message to handle
            handler: Async function that takes a Message and returns None
        """
        if msg_type not in self._handlers:
            self._handlers[msg_type] = []
        self._handlers[msg_type].append(handler)
    
    async def start(self):
        """Start the agent."""
        if self.status == AgentStatus.RUNNING:
            return
            
        self.status = AgentStatus.STARTING
        logger.info(f"Starting agent {self.agent_id}")
        
        # Start the message processing task
        self._process_task = asyncio.create_task(self._process_messages())
        
        # Subscribe to message types if the agent has a message bus
        if self.message_bus and hasattr(self, 'message_types') and self.message_types:
            self.message_bus.subscribe(self.agent_id, self.message_types)
        
        # Give the event loop a chance to start the task
        await asyncio.sleep(0)
            
        self.status = AgentStatus.RUNNING
        logger.info(f"Agent {self.agent_id} started successfully")
            
    async def stop(self):
        """Stop the agent gracefully."""
        if self.status != AgentStatus.RUNNING:
            logger.warning(f"Agent {self.agent_id} is not running")
            return
            
        self.status = AgentStatus.STOPPING
        logger.info(f"Stopping agent: {self.agent_id}")
        
        try:
            # Signal the run loop to stop
            self._stop_event.set()
            
            # Cancel all running tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._tasks:
                await asyncio.wait(self._tasks, timeout=5.0)
            
            # Call the subclass cleanup
            await self.on_stop()
            
            self.status = AgentStatus.STOPPED
            logger.info(f"Agent {self.agent_id} stopped successfully")
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Error stopping agent {self.agent_id}: {e}", exc_info=True)
            raise
    
    async def _process_messages(self):
        """Process incoming messages."""
        try:
            while self.status in (AgentStatus.RUNNING, AgentStatus.STARTING):
                try:
                    message = await self.message_bus.get_message(self.agent_id)
                    if message:
                        if message.msg_type == MessageType.ERROR:
                            await self.handle_error(None, message)
                        else:
                            await self.handle_message(message)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    try:
                        await self.handle_error(e, message)
                    except Exception as err:
                        logger.error(f"Error in error handler: {err}", exc_info=True)
                await asyncio.sleep(0.001)  # Small sleep to prevent busy waiting
        except asyncio.CancelledError:
            logger.info(f"Message processing cancelled for agent {self.agent_id}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in message processing: {e}", exc_info=True)
            self.status = AgentStatus.ERROR
            raise
    
    async def handle_message(self, message: Message) -> bool:
        """Handle an incoming message.
        
        Args:
            message: The message to handle
            
        Returns:
            bool: True if the message was handled successfully, False otherwise
        """
        logger.debug(f"Agent {self.agent_id} received message: {message}")
        # Default implementation just logs the message
        return True
    
    async def handle_error(self, error: Exception, message: Message):
        """Handle an error message.
        
        Args:
            error: The error that occurred
            message: The message that caused the error
        """
        logger.error(f"Error from {message.sender}: {message.payload.get('error')}")
    
    async def send_message(self, message: Message):
        """Send a message to the message bus."""
        if not self.message_bus:
            logger.warning("No message bus configured, message not sent")
            return
            
        try:
            await self.message_bus.publish(message)
            logger.debug(f"Sent message: {message.msg_type} to {message.recipients}")
        except Exception as e:
            logger.error(f"Error sending message: {e}", exc_info=True)
    
    async def _handle_heartbeat(self, message: Message):
        """Handle heartbeat messages."""
        logger.debug(f"Heartbeat received from {message.sender}")
    
    async def _handle_error(self, message: Message):
        """Handle error messages."""
        logger.error(f"Error from {message.sender}: {message.payload.get('error')}")
    
    # Methods to be overridden by subclasses
    async def on_start(self):
        """Called when the agent is starting up."""
        pass
    
    async def on_stop(self):
        """Called when the agent is shutting down."""
        pass

class MessageBus:
    """Simple in-memory message bus for inter-agent communication."""
    
    def __init__(self):
        self._queues = {}  # agent_id -> asyncio.Queue
        self._agent_subscriptions = {}  # agent_id -> set(MessageType)
        self._type_subscribers = {}  # MessageType -> set(agent_id)
    
    async def publish(self, message: Message):
        """Publish a message to all subscribers."""
        # Always send to explicitly mentioned recipients
        for recipient in message.recipients:
            if recipient in self._queues:
                await self._queues[recipient].put(message)
        
        # If no recipients specified, send to all subscribers of this message type
        if not message.recipients and message.msg_type in self._type_subscribers:
            for agent_id in self._type_subscribers[message.msg_type]:
                if agent_id in self._queues:
                    await self._queues[agent_id].put(message)
    
    async def get_message(self, agent_id: str) -> Optional[Message]:
        """Get the next message for the specified agent."""
        if agent_id not in self._queues:
            self._queues[agent_id] = asyncio.Queue()
            self._agent_subscriptions[agent_id] = set()
        return await self._queues[agent_id].get()
    
    def subscribe(self, agent_id: str, message_types: List[MessageType]):
        """Subscribe an agent to specific message types."""
        if agent_id not in self._queues:
            self._queues[agent_id] = asyncio.Queue()
            self._agent_subscriptions[agent_id] = set()
        
        for msg_type in message_types:
            # Add to agent's subscriptions
            self._agent_subscriptions[agent_id].add(msg_type)
            
            # Add to type subscribers
            if msg_type not in self._type_subscribers:
                self._type_subscribers[msg_type] = set()
            self._type_subscribers[msg_type].add(agent_id)
    
    def unsubscribe(self, agent_id: str, message_types: List[MessageType]):
        """Unsubscribe an agent from specific message types."""
        if agent_id not in self._agent_subscriptions:
            return
            
        for msg_type in message_types:
            # Remove from agent's subscriptions
            if msg_type in self._agent_subscriptions[agent_id]:
                self._agent_subscriptions[agent_id].remove(msg_type)
            
            # Remove from type subscribers
            if msg_type in self._type_subscribers and agent_id in self._type_subscribers[msg_type]:
                self._type_subscribers[msg_type].remove(agent_id)
                if not self._type_subscribers[msg_type]:
                    del self._type_subscribers[msg_type]
                    
    def is_subscribed(self, agent_id: str, message_type: MessageType) -> bool:
        """Check if an agent is subscribed to a specific message type.
        
        Args:
            agent_id: The ID of the agent to check
            message_type: The message type to check
            
        Returns:
            bool: True if the agent is subscribed to the message type, False otherwise
        """
        if agent_id not in self._agent_subscriptions:
            return False
        return message_type in self._agent_subscriptions[agent_id]

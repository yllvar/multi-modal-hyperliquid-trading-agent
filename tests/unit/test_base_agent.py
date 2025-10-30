# tests/unit/test_base_agent.py
import pytest
import asyncio
from unittest.mock import MagicMock, patch
from agents.base_agent import BaseAgent, Message, MessageType, MessageBus, AgentStatus

class TestBaseAgent:
    @pytest.fixture
    def message_bus(self):
        return MessageBus()

    @pytest.fixture
    async def agent(self, message_bus):
        class TestAgent(BaseAgent):
            def __init__(self, message_bus=None):
                super().__init__(agent_id="test_agent", message_bus=message_bus)
                self.handled_messages = []
                self.error_handler_called = False
                self._message_handled = asyncio.Event()
                self.message_types = [MessageType.MARKET_DATA, MessageType.ERROR]

            async def handle_message(self, message):
                self.handled_messages.append(message)
                self._message_handled.set()
                return True
                
            async def handle_error(self, error, message):
                self.error_handler_called = True
                if message and message.msg_type == MessageType.ERROR:
                    self.handled_messages.append(message)
                self._message_handled.set()
                return True
                
            async def wait_for_message(self, timeout=0.5):
                try:
                    await asyncio.wait_for(self._message_handled.wait(), timeout=timeout)
                    self._message_handled.clear()
                    return True
                except asyncio.TimeoutError:
                    return False
        return TestAgent(message_bus=message_bus)

    @pytest.mark.asyncio
    async def test_start_stop(self, agent):
        await agent.start()
        assert agent.status == AgentStatus.RUNNING
        await agent.stop()
        assert agent.status == AgentStatus.STOPPED

    @pytest.mark.asyncio
    async def test_message_handling(self, agent, message_bus):
        # Start the agent to ensure it's processing messages
        await agent.start()
        
        # Give the agent time to start processing
        await asyncio.sleep(0.1)
        
        # Create a test message
        test_message = Message(
            msg_type=MessageType.MARKET_DATA,
            sender="test_sender",
            recipients=[agent.agent_id],
            payload={"test": "data"}
        )
        
        # Publish the message
        await message_bus.publish(test_message)
        
        # Wait for the message to be handled
        message_received = await agent.wait_for_message()
        
        # Verify the message was handled
        assert message_received, "Message was not received within timeout"
        assert len(agent.handled_messages) == 1, f"Expected 1 message, got {len(agent.handled_messages)}"
        assert agent.handled_messages[0].payload == {"test": "data"}
        
        # Clean up
        await agent.stop()

    @pytest.mark.asyncio
    async def test_error_handling(self, agent, message_bus):
        # Start the agent to ensure it's processing messages
        await agent.start()
        
        # Give the agent time to start processing
        await asyncio.sleep(0.1)
        
        # Create an error message
        error_message = Message(
            msg_type=MessageType.ERROR,
            sender="test_sender",
            recipients=[agent.agent_id],
            payload={"error": "test error"}
        )
        
        # Publish the error message
        await message_bus.publish(error_message)
        
        # Wait for the error to be handled
        error_handled = await agent.wait_for_message()
        
        # Verify the error handler was called
        assert error_handled, "Error was not handled within timeout"
        assert agent.error_handler_called is True, "Error handler was not called"
        assert len(agent.handled_messages) == 1, "Error message was not processed"
        
        # Clean up
        await agent.stop()
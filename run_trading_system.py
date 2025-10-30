"""
AI Trading System - Main Entry Point

This script demonstrates how to initialize and run the AI Trading System
with all its components working together.
"""

import asyncio
import logging
import signal
import sys
from decimal import Decimal
from typing import Dict, Any

from agents.analysis_agent.agent import AnalysisAgent
from agents.trading_agent.agent import TradingAgent, OrderSide, OrderType
from agents.risk_agent.agent import RiskAgent, RiskParameters
from agents.execution_agent.agent import ExecutionAgent, ExecutionParameters
from agents.data_agent.data_fetcher import DataFetcher
from agents.base_agent import MessageBus

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

class AITradingSystem:
    """Main class for the AI Trading System."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trading system with configuration."""
        self.config = config
        self.message_bus = MessageBus()
        self.running = False
        self.agents = {}
        self.data_fetcher = None
        
    async def initialize(self):
        """Initialize all components of the trading system."""
        logger.info("Initializing AI Trading System...")
        
        # Initialize data fetcher
        self.data_fetcher = DataFetcher(
            exchange_id='binance',
            api_key=self.config.get('exchange_api_key'),
            api_secret=self.config.get('exchange_api_secret'),
            message_bus=self.message_bus
        )
        
        # Initialize risk agent
        risk_params = RiskParameters(
            max_position_size_pct=0.1,  # 10% of portfolio
            max_risk_per_trade_pct=0.02,  # 2% risk per trade
            daily_loss_limit_pct=0.05,  # 5% daily loss limit
            max_leverage=3.0,
            max_drawdown_pct=0.1,  # 10% max drawdown
            min_liquidity_usd=1000000,  # $1M minimum liquidity
            max_concentration_pct=0.3  # 30% max position concentration
        )
        self.agents['risk'] = RiskAgent(
            risk_parameters=risk_params,
            message_bus=self.message_bus
        )
        
        # Initialize execution agent
        exec_params = ExecutionParameters(
            slippage=0.001,  # 0.1% slippage
            max_slippage=0.01,  # 1% max slippage
            max_retries=3,
            retry_delay=0.1,
            use_vwap=True,
            vwap_window=5,  # 5 minutes
            min_order_size=10.0,  # $10 minimum
            max_order_size=100000.0  # $100k maximum
        )
        self.agents['execution'] = ExecutionAgent(
            execution_params=exec_params,
            message_bus=self.message_bus
        )
        
        # Initialize analysis agent
        self.agents['analysis'] = AnalysisAgent(
            message_bus=self.message_bus,
            model_name=self.config.get('ai_model', 'togethercomputer/llama-2-70b-chat'),
            api_key=self.config.get('together_ai_api_key')
        )
        
        # Initialize trading agent (should be last as it may depend on other agents)
        self.agents['trading'] = TradingAgent(
            initial_balance={'USDT': self.config.get('initial_balance', 10000)},
            max_position_size=0.1,  # 10% of portfolio
            max_risk_per_trade=0.02,  # 2% risk per trade
            default_slippage=0.001,  # 0.1%
            message_bus=self.message_bus
        )
        
        logger.info("All components initialized")
    
    async def start(self):
        """Start the trading system."""
        if self.running:
            logger.warning("Trading system is already running")
            return
            
        logger.info("Starting AI Trading System...")
        self.running = True
        
        # Start all agents
        tasks = [agent.start() for agent in self.agents.values()]
        tasks.append(self.data_fetcher.start())
        
        # Start the main event loop
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Shutdown signal received")
        except Exception as e:
            logger.exception("Error in trading system")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the trading system gracefully."""
        if not self.running:
            return
            
        logger.info("Stopping AI Trading System...")
        self.running = False
        
        # Stop all agents
        for agent in self.agents.values():
            try:
                await agent.stop()
            except Exception as e:
                logger.error(f"Error stopping agent {agent.__class__.__name__}: {e}")
        
        # Stop data fetcher
        if self.data_fetcher:
            try:
                await self.data_fetcher.stop()
            except Exception as e:
                logger.error(f"Error stopping data fetcher: {e}")
        
        logger.info("AI Trading System stopped")

async def main():
    """Main entry point for the trading system."""
    # Configuration
    config = {
        'initial_balance': 10000,  # USDT
        'ai_model': 'togethercomputer/llama-2-70b-chat',
        'exchange_api_key': None,  # Set your exchange API key here
        'exchange_api_secret': None,  # Set your exchange API secret here
        'together_ai_api_key': None,  # Set your Together.ai API key here
        'symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    }
    
    # Initialize trading system
    trading_system = AITradingSystem(config)
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(trading_system.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize and start the trading system
        await trading_system.initialize()
        await trading_system.start()
    except Exception as e:
        logger.exception("Fatal error in trading system")
        sys.exit(1)
    finally:
        # Ensure clean shutdown
        await trading_system.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.exception("Unhandled exception in main")
        sys.exit(1)

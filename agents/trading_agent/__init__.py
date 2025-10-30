"""
Trading Agent package for the AI Trading System.

This package contains the trading agent responsible for executing trading strategies
based on market data and analysis from other agents.
"""

from .agent import TradingAgent, TradingStrategy, Order, Position

__all__ = [
    'TradingAgent',
    'TradingStrategy',
    'Order',
    'Position'
]

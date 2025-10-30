"""
Execution Agent package for the AI Trading System.

This package contains the execution agent responsible for order routing,
optimization, and backtesting of trading strategies.
"""

from .agent import ExecutionAgent, ExecutionParameters, ExecutionResult, BacktestResult

__all__ = [
    'ExecutionAgent',
    'ExecutionParameters',
    'ExecutionResult',
    'BacktestResult'
]

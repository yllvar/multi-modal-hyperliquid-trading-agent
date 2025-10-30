"""
Data Agent package for the AI Trading System.

This package contains the data agent responsible for collecting, processing,
and distributing market data to other components of the trading system.
"""

from .data_fetcher import DataFetcher, MarketData

__all__ = [
    'DataFetcher',
    'MarketData',
]

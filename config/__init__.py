"""
Configuration module for the AI Trading System.

This module provides centralized configuration management for the entire system.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class Config:
    """Centralized configuration management for the trading system."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._config = {}
            self._defaults = {
                # General
                'log_level': 'INFO',
                'environment': 'development',
                
                # Message Bus
                'message_bus': {
                    'max_queue_size': 1000,
                    'timeout': 5.0
                },
                
                # Together.ai
                'together_ai': {
                    'api_key': None,
                    'base_url': 'https://api.together.xyz',
                    'default_model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                    'timeout': 30.0,
                    'max_retries': 3,
                    'retry_delay': 1.0
                },
                
                # Hyperliquid
                'hyperliquid': {
                    'api_url': 'https://api.hyperliquid.xyz',
                    'ws_url': 'wss://api.hyperliquid.xyz/ws',
                    'private_key': None,
                    'testnet': True,
                    'max_leverage': 10.0,
                    'default_slippage': 0.005  # 0.5%
                },
                
                # Data
                'data': {
                    'cache_dir': 'data/cache',
                    'max_cache_age': 3600,  # 1 hour
                    'tickers': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
                    'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d']
                },
                
                # Trading
                'trading': {
                    'max_position_size': 0.1,  # 10% of portfolio
                    'default_stop_loss': 0.02,  # 2%
                    'default_take_profit': 0.04,  # 4%
                    'max_daily_trades': 10,
                    'max_drawdown': 0.05  # 5%
                },
                
                # Risk Management
                'risk': {
                    'max_portfolio_risk': 0.02,  # 2% per trade
                    'max_daily_drawdown': 0.05,  # 5%
                    'max_position_risk': 0.1,  # 10% per position
                    'volatility_window': 20,  # 20 periods
                    'correlation_threshold': 0.7
                },
                
                # Backtesting
                'backtesting': {
                    'initial_balance': 10000.0,
                    'commission': 0.0005,  # 0.05%
                    'slippage': 0.0005,  # 0.05%
                    'start_date': '2023-01-01',
                    'end_date': '2023-12-31'
                }
            }
            
            # Load environment variables
            self._load_environment()
            
            # Load config file if exists
            self._load_config_file()
            
            # Apply overrides from environment variables
            self._apply_environment_overrides()
            
            # Ensure required directories exist
            self._ensure_directories()
            
            self._initialized = True
    
    def _load_environment(self):
        """Load configuration from environment variables."""
        # General
        self._config['log_level'] = os.getenv('LOG_LEVEL', self._defaults['log_level'])
        self._config['environment'] = os.getenv('ENVIRONMENT', self._defaults['environment'])
        
        # Together.ai
        self._config['together_ai'] = {
            'api_key': os.getenv('TOGETHER_AI_API_KEY', self._defaults['together_ai']['api_key']),
            'base_url': os.getenv('TOGETHER_AI_BASE_URL', self._defaults['together_ai']['base_url']),
            'default_model': os.getenv('TOGETHER_AI_DEFAULT_MODEL', self._defaults['together_ai']['default_model']),
            'timeout': float(os.getenv('TOGETHER_AI_TIMEOUT', self._defaults['together_ai']['timeout'])),
            'max_retries': int(os.getenv('TOGETHER_AI_MAX_RETRIES', self._defaults['together_ai']['max_retries'])),
            'retry_delay': float(os.getenv('TOGETHER_AI_RETRY_DELAY', self._defaults['together_ai']['retry_delay']))
        }
        
        # Hyperliquid
        self._config['hyperliquid'] = {
            'api_url': os.getenv('HYPERLIQUID_API_URL', self._defaults['hyperliquid']['api_url']),
            'ws_url': os.getenv('HYPERLIQUID_WS_URL', self._defaults['hyperliquid']['ws_url']),
            'private_key': os.getenv('HYPERLIQUID_PRIVATE_KEY', self._defaults['hyperliquid']['private_key']),
            'testnet': os.getenv('HYPERLIQUID_TESTNET', str(self._defaults['hyperliquid']['testnet'])).lower() == 'true',
            'max_leverage': float(os.getenv('HYPERLIQUID_MAX_LEVERAGE', self._defaults['hyperliquid']['max_leverage'])),
            'default_slippage': float(os.getenv('HYPERLIQUID_DEFAULT_SLIPPAGE', self._defaults['hyperliquid']['default_slippage']))
        }
        
        # Data
        self._config['data'] = {
            'cache_dir': os.getenv('CACHE_DIR', self._defaults['data']['cache_dir']),
            'max_cache_age': int(os.getenv('MAX_CACHE_AGE', self._defaults['data']['max_cache_age'])),
            'tickers': os.getenv('TICKERS', ','.join(self._defaults['data']['tickers'])).split(','),
            'timeframes': os.getenv('TIMEFRAMES', ','.join(self._defaults['data']['timeframes'])).split(',')
        }
    
    def _load_config_file(self):
        """Load configuration from a JSON file if it exists."""
        config_path = os.getenv('CONFIG_FILE', 'config/config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    self._merge_config(file_config)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
    
    def _merge_config(self, new_config: Dict[str, Any], target: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries."""
        if target is None:
            target = self._config
            
        for key, value in new_config.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(value, target[key])
            else:
                target[key] = value
        
        return target
    
    def _apply_environment_overrides(self):
        """Apply configuration overrides from environment variables."""
        # This method can be expanded to handle more complex overrides
        pass
    
    def _ensure_directories(self):
        """Ensure that all required directories exist."""
        os.makedirs(self._config['data']['cache_dir'], exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot notation key."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set a configuration value by dot notation key."""
        keys = key.split('.')
        current = self._config
        
        for i, k in enumerate(keys[:-1]):
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the current configuration as a dictionary."""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting of configuration."""
        self.set(key, value)

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance."""
    return config

# tests/conftest.py
import os
import sys
from unittest.mock import MagicMock, patch

# Mock the config module before anything else imports it
sys.modules['config'] = MagicMock()
from config import config

# Configure the mock
config._config = {
    'data': {
        'cache_dir': '/tmp/ai-trading-agent/cache',
        'data_dir': '/tmp/ai-trading-agent/data'
    },
    'together_ai': {
        'api_key': 'test_api_key',
        'model': 'test_model'
    }
}

# Ensure the cache directory exists
os.makedirs('/tmp/ai-trading-agent/cache', exist_ok=True)
os.makedirs('/tmp/ai-trading-agent/data', exist_ok=True)

# Now import other modules that depend on config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
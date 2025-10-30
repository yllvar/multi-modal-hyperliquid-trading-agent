#!/usr/bin/env python3
"""
Script to fetch all available trading symbols from Hyperliquid and generate a symbols configuration file.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any
from agents.data_agent.data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the config directory and symbols file
CONFIG_DIR = Path(__file__).parent / 'config'
SYMBOLS_FILE = CONFIG_DIR / 'symbols.json'

def create_default_symbol_config(symbol: str) -> Dict[str, Any]:
    """Create a default configuration for a symbol."""
    return {
        "enabled": True,
        "min_volume": 100000,  # Default minimum 24h volume
        "max_volume": 10000000,  # Default maximum 24h volume
        "notes": ""  # Optional notes about the symbol
    }

async def generate_symbols_config():
    """Fetch all available symbols and generate a configuration file."""
    fetcher = DataFetcher()
    
    try:
        # Initialize the fetcher
        await fetcher.start()
        logger.info("Successfully connected to Hyperliquid API")
        
        # Fetch all available symbols
        logger.info("Fetching available symbols...")
        symbols = await fetcher.fetch_available_symbols()
        
        # Create config directory if it doesn't exist
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Check if config file already exists
        existing_config = {}
        if SYMBOLS_FILE.exists():
            try:
                with open(SYMBOLS_FILE, 'r') as f:
                    existing_config = json.load(f)
                logger.info(f"Loaded existing configuration from {SYMBOLS_FILE}")
            except json.JSONDecodeError:
                logger.warning(f"Existing config file is invalid, will create a new one")
        
        # Get existing symbols configuration or create new ones
        symbols_config = existing_config.get('symbols', {})
        
        # Add any new symbols that don't exist in the config
        new_symbols = 0
        for symbol in symbols:
            if symbol not in symbols_config:
                symbols_config[symbol] = create_default_symbol_config(symbol)
                new_symbols += 1
        
        # Create the complete config
        config = {
            "version": "1.0",
            "last_updated": str(Path(__file__).name),
            "symbols": {k: symbols_config[k] for k in sorted(symbols_config.keys())},
            "default_settings": {
                "min_volume": 100000,
                "max_volume": 10000000,
                "enabled": True
            }
        }
        
        # Save the configuration
        with open(SYMBOLS_FILE, 'w') as f:
            json.dump(config, f, indent=2)
            f.write('\n')  # Add trailing newline for better git diffs
        
        logger.info(f"Successfully saved configuration to {SYMBOLS_FILE}")
        logger.info(f"Total symbols: {len(symbols_config)}")
        logger.info(f"New symbols added: {new_symbols}")
        
        return config
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    finally:
        # Ensure the fetcher is properly closed
        await fetcher.stop()

if __name__ == "__main__":
    asyncio.run(generate_symbols_config())
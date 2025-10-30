# validation.py
from jsonschema import validate, ValidationError
from pathlib import Path
import json
import sys

# Define the schema for the symbols configuration
SCHEMA = {
    "type": "object",
    "required": ["version", "symbols", "default_settings"],
    "properties": {
        "version": {"type": "string"},
        "last_updated": {"type": "string"},
        "symbols": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["enabled", "min_volume", "max_volume"],
                "properties": {
                    "enabled": {"type": "boolean"},
                    "min_volume": {"type": "number", "minimum": 0},
                    "max_volume": {"type": "number", "minimum": 0},
                    "notes": {"type": "string"},
                    "24h_volume": {"type": "number", "minimum": 0}  # Added for the volume data we collect
                }
            }
        },
        "default_settings": {
            "type": "object",
            "required": ["min_volume", "max_volume", "enabled"],
            "properties": {
                "min_volume": {"type": "number", "minimum": 0},
                "max_volume": {"type": "number", "minimum": 0},
                "enabled": {"type": "boolean"}
            }
        }
    }
}

def validate_config(file_path: str) -> bool:
    """
    Validate a JSON configuration file against the schema.
    
    Args:
        file_path: Path to the JSON file to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {file_path}: {str(e)}")
        return False
    
    try:
        validate(instance=config, schema=SCHEMA)
        print(f"✅ Configuration is valid: {file_path}")
        
        # Additional validation: Check if min_volume <= max_volume for each symbol
        for symbol, settings in config.get('symbols', {}).items():
            if settings.get('min_volume', 0) > settings.get('max_volume', float('inf')):
                print(f"⚠️  Warning: {symbol} has min_volume > max_volume")
                
        return True
        
    except ValidationError as e:
        print(f"❌ Configuration is invalid: {file_path}")
        print(f"Error: {e.message}")
        print(f"Path: {' -> '.join(map(str, e.path)) if e.path else 'root'}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate symbols configuration')
    parser.add_argument('files', nargs='+', help='Configuration files to validate')
    parser.add_argument('--strict', action='store_true', help='Treat warnings as errors')
    
    args = parser.parse_args()
    
    all_valid = True
    for file_path in args.files:
        if not validate_config(file_path) and not args.strict:
            all_valid = False
    
    sys.exit(0 if all_valid else 1)

if __name__ == "__main__":
    main()
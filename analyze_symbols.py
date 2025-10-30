# analyze_symbols.py
import json
from pathlib import Path
import requests
from collections import defaultdict
import statistics

def analyze_symbols():
    # Load the generated config
    config_path = Path('config/symbols.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Get volume data from a market data API
    # Note: You'll need to implement this based on your data source
    def get_24h_volume(symbol):
        # Example using CoinGecko API (you'll need to install requests)
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}/tickers"
            response = requests.get(url)
            data = response.json()
            # Extract 24h volume in USD
            return float(data.get('tickers', [{}])[0].get('volume', 0))
        except:
            return 0

    # Analyze volumes
    volumes = []
    volume_by_asset = {}

    for symbol in config['symbols']:
        volume = get_24h_volume(symbol)
        volume_by_asset[symbol] = volume
        if volume > 0:  # Only include assets with volume data
            volumes.append(volume)

    # Calculate statistics
    if volumes:
        avg_volume = statistics.mean(volumes)
        median_volume = statistics.median(volumes)
        min_volume = min(volumes)
        max_volume = max(volumes)

        print(f"Volume Statistics (USD):")
        print(f"  Average: ${avg_volume:,.2f}")
        print(f"  Median:  ${median_volume:,.2f}")
        print(f"  Min:     ${min_volume:,.2f}")
        print(f"  Max:     ${max_volume:,.2f}")

        # Suggest thresholds
        print("\nSuggested Volume Filters:")
        print(f"  min_volume: ${max(1000, median_volume * 0.1):,.0f}  # 10% of median volume")
        print(f"  max_volume: ${max_volume * 0.1:,.0f}  # 10% of max volume")

    # Update config with actual volumes
    for symbol, volume in volume_by_asset.items():
        if symbol in config['symbols']:
            config['symbols'][symbol]['24h_volume'] = volume

    # Save updated config
    with open('config/symbols_with_volumes.json', 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    analyze_symbols()
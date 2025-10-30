"""Client helper for interacting with the TAAPI technical analysis API."""

import requests
import os
import time
import logging
from src.config_loader import CONFIG


class TAAPIClient:
    """Fetches TA indicators with retry/backoff semantics for resilience."""

    def __init__(self):
        """Initialize TAAPI credentials and base URL."""
        self.api_key = CONFIG["taapi_api_key"]
        self.base_url = "https://api.taapi.io/"

    def _get_with_retry(self, url, params, retries=3, backoff=0.5):
        """Perform a GET request with exponential backoff retry logic."""
        for attempt in range(retries):
            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                if e.response.status_code >= 500 and attempt < retries - 1:
                    wait = backoff * (2 ** attempt)
                    logging.warning(f"TAAPI {e.response.status_code}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise
            except requests.Timeout as e:
                if attempt < retries - 1:
                    wait = backoff * (2 ** attempt)
                    logging.warning(f"TAAPI timeout, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Max retries exceeded")

    def get_indicators(self, asset, interval):
        """Return a curated bundle of intraday indicators for ``asset``."""
        params = {
            "secret": self.api_key,
            "exchange": "binance",
            "symbol": f"{asset}/USDT",
            "interval": interval
        }
        rsi_response = self._get_with_retry(f"{self.base_url}rsi", params)
        macd_response = self._get_with_retry(f"{self.base_url}macd", params)
        sma_response = self._get_with_retry(f"{self.base_url}sma", params)
        ema_response = self._get_with_retry(f"{self.base_url}ema", params)
        bbands_response = self._get_with_retry(f"{self.base_url}bbands", params)
        return {
            "rsi": rsi_response.get("value"),
            "macd": macd_response,
            "sma": sma_response.get("value"),
            "ema": ema_response.get("value"),
            "bbands": bbands_response
        }

    def get_historical_indicator(self, indicator, symbol, interval, results=10, params=None):
        """Fetch historical indicator data with optional overrides."""
        base_params = {
            "secret": self.api_key,
            "exchange": "binance",
            "symbol": symbol,
            "interval": interval,
            "results": results
        }
        if params:
            base_params.update(params)
        response = self._get_with_retry(f"{self.base_url}{indicator}", base_params)
        return response

    def fetch_series(self, indicator: str, symbol: str, interval: str, results: int = 10, params: dict | None = None, value_key: str = "value") -> list:
        """Fetch and normalize a historical indicator series.

        Args:
            indicator: TAAPI indicator slug (e.g. ``"ema"``).
            symbol: Market pair identifier (e.g. ``"BTC/USDT"``).
            interval: Candle interval requested from TAAPI.
            results: Number of datapoints to request.
            params: Additional TAAPI query parameters.
            value_key: Key to extract from the TAAPI response payload.

        Returns:
            List of floats rounded to 4 decimals, or an empty list on error.
        """
        try:
            data = self.get_historical_indicator(indicator, symbol, interval, results=results, params=params)
            if isinstance(data, dict):
                # Simple indicators: {"value": [1,2,3]}
                if value_key in data and isinstance(data[value_key], list):
                    return [round(v, 4) if isinstance(v, (int, float)) else v for v in data[value_key]]
                # Error response
                if "error" in data:
                    import logging
                    logging.error(f"TAAPI error for {indicator} {symbol} {interval}: {data.get('error')}")
                    return []
            return []
        except Exception as e:
            import logging
            logging.error(f"TAAPI fetch_series exception for {indicator}: {e}")
            return []

    def fetch_value(self, indicator: str, symbol: str, interval: str, params: dict | None = None, key: str = "value"):
        """Fetch a single indicator value for the latest candle."""
        try:
            base_params = {
                "secret": self.api_key,
                "exchange": "binance",
                "symbol": symbol,
                "interval": interval
            }
            if params:
                base_params.update(params)
            data = self._get_with_retry(f"{self.base_url}{indicator}", base_params)
            if isinstance(data, dict):
                val = data.get(key)
                return round(val, 4) if isinstance(val, (int, float)) else val
            return None
        except Exception:
            return None

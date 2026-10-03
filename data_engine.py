import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()    # Load API keys / settings from .env if present
except Exception:
    pass

try:
    from config.settings import API_CONFIG
    _YF_MAX_RETRIES = int(API_CONFIG.get("yfinance_max_retries", 3))
    _YF_TIMEOUT = int(API_CONFIG.get("yfinance_timeout", 30))
    _CACHE_UPDATE_INTERVAL = int(API_CONFIG.get("cache_update_interval", 2))
except Exception:
    _YF_MAX_RETRIES = 3
    _YF_TIMEOUT = 30
    _CACHE_UPDATE_INTERVAL = 2

DATA_DIR = "data"


def _history_with_retries(ticker, **kwargs):
    """Fetch ticker.history with retries + small backoff for transient failures."""
    last_err = None
    for attempt in range(1, _YF_MAX_RETRIES + 1):
        try:
            return ticker.history(timeout=_YF_TIMEOUT, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < _YF_MAX_RETRIES:
                import time
                time.sleep(min(2 ** attempt, 5))
    print(f"[{getattr(ticker, 'ticker', '?')}] History fetch failed after "
          f"{_YF_MAX_RETRIES} attempts: {last_err}")
    return None


def fetch_data(symbol: str, period="2y", interval="1d") -> pd.DataFrame:
    """
    Fetches financial data for a given symbol and interval.
    Caches data locally as CSV. If the CSV exists it fetches only the new
    data (since `last_date - cache_update_interval`) and appends it.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Replace unsafe characters for filenames
    safe_symbol = symbol.replace("=", "_").replace("^", "_")
    filepath = os.path.join(DATA_DIR, f"{safe_symbol}_{interval}.csv")

    ticker = yf.Ticker(symbol)

    if os.path.exists(filepath):
        try:
            df_existing = pd.read_csv(filepath, index_col=0, parse_dates=True)
            if len(df_existing) > 0:
                last_date = df_existing.index[-1]
                # Pull a small overlap so the still-updating last candle is
                # refreshed, without re-downloading the whole history.
                start_date = (last_date - timedelta(days=_CACHE_UPDATE_INTERVAL)).strftime('%Y-%m-%d')

                print(f"[{symbol}] Using cache; fetching updates since {start_date}...")
                df_new = _history_with_retries(ticker, start=start_date, interval=interval)
                if df_new is None or len(df_new) == 0:
                    # Cache is fresh enough; return it instead of forcing a download.
                    return df_existing

                # Ensure both indices share the same tz awareness before concatenation.
                if df_new.index.tz is not None and df_existing.index.tz is None:
                    df_existing.index = df_existing.index.tz_localize('UTC')
                elif df_new.index.tz is None and df_existing.index.tz is not None:
                    df_new.index = df_new.index.tz_localize('UTC')

                df_combined = pd.concat([df_existing, df_new])
                # Deduplicate by index, keeping the most recent print.
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                df_combined.sort_index(inplace=True)
                df_combined = validate_ohlcv(df_combined)
                df_combined.to_csv(filepath)
                return df_combined
        except Exception as e:
            print(f"Error reading cache for {symbol}: {e}. Redownloading full history.")

    # Fallback to full download if no cache or cache is corrupted.
    print(f"[{symbol}] Downloading full history ({period}, {interval})...")
    df = _history_with_retries(ticker, period=period, interval=interval)

    if df is None or len(df) == 0:
        print(f"[{symbol}] Warning: No data returned from Yahoo Finance.")
        return pd.DataFrame()

    df = validate_ohlcv(df)
    df.to_csv(filepath)
    return df


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and cleans OHLCV data:
    - Removes duplicate timestamps.
    - Drops zero volume rows if a Volume column exists.
    - Smooths bad prints (>3.5 sigma from a rolling mean) WHILE preserving the
      OHLC invariant Low <= Close <= High so ATR / Bollinger math stays valid.
    """
    if df is None or len(df) == 0:
        return df

    cleaned = df.copy()

    # 1. Remove duplicate index timestamps
    cleaned = cleaned[~cleaned.index.duplicated(keep='last')]
    cleaned.sort_index(inplace=True)

    # 2. Drop zero-volume candles (if a volume column exists)
    if 'Volume' in cleaned.columns:
        cleaned = cleaned[cleaned['Volume'] > 0]

    # 3. Clean bad prints (>3.5 sigma) and restore OHLC integrity.
    if 'Close' in cleaned.columns and len(cleaned) >= 20:
        rolling_mean = cleaned['Close'].rolling(window=20, min_periods=5).mean()
        rolling_std = cleaned['Close'].rolling(window=20, min_periods=5).std().replace(0, 1e-6)
        z_scores = np.abs((cleaned['Close'] - rolling_mean) / rolling_std)
        bad_prints = z_scores > 3.5
        if bad_prints.any():
            clipped_close = rolling_mean[bad_prints].fillna(cleaned['Close'][bad_prints])
            cleaned.loc[bad_prints, 'Close'] = clipped_close
            if 'High' in cleaned.columns:
                cleaned['High'] = cleaned[['Open', 'Close', 'High']].max(axis=1)
            if 'Low' in cleaned.columns:
                cleaned['Low'] = cleaned[['Open', 'Close', 'Low']].min(axis=1)
            cleaned['Close'] = cleaned['Close'].clip(
                lower=cleaned['Low'], upper=cleaned['High'])

    return cleaned


def fetch_intraday_1m(symbol: str, period: str = "7d") -> pd.DataFrame:
    """
    Fetches intraday 1-minute historical data (yfinance limits 1m to ~7 days).
    """
    valid_period = "7d" if period in ["1y", "2y", "max", "1mo", "6mo", "3mo"] else period
    df = fetch_data(symbol, period=valid_period, interval="1m")
    return validate_ohlcv(df)


def compute_latency(df: pd.DataFrame) -> dict:
    """
    Computes latency statistics between consecutive candles.
    """
    if df is None or len(df) < 2:
        return {"avg_latency_sec": 0, "max_latency_sec": 0, "status": "NO DATA"}

    timestamps = pd.to_datetime(df.index)
    deltas = timestamps.to_series().diff().dropna().dt.total_seconds()

    avg_lat = float(deltas.mean()) if len(deltas) > 0 else 0.0
    max_lat = float(deltas.max()) if len(deltas) > 0 else 0.0

    return {
        "avg_latency_sec": round(avg_lat, 2),
        "max_latency_sec": round(max_lat, 2),
        "status": "HEALTHY" if max_lat < (avg_lat * 3.0 + 60) else "LATENCY SPIKE DETECTED"
    }

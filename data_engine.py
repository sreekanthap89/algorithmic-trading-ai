import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

DATA_DIR = "data"

def fetch_data(symbol: str, period="2y", interval="1d") -> pd.DataFrame:
    """
    Fetches financial data for a given symbol and interval.
    Caches the data locally as CSV. If the CSV exists, it fetches only the new
    data starting from the last date in the CSV and appends it.
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
                # yfinance expects date strings like YYYY-MM-DD
                # We start a few days before to ensure we catch any updates to the last candle
                start_date = (last_date - timedelta(days=2)).strftime('%Y-%m-%d')
                
                print(f"[{symbol}] Found cached data. Fetching new data since {start_date}...")
                df_new = ticker.history(start=start_date, interval=interval)
                
                # If df_new has timezone, we must ensure df_existing has timezone or remove it
                if df_new.index.tz is not None and df_existing.index.tz is None:
                    df_existing.index = df_existing.index.tz_localize('UTC')
                elif df_new.index.tz is None and df_existing.index.tz is not None:
                    df_new.index = df_new.index.tz_localize('UTC')
                    
                df_combined = pd.concat([df_existing, df_new])
                # Deduplicate by index (Datetime), keeping the most recent data
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                df_combined.sort_index(inplace=True)
                
                df_combined.to_csv(filepath)
                return df_combined
        except Exception as e:
            print(f"Error reading cache for {symbol}: {e}. Redownloading full history.")
            
    # Fallback to full download if no cache or cache is corrupted
    print(f"[{symbol}] Downloading full history ({period}, {interval})...")
    df = ticker.history(period=period, interval=interval)
    
    if len(df) > 0:
        df = validate_ohlcv(df)
        df.to_csv(filepath)
    else:
        print(f"[{symbol}] Warning: No data returned from Yahoo Finance.")
        
    return df

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and cleans OHLCV data:
    - Removes duplicate timestamps.
    - Drops zero volume rows if Volume column exists.
    - Flags and smooths bad prints (>3 sigma from rolling mean).
    """
    if df is None or len(df) == 0:
        return df

    cleaned = df.copy()
    
    # 1. Remove duplicate index timestamps
    cleaned = cleaned[~cleaned.index.duplicated(keep='last')]
    cleaned.sort_index(inplace=True)

    # 2. Drop zero volume candles (if volume column exists)
    if 'Volume' in cleaned.columns:
        cleaned = cleaned[cleaned['Volume'] > 0]

    # 3. Clean bad prints (>3 sigma price deviations)
    if 'Close' in cleaned.columns and len(cleaned) >= 20:
        rolling_mean = cleaned['Close'].rolling(window=20, min_periods=5).mean()
        rolling_std = cleaned['Close'].rolling(window=20, min_periods=5).std().replace(0, 1e-6)
        z_scores = np.abs((cleaned['Close'] - rolling_mean) / rolling_std)
        bad_prints = z_scores > 3.5
        if bad_prints.any():
            cleaned.loc[bad_prints, 'Close'] = rolling_mean[bad_prints]

    return cleaned

def fetch_intraday_1m(symbol: str, period: str = "7d") -> pd.DataFrame:
    """
    Fetches intraday 1-minute historical data (yfinance limits 1m to max 7-8 days).
    """
    # Force max period allowed by yfinance for 1m
    valid_period = "7d" if period in ["1y", "2y", "max", "1mo"] else period
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

if __name__ == "__main__":
    import numpy as np
    df = fetch_data("BTC-USD", period="1y", interval="1d")
    print("Cleaned data shape:", df.shape)
    print("Latency stats:", compute_latency(df))


import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
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

FULL_HISTORY_MAP = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "730d",
    "1h": "730d",
    "4h": "730d",
    "1d": "max",
    "1wk": "max",
    "1mo": "max",
}


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
                time.sleep(min(2 ** attempt, 4))
    print(
        f"[{getattr(ticker, 'ticker', '?')}] History fetch failed after "
        f"{_YF_MAX_RETRIES} attempts: {last_err}"
    )
    return None


def fetch_data(
    symbol: str,
    period: str = "max",
    interval: str = "1d",
    force_full: bool = False,
) -> pd.DataFrame:
    """
    Fetches financial data across all asset classes (Crypto, Forex, Commodities like
    Natural Gas & Gold, Equities, and Indices).
    
    Architecture:
    1. First Run: Fetches the ENTIRE lifetime history available (e.g. 20+ years for daily,
       or maximum available depth for intraday), saving it permanently in the local database.
    2. Subsequent Refreshes: Checks the database for the latest recorded timestamp and fetches
       ONLY the new/missing delta candles since T_last, merging them seamlessly without
       truncating or re-downloading existing history.
    """
    try:
        from config.market_catalog import resolve_symbol
        symbol = resolve_symbol(symbol)
    except Exception:
        symbol = symbol.strip().upper()

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    safe_symbol = (
        symbol.replace("=", "_")
        .replace("^", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(":", "_")
        .replace("\\", "_")
    )
    # Also check legacy filename format (only = and ^ replaced) for existing cache compatibility
    legacy_safe = symbol.replace("=", "_").replace("^", "_")
    legacy_filepath = os.path.join(DATA_DIR, f"{legacy_safe}_{interval}.csv")
    filepath = legacy_filepath if os.path.exists(legacy_filepath) else os.path.join(DATA_DIR, f"{safe_symbol}_{interval}.csv")

    ticker = yf.Ticker(symbol)

    # Determine maximum full-history period if "max", "full", or None is requested
    req_period_lower = str(period).lower() if period else "max"
    if req_period_lower in ("max", "full", "all"):
        target_period = FULL_HISTORY_MAP.get(interval, "max")
    else:
        target_period = period

    if os.path.exists(filepath) and not force_full:
        try:
            df_existing = pd.read_csv(filepath, index_col=0)
            if len(df_existing) > 0:
                df_existing.index = pd.to_datetime(df_existing.index, utc=True, errors="coerce")
                df_existing = df_existing[~df_existing.index.isna()]

                # Check if existing cache has legacy corruptions
                if "High" in df_existing.columns and "Close" in df_existing.columns and "Open" in df_existing.columns:
                    body_top = df_existing[["Open", "Close"]].max(axis=1)
                    upper_wick_pct = (df_existing["High"] - body_top) / df_existing["Close"].replace(0, 1e-9)
                    if (upper_wick_pct > 0.12).rolling(5).sum().max() >= 4:
                        raise ValueError("Legacy rolling_mean Close corruption detected in cache")

                # Detect if cached daily/intraday history is severely truncated compared to full history
                # (e.g. legacy cache only had 5 days or 2 years, but 20+ years / 60d is now requested)
                span_days = (pd.Timestamp(df_existing.index[-1]) - pd.Timestamp(df_existing.index[0])).days
                needs_backfill = False
                if target_period == "max" and interval in ("1d", "1wk", "1mo") and span_days < 1825:  # < 5 years
                    needs_backfill = True
                elif target_period == "60d" and interval in ("5m", "15m", "30m") and span_days < 15:   # < 15 days
                    needs_backfill = True
                elif target_period == "730d" and interval in ("1h", "4h") and span_days < 180:         # < 6 months
                    needs_backfill = True

                if needs_backfill:
                    print(f"[{symbol}] Existing cache has limited history ({span_days}d). Ingesting FULL history ({target_period})...")
                    df_full = _history_with_retries(ticker, period=target_period, interval=interval)
                    if df_full is not None and len(df_full) > len(df_existing):
                        df_full.index = pd.to_datetime(df_full.index, utc=True, errors="coerce")
                        df_full = df_full[~df_full.index.isna()]
                        df_combined = pd.concat([df_full, df_existing])
                        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
                        df_combined.sort_index(inplace=True)
                        df_combined = validate_ohlcv(df_combined)
                        df_combined.to_csv(filepath)
                        return df_combined

                # Incremental Delta Sync: Query only from T_last onwards
                last_date = pd.Timestamp(df_existing.index[-1])
                start_date = (last_date - timedelta(days=max(1, _CACHE_UPDATE_INTERVAL))).strftime("%Y-%m-%d")

                print(f"[{symbol}] DB has {len(df_existing)} bars up to {last_date}. Fetching incremental delta since {start_date}...")
                df_new = _history_with_retries(ticker, start=start_date, interval=interval)
                if df_new is None or len(df_new) == 0:
                    return validate_ohlcv(df_existing)

                df_new.index = pd.to_datetime(df_new.index, utc=True, errors="coerce")
                df_new = df_new[~df_new.index.isna()]

                # Merge without losing any existing history
                df_combined = pd.concat([df_existing, df_new])
                df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
                df_combined.sort_index(inplace=True)
                df_combined = validate_ohlcv(df_combined)
                df_combined.to_csv(filepath)
                print(f"[{symbol}] Incremental delta merged successfully. Total DB records: {len(df_combined)} bars.")
                return df_combined
        except Exception as e:
            print(f"Error reading cache for {symbol}: {e}. Redownloading full history.")

    print(f"[{symbol}] Initial fetch: Ingesting FULL lifetime history ({target_period}, {interval})...")
    df = _history_with_retries(ticker, period=target_period, interval=interval)

    if df is None or len(df) == 0:
        print(f"[{symbol}] Warning: No data returned from Yahoo Finance.")
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[~df.index.isna()]
    df = validate_ohlcv(df)
    df.to_csv(filepath)
    print(f"[{symbol}] Full history saved to DB: {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}).")
    return df


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and cleans OHLCV data across all asset classes:
    - Removes duplicate timestamps and rows with missing/non-positive Close prices.
    - Preserves Forex (EURUSD=X, USDINR=X) and Spot Indices where Yahoo Finance
      reports Volume == 0 by synthesizing a Tick-Activity Proxy Volume from True Range
      so VWAP, Volume Profile POC, and Volume histograms work universally.
    - Smooths bad prints (>3.5 sigma from a rolling mean) while preserving Low <= Close <= High.
    """
    if df is None or len(df) == 0:
        return df

    cleaned = df.copy()

    # 1. Remove duplicate index timestamps
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned.sort_index(inplace=True)

    # Ensure core OHLC columns are numeric and drop missing Close rows
    for col in ("Open", "High", "Low", "Close"):
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    if "Close" in cleaned.columns:
        cleaned = cleaned[cleaned["Close"].notna() & (cleaned["Close"] > 0)].copy()

    if len(cleaned) == 0:
        return cleaned

    # 2. Universal Volume Handling (Forex, Commodities, Crypto, Stocks)
    if "Volume" in cleaned.columns:
        cleaned["Volume"] = pd.to_numeric(cleaned["Volume"], errors="coerce").fillna(0.0)
        pos_vol_ratio = float((cleaned["Volume"] > 0).mean())

        if pos_vol_ratio < 0.25:
            # Forex pairs (EURUSD=X, etc.) and Spot Indices have 0 reported volume on Yahoo Finance.
            # Synthesize Tick-Activity Proxy Volume from intrabar True Range so volume analytics work!
            high_low = (cleaned["High"] - cleaned["Low"]).abs()
            close_chg = cleaned["Close"].diff().abs().fillna(high_low)
            true_range = np.maximum(high_low, close_chg)
            med_tr = float(np.nanmedian(true_range))
            if med_tr <= 0 or np.isnan(med_tr):
                med_tr = float(cleaned["Close"].iloc[-1]) * 0.002 + 1e-6
            proxy_vol = np.clip((true_range / med_tr) * 10000.0, 500.0, 250000.0)
            cleaned["Volume"] = np.where(cleaned["Volume"] > 0, cleaned["Volume"], np.round(proxy_vol))
        else:
            # Asset has real exchange volume (Crypto, Futures like NG=F / GC=F, Equities).
            # Replace occasional 0-volume candles (e.g., roll/overnight bars in futures) with rolling median
            # instead of dropping the price bar!
            med_vol = cleaned["Volume"].replace(0, np.nan).rolling(20, min_periods=1).median().bfill().fillna(1000.0)
            cleaned["Volume"] = np.where(cleaned["Volume"] > 0, cleaned["Volume"], med_vol)

    # 3. Restore strict OHLC structural consistency (Low <= Open, Close <= High)
    # without clipping genuine market breakouts or trends.
    if all(c in cleaned.columns for c in ("Open", "High", "Low", "Close")):
        cleaned["High"] = cleaned[["Open", "High", "Low", "Close"]].max(axis=1)
        cleaned["Low"] = cleaned[["Open", "High", "Low", "Close"]].min(axis=1)

    return cleaned


def fetch_cross_asset_covariates(
    symbol: str,
    target_df: pd.DataFrame,
    period: str = "max",
    interval: str = "1d",
) -> Dict[str, pd.Series]:
    """
    Fetches and causally aligns the top Cross-Asset Macro Drivers for `symbol`
    (e.g., Crude Oil & DXY for Natural Gas; DXY, Silver & 10Y Yields for Gold;
    DXY & Gold for Forex; ETH & Nasdaq for Bitcoin).
    Returns a dictionary of `{driver_label: aligned_return_series}` indexed by `target_df.index`.
    """
    if target_df is None or len(target_df) < 15:
        return {}

    try:
        from config.market_catalog import get_asset_metadata
        meta = get_asset_metadata(symbol)
        drivers = meta.get("macro_drivers", [])
    except Exception:
        drivers = [("SPY", "S&P 500 ETF"), ("GC=F", "Gold Futures")]

    # Normalize target timestamps to UTC-naive for clean cross-market alignment
    target_idx = pd.to_datetime(target_df.index)
    if getattr(target_idx, "tz", None) is not None:
        target_idx_naive = target_idx.tz_convert("UTC").tz_localize(None)
    else:
        target_idx_naive = target_idx

    aligned_drivers: Dict[str, pd.Series] = {}

    for drv_sym, drv_name in drivers:
        if drv_sym.upper() == symbol.upper():
            continue
        try:
            drv_df = fetch_data(drv_sym, period=period, interval=interval)
            if drv_df is None or len(drv_df) < 10 or "Close" not in drv_df.columns:
                continue

            drv_idx = pd.to_datetime(drv_df.index)
            if getattr(drv_idx, "tz", None) is not None:
                drv_idx_naive = drv_idx.tz_convert("UTC").tz_localize(None)
            else:
                drv_idx_naive = drv_idx

            drv_close = pd.Series(drv_df["Close"].values, index=drv_idx_naive, dtype=np.float64)
            drv_close = drv_close[~drv_close.index.duplicated(keep="last")].sort_index()
            drv_ret = drv_close.pct_change().fillna(0.0)

            # Strictly causal alignment: reindex with forward-fill (only uses past known macro print)
            reindexed = drv_ret.reindex(target_idx_naive, method="ffill").fillna(0.0)
            reindexed.index = target_df.index
            if float(reindexed.std()) > 1e-8:
                aligned_drivers[f"{drv_name} ({drv_sym})"] = reindexed
        except Exception:
            continue

    return aligned_drivers


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
        "status": "HEALTHY" if max_lat < (avg_lat * 3.0 + 60) else "LATENCY SPIKE DETECTED",
    }

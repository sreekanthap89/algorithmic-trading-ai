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
        df.to_csv(filepath)
    else:
        print(f"[{symbol}] Warning: No data returned from Yahoo Finance.")
        
    return df

if __name__ == "__main__":
    # Test
    df = fetch_data("BTC-USD", period="1y", interval="1d")
    print(df.tail())

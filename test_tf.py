import os
import pandas as pd
from data_engine import fetch_data
from features import add_features
from models import TradingModel
import portfolio as pf

def test_all():
    symbol = "BTC-USD"
    timeframes = [("1d", "2y"), ("1h", "1mo"), ("5m", "5d")]
    
    for interval, period in timeframes:
        print(f"\n--- Testing Timeframe: {interval} ({period}) ---")
        df = fetch_data(symbol, period=period, interval=interval)
        print(f"Data fetched: {len(df)} rows")
        
        df_feat = add_features(df)
        print(f"Features engineered: {df_feat.shape}")
        
        model = TradingModel(symbol, interval)
        model.train_or_load(df_feat, force_retrain=True)
        
        pred = model.predict_next(df_feat)
        print(f"Prediction signal: {pred['signal']}, Prob UP: {pred['prob_up']:.4f}")
        
        # Test Buy & Sell operations on this timeframe
        last_price = float(df['Close'].iloc[-1])
        b_ok, b_msg = pf.buy_asset(symbol, 0.05, last_price)
        print(f"Buy execution ({interval}): {b_ok} -> {b_msg}")
        
        s_ok, s_msg = pf.sell_asset(symbol, 0.05, last_price)
        print(f"Sell execution ({interval}): {s_ok} -> {s_msg}")

if __name__ == "__main__":
    test_all()

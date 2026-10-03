import sys
import os

# Add current directory to sys.path
sys.path.append(os.getcwd())

try:
    print("--- Testing Imports ---")
    import pandas as pd
    import numpy as np
    from data_engine import fetch_data
    from features import add_features
    from models import TradingModel, run_monte_carlo, get_markov_regime
    from patterns import detect_and_draw_patterns
    import portfolio as pf
    from pages.execution_dashboard import render_execution_dashboard
    print("✅ All imports successful.")

    print("\n--- Testing Data & Feature Flow ---")
    # Using a dummy symbol or a known one from the data directory
    symbol = "BTC-USD"
    df = fetch_data(symbol, period="1mo", interval="1h")
    if df is not None and not df.empty:
        print(f"✅ Data fetched: {len(df)} rows.")
        df_feat = add_features(df)
        print("✅ Features added successfully.")
    else:
        print("❌ Data fetch failed.")

    print("\n--- Testing Model & Prediction Flow ---")
    model = TradingModel(symbol, "1h")
    # We shouldn't need to train if .pkl exists, but let's ensure it works
    model.train_or_load(df_feat, force_retrain=False)
    prediction = model.predict_next(df_feat)
    print(f"✅ Model prediction successful: {prediction}")

    print("\n--- Testing Regime Flow ---")
    regime = get_markov_regime(df_feat)
    print(f"✅ Regime detection successful: {regime}")

    print("\n--- Testing Pattern Detection Flow ---")
    import plotly.graph_objects as go
    dummy_fig = go.Figure()
    patterns = detect_and_draw_patterns(dummy_fig, df_feat, has_sub=False)
    print(f"✅ Pattern detection successful: {patterns}")

    print("\n--- Testing Trade Plan Flow ---")
    from app import compute_trade_plan
    last_close = float(df['Close'].iloc[-1])
    atr = float(df_feat['ATR'].iloc[-1])
    plan = compute_trade_plan(last_close, atr, prediction, regime, patterns)
    print(f"✅ Trade plan computed: {plan}")

    print("\n--- Testing Portfolio Flow ---")
    summary = pf.get_portfolio_summary({symbol: last_close})
    print(f"✅ Portfolio summary retrieved: Total Value ${summary['total_value']:.2f}")

    print("\n🌟 ALL FLOWS TESTED SUCCESSFULLY! 🌟")

except Exception as e:
    print(f"\n❌ TEST FAILED during execution!")
    import traceback
    traceback.print_exc()
    sys.exit(1)

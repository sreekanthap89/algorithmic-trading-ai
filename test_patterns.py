import pandas as pd
import numpy as np
from data_engine import fetch_data
from features import add_features
from patterns import detect_bearish_bat, detect_falling_wedge, detect_wolfe_wave, detect_cup_handle

# Fetch sample data
df = fetch_data("BTC-USD", period="2y", interval="1d")
if df is not None and len(df) > 0:
    df_feat = add_features(df)
    print(f"Data shape: {df_feat.shape}")
    print()
    
    # Test Wolfe Wave
    try:
        ww_result = detect_wolfe_wave(df_feat)
        print(f"✓ Wolfe Wave: {'Found' if ww_result else 'Not found'}")
    except Exception as e:
        print(f"✗ Wolfe Wave ERROR: {e}")
    
    # Test Cup Handle
    try:
        cup_result = detect_cup_handle(df_feat)
        print(f"✓ Cup Handle: {'Found' if cup_result else 'Not found'}")
    except Exception as e:
        print(f"✗ Cup Handle ERROR: {e}")
    
    # Test Bearish Bat
    try:
        bat_result = detect_bearish_bat(df_feat)
        print(f"✓ Bearish Bat: {'Found' if bat_result else 'Not found'}")
    except Exception as e:
        print(f"✗ Bearish Bat ERROR: {e}")
    
    # Test Falling Wedge
    try:
        wedge_result = detect_falling_wedge(df_feat)
        print(f"✓ Falling Wedge: {'Found' if wedge_result else 'Not found'}")
    except Exception as e:
        print(f"✗ Falling Wedge ERROR: {e}")
else:
    print("Failed to fetch data")

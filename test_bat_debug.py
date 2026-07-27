import pandas as pd
import numpy as np
from data_engine import fetch_data
from features import add_features
from patterns import _pivots

# Fetch sample data
df = fetch_data("BTC-USD", period="2y", interval="1d")
if df is not None and len(df) > 0:
    df_feat = add_features(df)
    
    # Test pivot finding
    lookback = 250
    sub = df_feat.tail(lookback)
    hi, lo = sub["High"].values, sub["Low"].values
    n = len(hi)
    order = max(2, min(4, n // 40))
    
    print(f"Data points: {n}")
    print(f"Order (pivot neighborhood): {order}")
    
    ph = _pivots(hi, "h", order)
    pl = _pivots(lo, "l", order)
    
    print(f"Found {len(ph)} swing highs")
    print(f"Found {len(pl)} swing lows")
    print()
    
    if len(ph) >= 3 and len(pl) >= 2:
        print("✓ Enough pivots found for Bat pattern")
        
        # Check candidate counts
        for i, xi in enumerate(reversed(ph[-10:])):
            yx = hi[xi]
            ai_list = [l for l in pl if l > xi][:8]
            print(f"  High #{i}: {len(ai_list)} lows after it")
            if ai_list:
                for j, ai in enumerate(ai_list[:3]):
                    ya = lo[ai]
                    xa = yx - ya
                    ab_ok = 0.25 <= (ya - ya) <= 0.65 if len(ph) > 0 else False
                    print(f"    Low #{j}: XA range = {xa:.2f}, Ratio check needed for B")
    else:
        print(f"✗ Not enough pivots: need 3+ highs and 2+ lows, got {len(ph)} and {len(pl)}")
else:
    print("Failed to fetch data")

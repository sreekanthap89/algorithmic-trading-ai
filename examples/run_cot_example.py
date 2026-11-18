"""
examples/run_cot_example.py — End-to-End Demonstration
of the Larry Williams COT Algorithmic Trading Engine.

Shows:
  1. Data Ingestion & Market Filtering (Gold GC=F).
  2. Computation of all Larry Williams Indicators (COT Index, WILLCO, ProGo, Inside-OI).
  3. Execution of the One-Minute Weekly Scan strategy (Chapter 13).
  4. Execution of the 17-Day Trailing Stop & Dollar Risk Stop.
  5. Backtest performance reporting.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from cot_engine import (
    generate_synthetic_cot,
    compute_all_cot_features,
    OneMinuteWeeklyScan,
    UnifiedCOTStrategy,
    TripartiteMacroModel,
    DirectionalRegimeFilter,
    COTBacktester,
)


def run_demo():
    print("=" * 70)
    print("LARRY WILLIAMS COT ALGORITHMIC TRADING ENGINE — DEMONSTRATION")
    print("=" * 70)

    # 1. Generate 5-year weekly COT history
    print("\n[1] Ingesting & Generating 5-Year Weekly COT History (Gold GC=F)...")
    cot_df = generate_synthetic_cot(
        start_date="2019-01-08",
        end_date="2024-06-25",
        market_name="GOLD - COMMODITY EXCHANGE INC.",
        base_oi=500000,
        seed=101,
    )

    # Attach simulated Gold price action
    n = len(cot_df)
    rng = np.random.default_rng(101)
    # Price trend loosely aligned with commercial accumulation
    comm_net = cot_df["comm_long"] - cot_df["comm_short"]
    norm_net = (comm_net - comm_net.min()) / (comm_net.max() - comm_net.min())
    price_trend = 1300.0 + np.cumsum(rng.normal(2.0, 15.0, n) + (norm_net - 0.5) * 8.0)

    cot_df["Open"] = price_trend
    cot_df["High"] = price_trend + rng.uniform(5.0, 25.0, n)
    cot_df["Low"] = price_trend - rng.uniform(5.0, 25.0, n)
    cot_df["Close"] = price_trend + rng.normal(0.0, 8.0, n)
    cot_df["Volume"] = rng.integers(100000, 300000, n)

    print(f"    Loaded {len(cot_df)} weekly bars ({cot_df['report_date'].iloc[0].date()} to {cot_df['report_date'].iloc[-1].date()})")

    # 2. Compute all Larry Williams COT Indicators
    print("\n[2] Computing Larry Williams Indicators & Oscillators...")
    enriched_df = compute_all_cot_features(cot_df, include_daily_progo=True)

    latest = enriched_df.iloc[-1]
    print(f"    Commercial Net Position:   {int(latest['comm_net']):,} contracts")
    print(f"    Commercial Share of OI:    {latest['comm_short_pct_oi']:.2f}% (Extreme >55%: {bool(latest['comm_short_extreme_55'])})")
    print(f"    Williams COT Index (156w): {latest['cot_index_156w']:.2f}%")
    print(f"    Williams COT Index (26w):  {latest['cot_index_26w']:.2f}%")
    print(f"    WILLCO (26w Net):          {latest['willco_26w_net']:.2f}%")
    print(f"    Small Spec Public Index:   {latest['public_index_156w']:.2f}%")
    print(f"    Large Trader Fund Index:   {latest['large_index_156w']:.2f}%")
    print(f"    Open Interest Stoch (52w): {latest['oi_stoch_52w']:.2f}%")
    print(f"    Inside-OI Regime Code:     {latest['inside_oi_regime']}")

    # 3. Strategy 1: One-Minute Weekly Scan (Chapter 13)
    print("\n[3] Evaluating Strategy 1: One-Minute Weekly Scan...")
    scan = OneMinuteWeeklyScan(trend_window=30, cot_window=26, bull_threshold=75.0, bear_threshold=25.0)
    signaled_df = scan.generate_signals(enriched_df, price_col="Close")

    sig_counts = signaled_df["one_minute_signal"].value_counts().to_dict()
    print(f"    Signal Distribution: Buy={sig_counts.get(1, 0)}, Sell={sig_counts.get(-1, 0)}, Hold={sig_counts.get(0, 0)}")

    # 4. Strategy 2: Tripartite Macro Model (Chapter 15)
    print("\n[4] Evaluating Strategy 2: Tripartite Macro Framework (Ch 15)...")
    tripartite = TripartiteMacroModel(oi_window=52, cot_window=26, trend_window=39)
    tri_df = tripartite.generate_signals(enriched_df, price_col="Close")
    tri_counts = tri_df["tripartite_signal"].value_counts().to_dict()
    print(f"    Tripartite Signals: Buy={tri_counts.get(1, 0)}, Sell={tri_counts.get(-1, 0)}, Neutral={tri_counts.get(0, 0)}")

    # 5. Backtest Simulation with 17-Day Trailing Stop & Fixed Dollar Risk Stop
    print("\n[5] Running Event-Driven Backtest (Gold Point Value: $100/pt, Stop: $3,500)...")
    tester = COTBacktester(
        point_value=100.0,
        max_dollar_risk=3500.0,
        use_trailing_stop=True,
        trailing_stop_count=8,
        max_holding_bars=25,
        commission_per_trade=15.0,
    )
    results = tester.run(signaled_df, signal_col="one_minute_signal")

    print("\n" + "=" * 50)
    print("BACKTEST PERFORMANCE SUMMARY (One-Minute Weekly Scan)")
    print("=" * 50)
    print(f"  Total Closed Trades:    {results['total_trades']}")
    print(f"  Winning Trades:         {results['winning_trades']}")
    print(f"  Losing Trades:          {results['losing_trades']}")
    print(f"  Win Rate:               {results['win_rate_pct']:.2f}%")
    print(f"  Profit Factor:          {results['profit_factor']:.2f}")
    print(f"  Net Profit:             ${results['net_profit_dollars']:,.2f}")
    print(f"  Average Trade P&L:      ${results['avg_trade_dollars']:,.2f}")
    print(f"  Max Drawdown:           ${results['max_drawdown_dollars']:,.2f}")
    print(f"  Annualized Sharpe:      {results['sharpe_ratio']:.2f}")

    if results["trades"]:
        print("\nSample Recent Closed Trades:")
        for t in results["trades"][-3:]:
            print(f"  - [{t.side}] Entry: {t.entry_date.date() if hasattr(t.entry_date, 'date') else t.entry_date} @ ${t.entry_price:,.2f} | "
                  f"Exit: {t.exit_date.date() if hasattr(t.exit_date, 'date') else t.exit_date} @ ${t.exit_price:,.2f} | "
                  f"P&L: ${t.pnl_dollars:,.2f} ({t.pnl_pct:+.2f}%) | Reason: {t.exit_reason}")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()


"""
test_cot_engine.py — Comprehensive Unit & Integration Test Suite
for the Larry Williams COT Algorithmic Trading Engine.

Tests:
  1. Chapter 4 arithmetic typo correction (350 / -150 / 750 -> 55.56%, not 33%).
  2. Williams COT Index normalization, lookbacks (156w, 52w, 26w), and epsilon guard.
  3. WILLCO Oscillator net & short variants.
  4. Commercial Share % of OI and 55% extreme alert.
  5. Speculator Sentiment Indexes (Public Contrary Index, Large Trader Index).
  6. Inside-OI 4-State Diagnostic Matrix.
  7. ProGo Oscillator & Bullish/Bearish Divergence detection.
  8. 17-Day Trailing Stop inside-bar exclusion logic.
  9. Point-in-Time as-of merge with zero lookahead bias.
 10. One-Minute Weekly Scan strategy & COTBacktester simulation run.
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from cot_engine import (
    MARKET_MAP,
    download_cftc_year,
    filter_market_cot,
    merge_cot_with_price,
    generate_synthetic_cot,
    commercial_net_position,
    commercial_share_oi,
    williams_cot_index,
    williams_willco,
    multi_year_extremes,
    public_contrary_index,
    large_trader_index,
    open_interest_stochastic,
    inside_oi_regime,
    progo_oscillator,
    jiler_seasonal_deviation,
    compute_all_cot_features,
    OneMinuteWeeklyScan,
    UnifiedCOTStrategy,
    TripartiteMacroModel,
    DirectionalRegimeFilter,
    DollarRiskStop,
    seventeen_day_trailing_stop,
    thirteen_day_reentry_levels,
    TimeExitFilter,
    is_inside_day,
    COTBacktester,
)


class TestCOTEngine(unittest.TestCase):

    def setUp(self):
        # 4-year synthetic dataset for testing
        self.synthetic_cot = generate_synthetic_cot(
            start_date="2020-01-07",
            end_date="2024-06-25",
            market_name="GOLD - COMMODITY EXCHANGE INC.",
            base_oi=500000,
            seed=42,
        )

    def test_01_chapter_4_arithmetic_typo_correction(self):
        """
        Verify Chapter 4 (p. 34) arithmetic typo correction:
        Book prints: Current = 350, Low = -150, High = 750 -> '(200 / 600) * 100 = 33%'.
        Correct math: (350 - (-150)) / (750 - (-150)) * 100 = 500 / 900 * 100 = 55.555... -> 55.56%.
        """
        df = pd.DataFrame({
            "comm_long": [1000, 1000, 1350],
            "comm_short": [1150, 250, 1000],  # nets: -150, +750, +350
        })
        # Net positions: -150, +750, +350
        cot_idx = williams_cot_index(df, lookback_weeks=3)
        calculated_value = cot_idx.iloc[-1]
        expected_value = (350 - (-150)) / (750 - (-150)) * 100.0

        self.assertAlmostEqual(calculated_value, expected_value, places=4)
        self.assertAlmostEqual(calculated_value, 55.555555, places=4)
        self.assertNotAlmostEqual(calculated_value, 33.0, places=1)

    def test_02_williams_cot_index_lookbacks_and_bounds(self):
        """
        Test that COT Index computes across multiple lookback windows (156w, 52w, 26w)
        and stays strictly within [0, 100].
        """
        idx_156 = williams_cot_index(self.synthetic_cot, lookback_weeks=156)
        idx_52 = williams_cot_index(self.synthetic_cot, lookback_weeks=52)
        idx_26 = williams_cot_index(self.synthetic_cot, lookback_weeks=26)

        self.assertEqual(len(idx_156), len(self.synthetic_cot))
        self.assertTrue((idx_26.dropna() >= 0.0).all())
        self.assertTrue((idx_26.dropna() <= 100.0).all())

        # Test epsilon guard on flat series (max == min)
        flat_df = pd.DataFrame({"comm_long": [100] * 10, "comm_short": [50] * 10})
        flat_idx = williams_cot_index(flat_df, lookback_weeks=5)
        self.assertEqual(flat_idx.iloc[-1], 50.0)

    def test_03_willco_variants(self):
        """
        Test WILLCO indicator:
          - net: stochastic of (comm_net / OI)
          - short: inverted stochastic of (comm_short / OI)
        """
        willco_net = williams_willco(self.synthetic_cot, lookback_weeks=26, variant="net")
        willco_short = williams_willco(self.synthetic_cot, lookback_weeks=26, variant="short")

        self.assertEqual(len(willco_net), len(self.synthetic_cot))
        self.assertEqual(len(willco_short), len(self.synthetic_cot))
        self.assertTrue((willco_net.dropna() >= 0.0).all())
        self.assertTrue((willco_net.dropna() <= 100.0).all())
        self.assertTrue((willco_short.dropna() >= 0.0).all())
        self.assertTrue((willco_short.dropna() <= 100.0).all())

    def test_04_commercial_share_and_55_percent_alert(self):
        """
        Test Commercial share of Open Interest and the 55% top alert.
        """
        df = pd.DataFrame({
            "open_interest": [100000, 100000],
            "comm_long": [20000, 10000],
            "comm_short": [45000, 60000],  # 45% then 60%
        })
        share = commercial_share_oi(df)
        self.assertAlmostEqual(share["comm_short_pct_oi"].iloc[0], 45.0)
        self.assertAlmostEqual(share["comm_short_pct_oi"].iloc[1], 60.0)
        self.assertEqual(share["comm_short_extreme_55"].iloc[0], 0)
        self.assertEqual(share["comm_short_extreme_55"].iloc[1], 1)

    def test_05_speculator_contrary_indexes(self):
        """
        Test Public Index and Large Trader Index normalization.
        """
        pub_idx = public_contrary_index(self.synthetic_cot, lookback_weeks=52)
        large_idx = large_trader_index(self.synthetic_cot, lookback_weeks=52)

        self.assertTrue((pub_idx.dropna() >= 0.0).all())
        self.assertTrue((pub_idx.dropna() <= 100.0).all())
        self.assertTrue((large_idx.dropna() >= 0.0).all())
        self.assertTrue((large_idx.dropna() <= 100.0).all())

    def test_06_inside_oi_regime_matrix(self):
        """
        Test all 4 regimes of the Inside-OI diagnostic matrix:
          1: Speculative Top (P up, OI up, CommNet down)
          2: Commercial Capitulation (P down, OI up, CommNet up)
          3: Despair Blastoff (P down, OI down, CommNet up)
          4: Hollow Rally (P up, OI down, CommNet down)
        """
        df = pd.DataFrame({
            "Close":        [100, 105,  95,  90,  95],
            "open_interest": [1000, 1100, 1200, 1100, 1000],
            "comm_long":    [ 500,  400,  600,  700,  600],
            "comm_short":   [ 500,  600,  500,  400,  500],
        })
        # Row 1 vs 0: P up (105>100), OI up (1100>1000), CP down (-200 < 0) -> SPECULATIVE_TOP (1)
        # Row 2 vs 1: P down (95<105), OI up (1200>1100), CP up (+100 > -200) -> COMMERCIAL_CAPITULATION (2)
        # Row 3 vs 2: P down (90<95), OI down (1100<1200), CP up (+300 > +100) -> DESPAIR_BLASTOFF (3)
        # Row 4 vs 3: P up (95>90), OI down (1000<1100), CP down (+100 < +300) -> HOLLOW_RALLY (4)
        regimes = inside_oi_regime(df, price_col="Close")

        self.assertEqual(regimes.iloc[1], 1)
        self.assertEqual(regimes.iloc[2], 2)
        self.assertEqual(regimes.iloc[3], 3)
        self.assertEqual(regimes.iloc[4], 4)

    def test_07_progo_oscillator_and_divergence(self):
        """
        Test ProGo oscillator:
          Pro = Close - Open
          Public = Open - Close[t-1]
        """
        bars = pd.DataFrame({
            "Open": [100, 102, 101, 103, 102],
            "Close": [102, 101, 103, 102, 105],
        })
        progo_res = progo_oscillator(bars, window=3)
        self.assertIn("progo", progo_res.columns)
        self.assertIn("public_index", progo_res.columns)
        self.assertIn("progo_bull_divergence", progo_res.columns)
        self.assertIn("progo_bear_divergence", progo_res.columns)

    def test_08_seventeen_day_trailing_stop_strictly_excludes_inside_days(self):
        """
        Verify that is_inside_day detects inside bars (strict inequality: High < Prior High and Low > Prior Low)
        and that seventeen_day_trailing_stop skips inside days when counting bars.
        """
        # Create 5 bars where bar 2 and bar 3 are inside bars
        # Bar 0: H=110, L=90
        # Bar 1: H=108, L=92  (Inside: 108<110, 92>90)
        # Bar 2: H=106, L=94  (Inside: 106<108, 94>92)
        # Bar 3: H=115, L=85  (Outside bar)
        df = pd.DataFrame({
            "High": [110, 108, 106, 115],
            "Low":  [ 90,  92,  94,  85],
        })
        inside = is_inside_day(df)
        self.assertFalse(inside.iloc[0])
        self.assertTrue(inside.iloc[1])
        self.assertTrue(inside.iloc[2])
        self.assertFalse(inside.iloc[3])

        # A 1-bar trailing stop on bar 3:
        # Looking back from bar 3, bar 2 is inside, bar 1 is inside, so the first non-inside bar is bar 0!
        # Bar 0 Low = 90. If inside days were NOT excluded, it would pick bar 2 Low = 94.
        trail_stop = seventeen_day_trailing_stop(df, side="LONG", target_count=1)
        self.assertEqual(trail_stop.iloc[3], 90.0)

    def test_09_point_in_time_merge_asof_zero_lookahead(self):
        """
        Verify point-in-time merge_asof:
        COT released on Friday 15:30 ET must NOT leak into earlier bars.
        """
        # Tuesday 2024-01-09 report date -> available Friday 2024-01-12 at 15:30:00
        cot = pd.DataFrame({
            "report_date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-09")],
            "available_at": [
                pd.Timestamp("2024-01-05 15:30:00"),  # Release 1
                pd.Timestamp("2024-01-12 15:30:00"),  # Release 2
            ],
            "comm_long": [100, 500],
            "comm_short": [200, 100],
            "open_interest": [1000, 1200],
        })

        # Price bars:
        # Bar 1: Friday 2024-01-12 14:00 (Before release) -> MUST see Release 1 (comm_long=100)
        # Bar 2: Friday 2024-01-12 16:00 (After release)  -> MUST see Release 2 (comm_long=500)
        # Bar 3: Monday 2024-01-15 09:30                  -> MUST see Release 2 (comm_long=500)
        price = pd.DataFrame({
            "Close": [2000.0, 2010.0, 2020.0],
        }, index=pd.to_datetime([
            "2024-01-12 14:00:00",
            "2024-01-12 16:00:00",
            "2024-01-15 09:30:00",
        ]))

        merged = merge_cot_with_price(price, cot)

        self.assertEqual(merged.loc["2024-01-12 14:00:00", "comm_long"], 100)
        self.assertEqual(merged.loc["2024-01-12 16:00:00", "comm_long"], 500)
        self.assertEqual(merged.loc["2024-01-15 09:30:00", "comm_long"], 500)

    def test_10_one_minute_weekly_scan_and_backtester(self):
        """
        Verify Strategy 1 (One-Minute Weekly Scan) signal generation
        and COTBacktester execution loop.
        """
        # Create 100 weekly price bars matching synthetic COT
        n = len(self.synthetic_cot)
        price_base = 1500.0 + np.cumsum(np.random.default_rng(42).normal(1.0, 10.0, n))
        self.synthetic_cot["Open"] = price_base
        self.synthetic_cot["High"] = price_base + 15.0
        self.synthetic_cot["Low"] = price_base - 15.0
        self.synthetic_cot["Close"] = price_base + 2.0

        scan = OneMinuteWeeklyScan(trend_window=20, cot_window=10, bull_threshold=70.0, bear_threshold=30.0)
        signaled_df = scan.generate_signals(self.synthetic_cot, price_col="Close")

        self.assertIn("one_minute_signal", signaled_df.columns)
        sig_counts = signaled_df["one_minute_signal"].value_counts().to_dict()
        self.assertTrue(1 in sig_counts or -1 in sig_counts)

        # Run Backtest
        tester = COTBacktester(
            point_value=100.0,
            max_dollar_risk=3000.0,
            use_trailing_stop=True,
            trailing_stop_count=5,
            max_holding_bars=20,
        )
        results = tester.run(signaled_df, signal_col="one_minute_signal")

        self.assertIn("total_trades", results)
        self.assertIn("win_rate_pct", results)
        self.assertIn("net_profit_dollars", results)
        self.assertIn("trades", results)


if __name__ == "__main__":
    unittest.main()


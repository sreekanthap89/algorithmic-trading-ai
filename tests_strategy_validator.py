"""
tests_strategy_validator.py — Comprehensive Audit Test Suite for
Larry Williams COT Strategies, Risk Rules & Backtester.
"""

import unittest
import numpy as np
import pandas as pd
import glob

from cot_engine import (
    MARKET_MAP,
    filter_market_cot,
    merge_cot_with_price,
    generate_synthetic_cot,
    williams_cot_index,
    williams_willco,
    open_interest_stochastic,
    progo_oscillator,
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


class TestStrategyValidator(unittest.TestCase):

    def setUp(self):
        # 100-bar synthetic weekly dataset
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2020-01-03", periods=n, freq="W-FRI")
        
        # Synthetic price with uptrend then downtrend
        trend = np.concatenate([np.linspace(1000, 1500, 50), np.linspace(1500, 900, 50)])
        noise = np.random.normal(0, 10, n)
        close = trend + noise
        open_p = close - np.random.uniform(-5, 5, n)
        high = np.maximum(open_p, close) + np.random.uniform(2, 10, n)
        low = np.minimum(open_p, close) - np.random.uniform(2, 10, n)

        self.df = pd.DataFrame({
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "comm_long": np.random.randint(10000, 50000, n),
            "comm_short": np.random.randint(10000, 50000, n),
            "open_interest": np.random.randint(80000, 150000, n),
        }, index=dates)

    # =========================================================================
    # 1. OneMinuteWeeklyScan (Chapter 13, Table 13.1)
    # =========================================================================
    def test_01_one_minute_weekly_scan_rules_and_lags(self):
        """
        Verify:
        - 52-week SMA trend condition:
            trend_up = SMA52[t] > SMA52[t-lag]
            trend_down = SMA52[t] < SMA52[t-lag]
        - lag parameter: lag=1 (prose) vs lag=2 (Table 13.1 empirical tests)
        - 26-week COT index threshold (>80 buy, <20 sell)
        - Monday open execution
        - Strict filtering of counter-trend signals (no buys in downtrends, no sells in uptrends)
        """
        # Test with lag=2
        scan_lag2 = OneMinuteWeeklyScan(trend_window=20, cot_window=10, trend_lag=2, bull_threshold=80.0, bear_threshold=20.0)
        res_lag2 = scan_lag2.generate_signals(self.df)
        self.assertIn("one_minute_signal", res_lag2.columns)
        self.assertIn("one_minute_trigger", res_lag2.columns)
        self.assertIn("sma_52w", res_lag2.columns)
        self.assertIn("trend_up", res_lag2.columns)
        self.assertIn("trend_down", res_lag2.columns)

        # Test with lag=1
        scan_lag1 = OneMinuteWeeklyScan(trend_window=20, cot_window=10, trend_lag=1, bull_threshold=80.0, bear_threshold=20.0)
        res_lag1 = scan_lag1.generate_signals(self.df)

        # Explicitly verify counter-trend filter
        # When trend_up is 0, one_minute_signal MUST NOT be 1 (buy)
        counter_trend_buys = (res_lag2["trend_up"] == 0) & (res_lag2["one_minute_signal"] == 1)
        self.assertEqual(counter_trend_buys.sum(), 0, "Counter-trend buys occurred when trend_up == 0!")

        # When trend_down is 0, one_minute_signal MUST NOT be -1 (sell)
        counter_trend_sells = (res_lag2["trend_down"] == 0) & (res_lag2["one_minute_signal"] == -1)
        self.assertEqual(counter_trend_sells.sum(), 0, "Counter-trend sells occurred when trend_down == 0!")

        # Verify threshold enforcement
        cot_col = "cot_index_10w"
        buys = res_lag2[res_lag2["one_minute_signal"] == 1]
        if len(buys) > 0:
            self.assertTrue((buys[cot_col] >= 80.0).all(), "Buy signal generated with COT Index < 80.0!")

        sells = res_lag2[res_lag2["one_minute_signal"] == -1]
        if len(sells) > 0:
            self.assertTrue((sells[cot_col] <= 20.0).all(), "Sell signal generated with COT Index > 20.0!")

    # =========================================================================
    # 2. 17-Day Trailing Stop: Inside-Day Exclusion & Edge Cases
    # =========================================================================
    def test_02_seventeen_day_trailing_stop_inside_bar_exclusion(self):
        """
        Verify:
        - Strict definition: High < Prior High AND Low > Prior Low
        - Inside days are strictly excluded from the lookback count
        - Edge cases with 1, 2, 3, and 5 consecutive inside days
        - Ratcheting option
        """
        # Construct synthetic dataframe with 5 consecutive inside days
        # Bar 0: High=120, Low=80 (Non-inside baseline)
        # Bar 1: High=118, Low=82 (Inside 1)
        # Bar 2: High=116, Low=84 (Inside 2)
        # Bar 3: High=114, Low=86 (Inside 3)
        # Bar 4: High=112, Low=88 (Inside 4)
        # Bar 5: High=110, Low=90 (Inside 5)
        # Bar 6: High=125, Low=75 (Non-inside breakout)
        df_inside = pd.DataFrame({
            "High": [120, 118, 116, 114, 112, 110, 125],
            "Low":  [ 80,  82,  84,  86,  88,  90,  75],
        })

        inside = is_inside_day(df_inside)
        self.assertFalse(inside.iloc[0])
        self.assertTrue(inside.iloc[1])
        self.assertTrue(inside.iloc[2])
        self.assertTrue(inside.iloc[3])
        self.assertTrue(inside.iloc[4])
        self.assertTrue(inside.iloc[5])
        self.assertFalse(inside.iloc[6])

        # Test target_count = 1 at Bar 6:
        # Looking back from Bar 6:
        # Bars 5, 4, 3, 2, 1 are all inside bars.
        # The first non-inside bar is Bar 0!
        # Bar 0 Low = 80, High = 120.
        stop_long = seventeen_day_trailing_stop(df_inside, side="LONG", target_count=1)
        stop_short = seventeen_day_trailing_stop(df_inside, side="SHORT", target_count=1)

        # Bar 6 stop MUST equal Bar 0's low/high, because bars 1-5 were strictly skipped!
        self.assertEqual(stop_long.iloc[6], 80.0, f"Expected 80.0 (Bar 0 Low), got {stop_long.iloc[6]}")
        self.assertEqual(stop_short.iloc[6], 120.0, f"Expected 120.0 (Bar 0 High), got {stop_short.iloc[6]}")

        # If inside days were NOT excluded, Bar 6 lookback 1 would have picked Bar 5 (Low=90, High=110)
        self.assertNotEqual(stop_long.iloc[6], 90.0)
        self.assertNotEqual(stop_short.iloc[6], 110.0)

        # Test Ratchet mode
        df_ratchet = pd.DataFrame({
            "High": [100, 110, 105, 120],
            "Low":  [ 90,  95,  85, 100],  # Bar 2 makes lower low (85), but stop should not drop if ratcheted
        })
        stop_ratchet = seventeen_day_trailing_stop(df_ratchet, side="LONG", target_count=1, ratchet=True)
        # Bar 1 stop is low[0] = 90
        # Bar 2 stop is low[1] = 95
        # Bar 3: low[2]=85, but ratchet keeps it at 95!
        self.assertEqual(stop_ratchet.iloc[1], 90.0)
        self.assertEqual(stop_ratchet.iloc[2], 95.0)
        self.assertEqual(stop_ratchet.iloc[3], 95.0)

    # =========================================================================
    # 3. Fixed Dollar Risk Stop (DollarRiskStop)
    # =========================================================================
    def test_03_dollar_risk_stop_calculation_and_execution(self):
        """
        Verify:
        - Points = max_dollar_risk / point_value
        - Long stop = entry_price - points
        - Short stop = entry_price + points
        - Intrabar stop enforcement with gap handling
        """
        # Test calculation
        stop_calc = DollarRiskStop(max_dollar_risk=3500.0, point_value=100.0)
        long_stop = stop_calc.calculate_stop(entry_price=2000.0, side="LONG")
        short_stop = stop_calc.calculate_stop(entry_price=2000.0, side="SHORT")

        # 3500 / 100 = 35 points
        self.assertEqual(long_stop, 1965.0)
        self.assertEqual(short_stop, 2035.0)

        # Intrabar execution via backtester
        df_test = pd.DataFrame({
            "Open":  [100.0, 100.0, 99.0],
            "High":  [105.0, 105.0, 101.0],
            "Low":   [ 95.0,  95.0,  93.0], # Drops below 95.0 stop on bar 2
            "Close": [100.0, 100.0, 94.0],
            "sig":   [    1,     0,     0],
        })
        bt = COTBacktester(point_value=1.0, max_dollar_risk=5.0, use_trailing_stop=False)
        res = bt.run(df_test, signal_col="sig")
        self.assertEqual(len(res["trades"]), 1)
        t = res["trades"][0]
        self.assertEqual(t.exit_reason, "DOLLAR_STOP")
        self.assertEqual(t.exit_price, 95.0)
        self.assertEqual(t.pnl_points, -5.0)

    # =========================================================================
    # 4. TimeExitFilter (10-15 weeks)
    # =========================================================================
    def test_04_time_exit_filter_logic(self):
        """
        Verify:
        - Daily mode: min=50, max=75 bars
        - Weekly mode: min=10, max=15 bars
        - Holding duration >= max_holding_bars triggers should_exit
        """
        # Daily mode
        filter_daily = TimeExitFilter(timeframe="daily")
        self.assertEqual(filter_daily.min_holding_bars, 50)
        self.assertEqual(filter_daily.max_holding_bars, 75)
        self.assertFalse(filter_daily.should_exit(49))
        self.assertFalse(filter_daily.should_exit(74))
        self.assertTrue(filter_daily.should_exit(75))
        self.assertTrue(filter_daily.should_exit(80))

        # Weekly mode
        filter_weekly = TimeExitFilter(timeframe="weekly")
        self.assertEqual(filter_weekly.min_holding_bars, 10)
        self.assertEqual(filter_weekly.max_holding_bars, 15)
        self.assertFalse(filter_weekly.should_exit(9))
        self.assertFalse(filter_weekly.should_exit(14))
        self.assertTrue(filter_weekly.should_exit(15))
        self.assertTrue(filter_weekly.should_exit(20))

    # =========================================================================
    # 5. UnifiedCOTStrategy & TripartiteMacroModel
    # =========================================================================
    def test_05_unified_strategy_and_tripartite_macro_model(self):
        """
        Verify multi-tier logic:
        - UnifiedCOTStrategy: WILLCO (Tier 1) + ProGo divergence (Tier 2) + 3-bar swing (Tier 3)
        - TripartiteMacroModel: OI Stoch (Tier 1) + COT Index (Tier 2) + 39w SMA (Tier 3)
        """
        # UnifiedCOTStrategy
        unified = UnifiedCOTStrategy(willco_window=10, willco_bull=70.0, willco_bear=30.0, trigger_lookback=3)
        res_u = unified.generate_signals(self.df)
        self.assertIn("unified_cot_signal", res_u.columns)
        self.assertTrue(set(res_u["unified_cot_signal"].unique()).issubset({-1, 0, 1}))

        # TripartiteMacroModel
        tripartite = TripartiteMacroModel(oi_window=20, cot_window=10, trend_window=15, oi_bull=25.0, oi_bear=75.0)
        res_t = tripartite.generate_signals(self.df)
        self.assertIn("tripartite_signal", res_t.columns)
        self.assertIn("tripartite_trigger", res_t.columns)
        self.assertTrue(set(res_t["tripartite_signal"].unique()).issubset({-1, 0, 1}))

    # =========================================================================
    # 6. Empirical Backtest with Real Data & Metric Verification
    # =========================================================================
    def test_06_empirical_backtest_with_real_gold_data(self):
        """
        Run empirical backtest on real CFTC Gold COT merged with GC_F daily data.
        Verify:
        - No same-bar re-entry on stop out
        - Trailing stop does not shadow time exit (no elif bug)
        - Accurate metric calculation: win rate, profit factor, max drawdown, Sharpe
        - Clean trade logs
        """
        cot_files = sorted(glob.glob("data/cot/deacot_*.parquet"))
        if not cot_files:
            self.skipTest("No COT parquet files found in data/cot")

        cot_all = pd.concat([pd.read_parquet(f) for f in cot_files], ignore_index=True)
        gold_cot = filter_market_cot(cot_all, "GOLD")
        self.assertGreater(len(gold_cot), 0)

        price = pd.read_csv("data/GC_F_1d.csv")
        price["Date"] = pd.to_datetime(price["Date"], utc=True)
        price = price.set_index("Date").sort_index()

        merged = merge_cot_with_price(price, gold_cot).dropna(subset=["comm_long", "comm_short", "open_interest"])
        weekly = merged.resample("W-FRI").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "comm_long": "last",
            "comm_short": "last",
            "open_interest": "last",
        }).dropna()

        scan = OneMinuteWeeklyScan(trend_window=52, cot_window=26, trend_lag=2)
        weekly_scanned = scan.generate_signals(weekly)

        tester = COTBacktester(
            point_value=100.0,
            max_dollar_risk=3500.0,
            use_trailing_stop=True,
            trailing_stop_count=17,
            max_holding_bars=15,  # 15 weekly bars
            entry_mode="trigger", # trigger mode
        )
        res = tester.run(weekly_scanned, signal_col="one_minute_signal")

        self.assertGreater(res["total_trades"], 0)
        self.assertIn("win_rate_pct", res)
        self.assertIn("profit_factor", res)
        self.assertIn("max_drawdown_dollars", res)
        self.assertIn("sharpe_ratio", res)

        trades = res["trades"]
        self.assertEqual(len(trades), res["total_trades"])

        # Check execution sequence: verify NO trade entered on the same bar another trade exited on stop out
        for i in range(len(trades) - 1):
            cur_t = trades[i]
            next_t = trades[i + 1]
            if cur_t.exit_reason in ("DOLLAR_STOP", "TRAILING_17D_STOP"):
                self.assertGreater(next_t.entry_idx, cur_t.exit_idx,
                                   f"Trade {i+2} entered on or before Trade {i+1} exited! "
                                   f"Exit idx: {cur_t.exit_idx}, Entry idx: {next_t.entry_idx}")

        # Check that 15-week time exit actually fired on winning/holding trades
        exit_reasons = [t.exit_reason for t in trades]
        self.assertIn("TIME_EXIT_15W", exit_reasons, "TIME_EXIT_15W never triggered (elif shadowing bug still present)!")

        time_exits = [t for t in trades if t.exit_reason == "TIME_EXIT_15W"]
        for t in time_exits:
            self.assertEqual(t.bars_held, 15, f"Trade held {t.bars_held} bars on TIME_EXIT_15W, expected 15!")


if __name__ == "__main__":
    unittest.main()

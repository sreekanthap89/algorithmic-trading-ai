"""
test_ch10_ch15_features.py — Comprehensive Unit & Integration Test Suite
for Features from Chapters 10–15 of Larry Williams' 'Secrets of the COT Report' (Wiley, 2005).

Feature Coverage:
  - Feature 10.1: Commercials % of OI (CL%OI & CS%OI, Bearish Top >= 55%, Bullish Bottom < 30%)
  - Feature 10.2: WILLCO 26w Net and Short variants (buy >= 75-80%, peak >= 90%, sell <= 20-25%)
  - Feature 10.3: ProGo & Public Index (14-day SMA of Close - Open vs Open - Prior Close, divergences)
  - Feature 10.4: Unified Multi-Condition Trading System (WILLCO + Seasonal + ProGo + Trendline trigger)
  - Feature 10.5: Consecutive Down Closes Mean-Reversion Pattern (streak >= 3 buy next open, 1-day hold)
  - Feature 10.6: 6-Month COT Commercial Index Directional Filter (COT Index 26w > 50% long, < 50% short)
  - Feature 11.1: Aggregated Dollar-Weighted Index Commercial Position (sum across ES, SP, NQ, ND, YM, DJ; -$38B / +$30B alert)
  - Feature 11.2: Synthetic Commercial Proxy Indicator (scale-down accumulation / scale-up distribution, buy >= 80%, sell <= 20%)
  - Feature 12.1: Dollar-Based Protective Stop Loss (DollarRiskStop)
  - Feature 12.2: 17-Day Trend Trailing Stop strictly excluding inside days (seventeen_day_trailing_stop)
  - Feature 12.3: 13-Day Trend Re-entry Rule (thirteen_day_reentry_levels excluding inside days)
  - Feature 12.4: Time-Based Trend Exhaustion (TimeExitFilter) & Parabolic Windfall Exit (ParabolicWindfallExit)
  - Feature 12.5: Anti-Plunging & Position Sizing Discipline (AntiPlungingPositionSizer)
  - Feature 13.1: The 60-Second Weekly Screening Strategy (OneMinuteWeeklyScan, 52w SMA trend, 26w COT Index > 80 / < 20)
  - Feature 13.2: Gold / U.S. Dollar Spread Valuation Index ((SMA3 - SMA21) / 100 > 0.30 overbought short setup)
  - Feature 13.3: Gold Seasonal Cycle (July accumulation window, Dec/Jan distribution window)
  - Feature 14.1 - 14.3: Classical Chart Patterns & 1-2-3 Reversals (FlagBracketExecution, detect_123_reversal, classical_chart_breakouts)
  - Feature 15.1: Tripartite Execution Architecture (Tier 1 OI Stoch, Tier 2 COT Index / WILLCO, Tier 3 39-week SMA)
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from cot_engine import (
    # Indicators
    commercial_share_oi,
    williams_cot_index,
    williams_willco,
    progo_oscillator,
    open_interest_stochastic,
    aggregated_dollar_commercial_position,
    synthetic_commercial_proxy,
    DEFAULT_INDEX_MULTIPLIERS,
    # Strategies
    OneMinuteWeeklyScan,
    UnifiedCOTStrategy,
    ConsecutiveDownClosesStrategy,
    consecutive_down_closes_pattern,
    DirectionalRegimeFilter,
    intermarket_gold_dollar_valuation,
    gold_usd_valuation_signals,
    gold_seasonal_cycle,
    detect_123_reversal,
    classical_chart_breakouts,
    TripartiteMacroModel,
    # Risk
    DollarRiskStop,
    seventeen_day_trailing_stop,
    thirteen_day_reentry_levels,
    TimeExitFilter,
    ParabolicWindfallExit,
    AntiPlungingPositionSizer,
    FlagBracketExecution,
    is_inside_day,
)


class TestCh10To15Features(unittest.TestCase):

    def setUp(self):
        # 100-week synthetic dataset
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2022-01-07", periods=n, freq="W-FRI")
        
        # Synthetic prices
        t = np.linspace(0, 4 * np.pi, n)
        close = 100.0 + 20.0 * np.sin(t) + np.cumsum(np.random.normal(0, 1.5, n))
        high = close + np.random.uniform(1.0, 3.0, n)
        low = close - np.random.uniform(1.0, 3.0, n)
        open_p = (close + low) / 2.0
        oi = 100000 + 20000 * np.cos(t) + np.random.normal(0, 1000, n)
        
        # Commercial commitments
        comm_long = np.clip(30000 + 15000 * np.sin(t), 1000, None)
        comm_short = np.clip(50000 - 15000 * np.sin(t), 1000, None)
        nonrep_long = np.clip(20000 + 5000 * np.sin(t), 500, None)
        nonrep_short = np.clip(15000 - 5000 * np.sin(t), 500, None)

        self.df = pd.DataFrame({
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "open_interest": oi,
            "comm_long": comm_long,
            "comm_short": comm_short,
            "nonrep_long": nonrep_long,
            "nonrep_short": nonrep_short,
            "Volume": np.random.uniform(10000, 50000, n),
        }, index=dates)

    # =========================================================================
    # 1. Feature 10.1: Commercials % of OI (CL%OI & CS%OI, Bearish Top >= 55%, Bullish Bottom < 30%)
    # =========================================================================
    def test_feature_10_1_commercial_share_oi_alerts(self):
        # Explicit test df
        df = pd.DataFrame({
            "comm_long":  [2000, 4000, 1000],
            "comm_short": [6000, 2000, 5500],
            "open_interest": [10000, 10000, 10000],
        })
        res = commercial_share_oi(df)
        # Bar 0: CS%OI = 60% (>= 55% -> Bearish top)
        self.assertEqual(res["comm_short_pct_oi"].iloc[0], 60.0)
        self.assertEqual(res["comm_short_extreme_55"].iloc[0], 1)
        self.assertEqual(res["comm_short_bottom_30"].iloc[0], 0)

        # Bar 1: CS%OI = 20% (< 30% -> Bullish bottom)
        self.assertEqual(res["comm_short_pct_oi"].iloc[1], 20.0)
        self.assertEqual(res["comm_short_extreme_55"].iloc[1], 0)
        self.assertEqual(res["comm_short_bottom_30"].iloc[1], 1)

        # Bar 2: CS%OI = 55% (boundary condition: 55% is extreme)
        self.assertAlmostEqual(res["comm_short_pct_oi"].iloc[2], 55.0, places=5)
        self.assertEqual(res["comm_short_extreme_55"].iloc[2], 1)

    # =========================================================================
    # 2. Feature 10.2: WILLCO 26w Net and Short Variants
    # =========================================================================
    def test_feature_10_2_willco_variants_and_thresholds(self):
        willco_net = williams_willco(self.df, lookback_weeks=26, variant="net")
        willco_short = williams_willco(self.df, lookback_weeks=26, variant="short")

        # Must be bounded strictly in [0.0, 100.0]
        valid_net = willco_net.dropna()
        self.assertTrue((valid_net >= 0.0).all() and (valid_net <= 100.0).all())
        valid_short = willco_short.dropna()
        self.assertTrue((valid_short >= 0.0).all() and (valid_short <= 100.0).all())

        # Check peak conviction trigger zone >= 75-80% and sell <= 20-25%
        self.assertTrue((valid_net >= 75.0).any(), "Expected some WILLCO buy setup zones")
        self.assertTrue((valid_net <= 25.0).any(), "Expected some WILLCO sell setup zones")

    # =========================================================================
    # 3. Feature 10.3: ProGo & Public Index with Divergences
    # =========================================================================
    def test_feature_10_3_progo_divergences(self):
        # 30 bars with synthetic accumulation divergence:
        # Price drops to new low, but intraday buyers close strong so ProGo rises
        n = 30
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        close = np.linspace(100, 80, n)  # Steadily falling price to new lows
        open_p = close - 2.0             # Close finishes strong (Close > Open) -> ProGo positive & rising
        
        df_div = pd.DataFrame({
            "Open": open_p,
            "Close": close,
        }, index=dates)

        res = progo_oscillator(df_div, window=5, div_lookback=10)
        self.assertIn("progo", res.columns)
        self.assertIn("public_index", res.columns)
        self.assertIn("progo_bull_divergence", res.columns)
        self.assertIn("progo_bear_divergence", res.columns)
        # Verify progo is positive (Close > Open)
        self.assertTrue((res["progo"].dropna() > 0).all())

    # =========================================================================
    # 4. Feature 10.4: Unified Multi-Condition Trading System
    # =========================================================================
    def test_feature_10_4_unified_cot_strategy(self):
        strat = UnifiedCOTStrategy(
            willco_window=26,
            willco_bull=70.0,
            willco_bear=30.0,
            trigger_lookback=3,
            use_seasonal_filter=True,
        )
        # Seasonal mask where second half is seasonally bullish
        seasonal_long = pd.Series(False, index=self.df.index)
        seasonal_long.iloc[50:] = True

        res = strat.generate_signals(self.df, seasonal_long_mask=seasonal_long)
        self.assertIn("unified_cot_signal", res.columns)
        signals = res["unified_cot_signal"]
        self.assertTrue(set(signals.unique()).issubset({-1, 0, 1}))
        # In the first 50 bars, seasonal_long was False, so no long signals allowed
        self.assertTrue((signals.iloc[:50] != 1).all())

    # =========================================================================
    # 5. Feature 10.5: Consecutive Down Closes Mean-Reversion Pattern
    # =========================================================================
    def test_feature_10_5_consecutive_down_closes(self):
        prices = [100.0, 99.0, 98.0, 97.0, 96.0, 99.0, 98.0]
        # Day 0: 100
        # Day 1: 99  (down 1)
        # Day 2: 98  (down 2)
        # Day 3: 97  (down 3) -> trigger at close
        # Day 4: 96  (down 4) -> in position from Day 3 trigger, and triggers again (streak=4 >= 3)
        # Day 5: 99  (up, streak resets to 0) -> in position from Day 4 trigger
        # Day 6: 98  (down 1)
        df_down = pd.DataFrame({"Close": prices, "Open": prices})

        res = consecutive_down_closes_pattern(df_down, min_consecutive=3)
        self.assertEqual(res["consecutive_down_count"].tolist(), [0, 1, 2, 3, 4, 0, 1])
        self.assertEqual(res["consecutive_down_trigger"].tolist(), [0, 0, 0, 1, 1, 0, 0])
        # Position active on Day 4 and Day 5
        self.assertEqual(res["consecutive_down_position"].tolist(), [0, 0, 0, 0, 1, 1, 0])

        # Test class strategy
        strat = ConsecutiveDownClosesStrategy(min_consecutive=3)
        res_strat = strat.generate_signals(df_down)
        self.assertIn("consecutive_down_trigger", res_strat.columns)

    # =========================================================================
    # 6. Feature 10.6: 6-Month COT Commercial Index Directional Filter
    # =========================================================================
    def test_feature_10_6_directional_regime_filter(self):
        filt = DirectionalRegimeFilter(cot_window=26)
        regime = filt.get_regime(self.df)
        self.assertTrue(set(regime.unique()).issubset({-1, 0, 1}))

        # Helper methods
        self.assertTrue(filt.is_long_permitted(75.0))
        self.assertFalse(filt.is_long_permitted(45.0))
        self.assertTrue(filt.is_short_permitted(25.0))
        self.assertFalse(filt.is_short_permitted(65.0))

        # Test signal filtering
        raw_signals = pd.Series([1, -1, 1, -1], index=self.df.index[:4])
        # Force regime to +1 (long only)
        mock_df = self.df.iloc[:4].copy()
        mock_df["cot_index_26w"] = [80.0, 80.0, 80.0, 80.0]
        filtered = filt.filter_signals(mock_df, raw_signals)
        # Long signals (+1) retained, short signals (-1) suppressed to 0
        self.assertEqual(filtered.tolist(), [1, 0, 1, 0])

    # =========================================================================
    # 7. Feature 11.1: Aggregated Dollar-Weighted Index Commercial Position
    # =========================================================================
    def test_feature_11_1_aggregated_dollar_commercial_position(self):
        dates = pd.date_range("2024-01-05", periods=5, freq="W-FRI")
        # Build 6 index futures DataFrames
        market_data = {}
        for sym, mult in DEFAULT_INDEX_MULTIPLIERS.items():
            market_data[sym] = pd.DataFrame({
                "comm_long": [10000, 10000, 5000, 2000, 1000],
                "comm_short": [20000, 30000, 50000, 80000, 100000], # Net short expanding
                "nonrep_long": [15000, 25000, 40000, 70000, 90000], # Net long expanding
                "nonrep_short": [5000, 5000, 5000, 5000, 5000],
                "Close": [5000.0, 5050.0, 5100.0, 5150.0, 5200.0],
            }, index=dates)

        res = aggregated_dollar_commercial_position(market_data)
        self.assertIn("aggregated_dollar_comm_position", res.columns)
        self.assertIn("aggregated_dollar_nonrep_position", res.columns)
        self.assertIn("extreme_imbalance_alert", res.columns)

        # Comm dollar must be negative (net short)
        self.assertTrue((res["aggregated_dollar_comm_position"] < 0).all())
        # Nonrep dollar must be positive (net long)
        self.assertTrue((res["aggregated_dollar_nonrep_position"] > 0).all())

        # Test extreme alert triggering with lower thresholds
        alert_df = aggregated_dollar_commercial_position(
            market_data,
            comm_short_alert_threshold=-1e9,
            nonrep_long_alert_threshold=1e9,
        )
        self.assertTrue((alert_df["extreme_imbalance_alert"] == 1).any())

    # =========================================================================
    # 8. Feature 11.2: Synthetic Commercial Proxy Indicator
    # =========================================================================
    def test_feature_11_2_synthetic_commercial_proxy(self):
        # 150 daily bars for an individual stock (e.g. MSFT)
        n = 150
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 200.0 + np.cumsum(np.random.normal(0, 2.0, n))
        high = close + np.random.uniform(1.0, 3.0, n)
        low = close - np.random.uniform(1.0, 3.0, n)
        vol = np.random.uniform(500000, 2000000, n)

        stock_df = pd.DataFrame({
            "Close": close,
            "High": high,
            "Low": low,
            "Volume": vol,
        }, index=dates)

        res = synthetic_commercial_proxy(stock_df, lookback_bars=60)
        self.assertIn("synthetic_proxy_index", res.columns)
        self.assertIn("synthetic_proxy_buy_setup", res.columns)
        self.assertIn("synthetic_proxy_sell_setup", res.columns)

        valid_proxy = res["synthetic_proxy_index"].dropna()
        self.assertTrue((valid_proxy >= 0.0).all() and (valid_proxy <= 100.0).all())

    # =========================================================================
    # 9. Feature 12.1: Dollar-Based Protective Stop Loss
    # =========================================================================
    def test_feature_12_1_dollar_risk_stop(self):
        # Gold contract: multiplier = 100
        stop_rule = DollarRiskStop(max_dollar_risk=2000.0, point_value=100.0)
        # $2000 / 100 = 20 points
        long_stop = stop_rule.calculate_stop(entry_price=1800.0, side="LONG")
        self.assertEqual(long_stop, 1780.0)

        short_stop = stop_rule.calculate_stop(entry_price=1800.0, side="SHORT")
        self.assertEqual(short_stop, 1820.0)

        # Check breach evaluation
        stopped, price = stop_rule.check_stop(1800.0, bar_high=1795.0, bar_low=1775.0, side="LONG")
        self.assertTrue(stopped)
        self.assertEqual(price, 1780.0)

        stopped, price = stop_rule.check_stop(1800.0, bar_high=1795.0, bar_low=1785.0, side="LONG")
        self.assertFalse(stopped)

    # =========================================================================
    # 10. Feature 12.2: 17-Day Trend Trailing Stop (strictly excluding inside days)
    # =========================================================================
    def test_feature_12_2_seventeen_day_trailing_stop(self):
        # Construct synthetic dataframe where inside days are injected
        # Bar 0: H=120, L=80
        # Bar 1: H=115, L=85 (Inside)
        # Bar 2: H=110, L=90 (Inside)
        # Bar 3: H=125, L=95 (Outside)
        df_inside = pd.DataFrame({
            "High": [120, 115, 110, 125],
            "Low":  [ 80,  85,  90,  95],
        })
        inside = is_inside_day(df_inside)
        self.assertFalse(inside.iloc[0])
        self.assertTrue(inside.iloc[1])
        self.assertTrue(inside.iloc[2])
        self.assertFalse(inside.iloc[3])

        # 1-bar lookback at Bar 3: skips Bar 2 and Bar 1 (inside), picking Bar 0 (Low=80, High=120)
        stop_long = seventeen_day_trailing_stop(df_inside, side="LONG", target_count=1)
        self.assertEqual(stop_long.iloc[3], 80.0)

    # =========================================================================
    # 11. Feature 12.3: 13-Day Trend Re-entry Rule (excluding inside days)
    # =========================================================================
    def test_feature_12_3_thirteen_day_reentry_levels(self):
        df_inside = pd.DataFrame({
            "High": [120, 115, 110, 125],
            "Low":  [ 80,  85,  90,  95],
        })
        # Lookback 1 excluding inside days: at Bar 3, skips bars 1 and 2, reaches Bar 0
        reentry_h, reentry_l = thirteen_day_reentry_levels(df_inside, lookback_bars=1, exclude_inside_days=True)
        self.assertEqual(reentry_h.iloc[3], 120.0)
        self.assertEqual(reentry_l.iloc[3], 80.0)

    # =========================================================================
    # 12. Feature 12.4: TimeExitFilter & ParabolicWindfallExit
    # =========================================================================
    def test_feature_12_4_time_and_windfall_exits(self):
        # TimeExitFilter: 10-15 weeks (50 to 75 daily bars)
        daily_filter = TimeExitFilter(min_holding_bars=50, max_holding_bars=75, timeframe="daily")
        self.assertFalse(daily_filter.should_exit(49))
        self.assertTrue(daily_filter.should_exit(75))
        self.assertTrue(daily_filter.should_exit(80))

        # Weekly timeframe
        weekly_filter = TimeExitFilter(timeframe="weekly")
        self.assertEqual(weekly_filter.max_holding_bars, 15)
        self.assertTrue(weekly_filter.should_exit(15))

        # ParabolicWindfallExit
        windfall = ParabolicWindfallExit(expansion_multiplier=2.5, atr_window=10)
        # Entry 100, bar range = 15 points (high 120, low 105, close 119), atr = 4.0
        # 15 >= 2.5 * 4.0 (10.0) -> surge
        should_exit, reason = windfall.check_exit(
            entry_price=100.0,
            bar_high=120.0,
            bar_low=105.0,
            bar_close=119.0,
            atr=4.0,
            side="LONG",
        )
        self.assertTrue(should_exit)
        self.assertIn("PARABOLIC_WINDFALL", reason)

    # =========================================================================
    # 13. Feature 12.5: Anti-Plunging & Position Sizing Discipline
    # =========================================================================
    def test_feature_12_5_anti_plunging_position_sizer(self):
        # Account equity $100,000, 1% risk fraction = $1,000 risk
        sizer = AntiPlungingPositionSizer(
            account_equity=100000.0,
            risk_fraction=0.01,
            contract_point_value=50.0, # ES multiplier
            max_contracts_cap=10,
        )
        self.assertEqual(sizer.calculate_max_risk_dollars(), 1000.0)

        # Stop distance = 10 ES points ($500 loss per contract) -> 1000 // 500 = 2 contracts
        contracts = sizer.calculate_contracts(stop_distance_points=10.0)
        self.assertEqual(contracts, 2)

        # Tight stop: 1 point ($50 loss) -> 1000 // 50 = 20 contracts -> capped at 10
        contracts_capped = sizer.calculate_contracts(stop_distance_points=1.0)
        self.assertEqual(contracts_capped, 10)

    # =========================================================================
    # 14. Feature 13.1: The 60-Second Weekly Screening Strategy
    # =========================================================================
    def test_feature_13_1_one_minute_weekly_scan(self):
        scan = OneMinuteWeeklyScan(trend_window=52, cot_window=26, trend_lag=2)
        res = scan.generate_signals(self.df)
        self.assertIn("sma_52w", res.columns)
        self.assertIn("trend_up", res.columns)
        self.assertIn("trend_down", res.columns)
        self.assertIn("one_minute_signal", res.columns)
        self.assertIn("one_minute_trigger", res.columns)

        buys = res[res["one_minute_signal"] == 1]
        if len(buys) > 0:
            self.assertTrue((buys["trend_up"] == 1).all())
            self.assertTrue((buys["cot_index_26w"] >= 80.0).all())

    # =========================================================================
    # 15. Feature 13.2: Gold / U.S. Dollar Spread Valuation Index
    # =========================================================================
    def test_feature_13_2_gold_dollar_valuation(self):
        dates = pd.date_range("2023-01-01", periods=30, freq="W-FRI")
        # Overbought scenario: Gold rallies violently while Dollar crashes
        gold = pd.Series(np.linspace(1800, 2100, 30), index=dates)
        usd = pd.Series(np.linspace(105, 95, 30), index=dates)

        ratio = intermarket_gold_dollar_valuation(gold, usd, short_window=3, long_window=21)
        res = gold_usd_valuation_signals(gold, usd, short_window=3, long_window=21)
        self.assertIn("gold_usd_spread_ratio", res.columns)
        self.assertIn("gold_overbought_short_alert", res.columns)
        self.assertIn("gold_undervalued_long_alert", res.columns)
        # Because gold spread surged dramatically, ratio > 0.30 overbought alert must fire
        self.assertTrue((res["gold_overbought_short_alert"].iloc[-5:] == 1).any())

    # =========================================================================
    # 16. Feature 13.3: Gold Seasonal Cycle
    # =========================================================================
    def test_feature_13_3_gold_seasonal_cycle(self):
        dates = pd.date_range("2024-01-01", periods=12, freq="MS") # First of each month
        res = gold_seasonal_cycle(dates)
        # July (month 7) is sowing window
        self.assertEqual(res.loc[pd.Timestamp("2024-07-01"), "gold_sowing_window"], 1)
        self.assertEqual(res.loc[pd.Timestamp("2024-07-01"), "gold_reaping_window"], 0)
        self.assertEqual(res.loc[pd.Timestamp("2024-07-01"), "gold_seasonal_bias"], 1)

        # December (month 12), January (1), February (2) are reaping windows
        self.assertEqual(res.loc[pd.Timestamp("2024-12-01"), "gold_reaping_window"], 1)
        self.assertEqual(res.loc[pd.Timestamp("2024-01-01"), "gold_reaping_window"], 1)
        self.assertEqual(res.loc[pd.Timestamp("2024-02-01"), "gold_reaping_window"], 1)
        self.assertEqual(res.loc[pd.Timestamp("2024-12-01"), "gold_seasonal_bias"], -1)

    # =========================================================================
    # 17. Feature 14.1 - 14.3: Classical Chart Patterns & 1-2-3 Reversals
    # =========================================================================
    def test_feature_14_chart_patterns_and_123_reversal(self):
        # 1. Flag bracket execution
        flag = FlagBracketExecution(high_trigger=105.0, low_trigger=95.0)
        # Bar tests high trigger
        order = flag.evaluate_bar(bar_high=106.0, bar_low=98.0)
        self.assertIsNotNone(order)
        self.assertEqual(order["action"], "BUY_ENTER")
        self.assertEqual(order["entry_price"], 105.0)
        self.assertEqual(order["initial_stop"], 95.0) # Opposite bracket order becomes initial stop

        # 2. 1-2-3 Reversal detection
        res_123 = detect_123_reversal(self.df, lookback=10)
        self.assertIn("reversal_123_top_trigger", res_123.columns)
        self.assertIn("reversal_123_bottom_trigger", res_123.columns)

        # 3. Classical chart breakouts
        res_bk = classical_chart_breakouts(self.df, lookback=10)
        self.assertIn("pattern_breakout_long", res_bk.columns)
        self.assertIn("pattern_breakout_short", res_bk.columns)

    # =========================================================================
    # 18. Feature 15.1: Tripartite Execution Architecture
    # =========================================================================
    def test_feature_15_1_tripartite_macro_model(self):
        model = TripartiteMacroModel(
            oi_window=52,
            cot_window=26,
            trend_window=39,
            oi_bull=30.0,
            oi_bear=70.0,
            cot_bull=70.0,
            cot_bear=30.0,
            cot_metric="cot_index",
        )
        res = model.generate_signals(self.df)
        self.assertIn("sma_39w", res.columns)
        self.assertIn("tripartite_signal", res.columns)
        self.assertIn("tripartite_trigger", res.columns)
        self.assertTrue(set(res["tripartite_signal"].unique()).issubset({-1, 0, 1}))


if __name__ == "__main__":
    unittest.main()

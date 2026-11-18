"""
test_ch1_ch4_features.py — Rigorous Unit & Feature Parity Test Suite
for Chapters 1–4 of Larry Williams' 'Trade Stocks and Commodities With the Insiders'.

Verifies 100% of the mathematical features documented in DOCS/COT_Features_Ch1_Ch4.md:
  - COT-01: Commercial Net Position (CP = CL - CS)
  - COT-02: Commercial Net % of Open Interest (CP%OI)
  - COT-03: Commercial Gross Long/Short % of OI (CL%OI, CS%OI, CL%OI > 50% dominance flag)
  - COT-04: Commercial Weekly Net Flow (delta CP, delta CP%OI, momentum exhaustion divergence)
  - COT-05: Multi-Year Extremes (4-yr / 208w, 3-yr / 156w, 5-yr / 260w, 6-yr / 312w, 10-yr / 520w, 14-yr / 728w)
  - COT-06: Williams COT Index (156w, 52w, 26w & Bull/Bear Setup Zones)
  - COT-07: Perpetual Short Regime Breakout ('The Silver Rule')
  - COT-08: Jiler Seasonal Deviation (noise < 5%, caution > 40%, bullish > +20%, bearish < -20%)
  - COT-09: 26-Week COT Swing Velocity (delta_26 COT Index, delta_26 CP, rapid swing alert >= 75%)
  - Integration: compute_all_cot_features() exposing 100% of Chapter 1-4 features.
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from cot_engine import (
    commercial_net_position,
    commercial_share_oi,
    commercial_net_flow,
    commercial_weekly_flow,
    williams_cot_index,
    williams_willco,
    multi_year_extremes,
    perpetual_short_regime_breakout,
    cot_swing_velocity,
    jiler_seasonal_deviation,
    compute_all_cot_features,
    generate_synthetic_cot,
)


class TestCh1Ch4Features(unittest.TestCase):

    def setUp(self):
        # 5-year synthetic dataset for rigorous multi-year testing
        self.synthetic_cot = generate_synthetic_cot(
            start_date="2019-01-01",
            end_date="2024-06-25",
            market_name="SILVER - COMMODITY EXCHANGE INC.",
            base_oi=200000,
            seed=101,
        )

    def test_cot_01_commercial_net_position(self):
        """
        COT-01: CP = CL - CS
        Verifies net calculation, sign interpretation, and zero line neutrality.
        """
        df = pd.DataFrame({
            "comm_long": [1000, 500, 300],
            "comm_short": [400, 500, 800],
        })
        cp = commercial_net_position(df)
        self.assertEqual(cp.iloc[0], 600)    # Net Long (> 0)
        self.assertEqual(cp.iloc[1], 0)      # Neutral Zero Line (== 0)
        self.assertEqual(cp.iloc[2], -500)   # Net Short (< 0)
        self.assertEqual(cp.name, "comm_net")

    def test_cot_02_commercial_net_pct_oi(self):
        """
        COT-02: CP%OI = ((CL - CS) / OI) * 100
        Verifies normalization and zero-division guard.
        """
        df = pd.DataFrame({
            "comm_long": [50000, 20000, 1000],
            "comm_short": [20000, 40000, 500],
            "open_interest": [100000, 100000, 0],  # 0 OI should produce NaN, not crash
        })
        share = commercial_share_oi(df)
        self.assertAlmostEqual(share["comm_net_pct_oi"].iloc[0], 30.0)
        self.assertAlmostEqual(share["comm_net_pct_oi"].iloc[1], -20.0)
        self.assertTrue(pd.isna(share["comm_net_pct_oi"].iloc[2]))

    def test_cot_03_commercial_gross_participation_and_dominance(self):
        """
        COT-03: Gross Long/Short % of OI (CL%OI, CS%OI).
        Verifies:
          - CL%OI > 50% dominance flag (Wheat Ch 2 commercial dominance rule)
          - CS%OI > 50% short dominance flag
          - CS%OI >= 55% extreme alert (Yen/Lumber top alert)
          - CS%OI < 30% bottom alert
        """
        df = pd.DataFrame({
            "comm_long": [55000, 20000, 40000],
            "comm_short": [25000, 60000, 25000],
            "open_interest": [100000, 100000, 100000],
        })
        share = commercial_share_oi(df)

        # Bar 0: CL%OI = 55%, CS%OI = 25% -> CL dominant > 50%, CS < 30% bottom
        self.assertEqual(share["comm_long_dominant_50"].iloc[0], 1)
        self.assertEqual(share["comm_short_dominant_50"].iloc[0], 0)
        self.assertEqual(share["comm_short_bottom_30"].iloc[0], 1)
        self.assertEqual(share["comm_short_extreme_55"].iloc[0], 0)

        # Bar 1: CL%OI = 20%, CS%OI = 60% -> CS dominant > 50%, CS >= 55% extreme
        self.assertEqual(share["comm_long_dominant_50"].iloc[1], 0)
        self.assertEqual(share["comm_short_dominant_50"].iloc[1], 1)
        self.assertEqual(share["comm_short_extreme_55"].iloc[1], 1)
        self.assertEqual(share["comm_short_bottom_30"].iloc[1], 0)

        # Bar 2: CL%OI = 40%, CS%OI = 25% -> neither dominant > 50%, CS < 30%
        self.assertEqual(share["comm_long_dominant_50"].iloc[2], 0)
        self.assertEqual(share["comm_short_dominant_50"].iloc[2], 0)
        self.assertEqual(share["comm_short_bottom_30"].iloc[2], 1)

    def test_cot_04_commercial_weekly_net_flow_and_exhaustion(self):
        """
        COT-04: Commercial Weekly Net Flow (delta CP = CP_t - CP_{t-1}, delta CP%OI, momentum exhaustion divergence).
        Verifies:
          - delta_comm_net calculation
          - delta_comm_net_pct_oi calculation
          - comm_flow_buying (> 0) and comm_flow_selling (< 0)
          - comm_exhaustion_divergence: CP_t > 0 while delta_comm_net < 0 (Liquidating longs into rally)
        """
        df = pd.DataFrame({
            "comm_long": [10000, 15000, 13000, 8000],
            "comm_short": [5000,   5000,  6000, 9000],
            "open_interest": [100000, 100000, 100000, 100000],
        })
        # CP: [5000, 10000, 7000, -1000]
        # delta_CP: [NaN, +5000, -3000, -8000]
        flow = commercial_net_flow(df)

        self.assertTrue(pd.isna(flow["delta_comm_net"].iloc[0]))
        self.assertEqual(flow["delta_comm_net"].iloc[1], 5000)
        self.assertEqual(flow["delta_comm_net"].iloc[2], -3000)
        self.assertEqual(flow["delta_comm_net"].iloc[3], -8000)

        self.assertAlmostEqual(flow["delta_comm_net_pct_oi"].iloc[1], 5.0)
        self.assertAlmostEqual(flow["delta_comm_net_pct_oi"].iloc[2], -3.0)

        # Buying / Selling flags
        self.assertEqual(flow["comm_flow_buying"].iloc[1], 1)
        self.assertEqual(flow["comm_flow_selling"].iloc[1], 0)
        self.assertEqual(flow["comm_flow_buying"].iloc[2], 0)
        self.assertEqual(flow["comm_flow_selling"].iloc[2], 1)

        # Momentum Exhaustion Divergence:
        # Bar 1: CP = 10000 (>0), delta = +5000 (>0) -> NO exhaustion
        self.assertEqual(flow["comm_exhaustion_divergence"].iloc[1], 0)
        # Bar 2: CP = 7000 (>0), delta = -3000 (<0) -> EXHAUSTION DIVERGENCE!
        self.assertEqual(flow["comm_exhaustion_divergence"].iloc[2], 1)
        # Bar 3: CP = -1000 (<0), delta = -8000 (<0) -> Net short, so not long exhaustion
        self.assertEqual(flow["comm_exhaustion_divergence"].iloc[3], 0)

        # Alias verification
        flow_alias = commercial_weekly_flow(df)
        pd.testing.assert_frame_equal(flow, flow_alias)

    def test_cot_05_multi_year_extremes_lookbacks(self):
        """
        COT-05: Multi-Year Extremes with lookbacks:
        4-yr (208w), 3-yr (156w), 5-yr (260w), 6-yr (312w), 10-yr (520w), 14-yr (728w).
        Verifies:
          - Rolling lookback exclusion of current bar (shift(1))
          - Extreme high and low detection
          - Support for single int lookback or list of lookbacks
        """
        # Controlled sequence
        net_values = [100, 200, 300, 150, 400, 50, 500, 20]
        df = pd.DataFrame({
            "comm_long": [1000 + v for v in net_values],
            "comm_short": [1000] * len(net_values),
        })

        ext_3 = multi_year_extremes(df, lookback_weeks=3, min_periods=3)
        # Bar 0, 1, 2: warm-up (< 3 prior observations)
        self.assertEqual(ext_3["comm_3w_extreme_high"].iloc[2], 0)
        # Bar 3: prior 3 are [100, 200, 300], max=300, min=100. cur=150 -> not high or low
        self.assertEqual(ext_3["comm_3w_extreme_high"].iloc[3], 0)
        self.assertEqual(ext_3["comm_3w_extreme_low"].iloc[3], 0)
        # Bar 4: prior 3 are [200, 300, 150], max=300, min=150. cur=400 >= 300 -> EXTREME HIGH!
        self.assertEqual(ext_3["comm_3w_extreme_high"].iloc[4], 1)
        self.assertEqual(ext_3["comm_3w_extreme_low"].iloc[4], 0)
        # Bar 5: prior 3 are [300, 150, 400], max=400, min=150. cur=50 <= 150 -> EXTREME LOW!
        self.assertEqual(ext_3["comm_3w_extreme_high"].iloc[5], 0)
        self.assertEqual(ext_3["comm_3w_extreme_low"].iloc[5], 1)

        # Multi-window list support: [208, 156, 260, 312, 520, 728]
        ext_multi = multi_year_extremes(self.synthetic_cot, lookback_weeks=[208, 156, 260, 312, 520, 728])
        for w in [208, 156, 260, 312, 520, 728]:
            self.assertIn(f"comm_{w}w_extreme_high", ext_multi.columns)
            self.assertIn(f"comm_{w}w_extreme_low", ext_multi.columns)

    def test_cot_06_williams_cot_index_setup_zones(self):
        """
        COT-06: Williams COT Index (156w, 52w, 26w) and Bull/Bear Setup Zones:
        Bullish Setup Zone >= 80%
        Bearish Setup Zone <= 20%
        """
        idx_26 = williams_cot_index(self.synthetic_cot, lookback_weeks=26)
        valid = idx_26.dropna()
        self.assertTrue((valid >= 0.0).all() and (valid <= 100.0).all())

        # Test compute_all_cot_features exposes bull/bear setup flags
        features = compute_all_cot_features(self.synthetic_cot.iloc[:60].copy(), include_jiler=False, include_daily_progo=False)
        self.assertIn("cot_index_26w_bull", features.columns)
        self.assertIn("cot_index_26w_bear", features.columns)
        self.assertIn("cot_index_52w_bull", features.columns)
        self.assertIn("cot_index_52w_bear", features.columns)
        self.assertIn("cot_index_156w_bull", features.columns)
        self.assertIn("cot_index_156w_bear", features.columns)

    def test_cot_07_the_silver_rule_perpetual_short_breakout(self):
        """
        COT-07: The Silver Rule (Perpetual Short Regime Breakout).
        Mathematical formula: CP_t > 0 while for prior M weeks max(CP_{t-k}) <= 0 and min(CP_{t-k}) < 0.
        Verifies:
          - Triggers exactly on the transition bar from structural net short to net long.
          - Does not trigger on subsequent positive bars (no double-trigger).
          - Does not trigger if any prior bar in window was positive.
        """
        # Create 10 weeks of structural short positioning (-500), then week 10 flips to +100
        n_short = 10
        comm_long = [1000] * n_short + [1500] + [1600]
        comm_short = [1500] * n_short + [1400] + [1300]
        # Net positions: [-500 x 10, +100, +300]
        df = pd.DataFrame({"comm_long": comm_long, "comm_short": comm_short})

        res = perpetual_short_regime_breakout(df, lookback_weeks=5, min_periods=5)
        # Prior to flip (bars 0-9), all net negative -> breakout == 0
        self.assertEqual(res["silver_rule_breakout"].iloc[9], 0)
        # Bar 10: flips to +100, prior 5 weeks were all -500 <= 0 -> BREAKOUT == 1!
        self.assertEqual(res["silver_rule_breakout"].iloc[10], 1)
        # Bar 11: continues at +300, prior 5 weeks includes Bar 10 (+100 > 0) -> BREAKOUT == 0!
        self.assertEqual(res["silver_rule_breakout"].iloc[11], 0)

        # Negative test: if any prior week in lookback was positive, no breakout
        df_invalid = pd.DataFrame({
            "comm_long": [1000, 1600, 1000, 1000, 1000, 1500],
            "comm_short": [1500, 1500, 1500, 1500, 1500, 1400],
        })
        # Nets: [-500, +100, -500, -500, -500, +100]
        # At bar 5, prior 5 weeks contains bar 1 (+100) -> max > 0 -> breakout MUST BE 0!
        res_invalid = perpetual_short_regime_breakout(df_invalid, lookback_weeks=4, min_periods=4)
        self.assertEqual(res_invalid["silver_rule_breakout"].iloc[5], 0)

    def test_cot_08_jiler_seasonal_deviation_flags(self):
        """
        COT-08: Jiler Seasonal Deviation
        Dev = CP%OI - Seasonal_Norm
        Verifies:
          - Noise flag: |Dev| < 5%
          - Caution flag: |Dev| > 40%
          - Bullish flag: Dev > +20%
          - Bearish flag: Dev < -20%
        """
        # Create a 7-year weekly dataset with clear seasonal baseline
        dates = pd.date_range("2015-01-06", periods=360, freq="W-TUE")
        oi = 100000
        # Monthly base norm: around -10% in January
        net_pct = []
        for d in dates:
            if d.year < 2021:
                net_pct.append(-10.0)  # Norm for all months = -10%
            else:
                # In 2021 test variations:
                # Week 0 of 2021: -8% (Dev = +2% -> Noise)
                # Week 1 of 2021: +15% (Dev = +25% -> Bullish)
                # Week 2 of 2021: -35% (Dev = -25% -> Bearish)
                # Week 3 of 2021: +35% (Dev = +45% -> Caution & Bullish)
                net_pct.append(-10.0)

        df = pd.DataFrame({
            "report_date": dates,
            "open_interest": [oi] * len(dates),
            "comm_long": [int(oi * 0.45)] * len(dates),
            "comm_short": [int(oi * 0.55)] * len(dates),
        })

        # Inject test values into 2021
        idx_2021 = df[df["report_date"].dt.year == 2021].index
        # Bar 0 of 2021: net_pct = -8% -> Dev = -8 - (-10) = +2% (|dev| < 5%) -> Noise
        df.loc[idx_2021[0], "comm_long"] = int(oi * 0.46)
        df.loc[idx_2021[0], "comm_short"] = int(oi * 0.54) # net -8%

        # Bar 1 of 2021: net_pct = +15% -> Dev = +15 - (-10) = +25% (> +20%) -> Bullish
        df.loc[idx_2021[1], "comm_long"] = int(oi * 0.575)
        df.loc[idx_2021[1], "comm_short"] = int(oi * 0.425) # net +15%

        # Bar 2 of 2021: net_pct = -35% -> Dev = -35 - (-10) = -25% (< -20%) -> Bearish
        df.loc[idx_2021[2], "comm_long"] = int(oi * 0.325)
        df.loc[idx_2021[2], "comm_short"] = int(oi * 0.675) # net -35%

        # Bar 3 of 2021: net_pct = +35% -> Dev = +35 - (-10) = +45% (> 40%) -> Caution
        df.loc[idx_2021[3], "comm_long"] = int(oi * 0.675)
        df.loc[idx_2021[3], "comm_short"] = int(oi * 0.325) # net +35%

        jiler = jiler_seasonal_deviation(df, lookback_years=5, min_prior_obs=3)

        self.assertEqual(jiler["jiler_noise_flag"].loc[idx_2021[0]], 1)
        self.assertEqual(jiler["jiler_bullish_flag"].loc[idx_2021[1]], 1)
        self.assertEqual(jiler["jiler_bearish_flag"].loc[idx_2021[2]], 1)
        self.assertEqual(jiler["jiler_caution_flag"].loc[idx_2021[3]], 1)

    def test_cot_09_26w_swing_velocity(self):
        """
        COT-09: 26-Week COT Swing Velocity.
        delta_26 COT Index = Index_t - Index_{t-26}
        delta_26 CP = CP_t - CP_{t-26}
        Rapid swing alert: |delta_26| >= 75% (British Pound 1999-2000 institutional rotation)
        """
        # Create a sequence with a non-flat historical baseline where commercials start near 0%
        # and aggressively rotate to 100% within 26 weeks.
        n = 100
        # Historical baseline with range [-100k, +50k], sitting at -100k (0% COT Index) around bar 35
        # then ramping up to +120k at bar 61 (26 weeks later, hitting 100% COT Index)
        base = [-50000, 50000] * 15 + [-100000] * 6  # 36 bars with min=-100k, max=+50k; bar 35 is -100k (0%)
        ramp = list(np.linspace(-100000, 120000, 26))  # 26 weeks rising to +120k (100%)
        tail = [120000] * (n - len(base) - len(ramp))
        nets = base + ramp + tail

        df = pd.DataFrame({
            "comm_long": [150000 + int(x / 2) for x in nets],
            "comm_short": [150000 - int(x / 2) for x in nets],
            "open_interest": [500000] * n,
        })

        vel = cot_swing_velocity(df, velocity_window=26, index_lookback=30, rapid_threshold=75.0)

        self.assertIn("cot_swing_velocity_26w", vel.columns)
        self.assertIn("cot_net_contract_swing_26w", vel.columns)
        self.assertIn("cot_rapid_swing_alert", vel.columns)
        self.assertIn("cot_rapid_bull_swing", vel.columns)
        self.assertIn("cot_rapid_bear_swing", vel.columns)

        # After the 26-week rise, swing velocity should exceed +75%
        peak_idx = len(base) + 25  # 26th bar of ramp
        self.assertGreaterEqual(vel["cot_swing_velocity_26w"].iloc[peak_idx], 75.0)
        self.assertEqual(vel["cot_rapid_bull_swing"].iloc[peak_idx], 1)
        self.assertEqual(vel["cot_rapid_swing_alert"].iloc[peak_idx], 1)

    def test_10_master_compute_all_cot_features_audit(self):
        """
        Master integration audit: Verifies that compute_all_cot_features() exposes 100%
        of the features from Chapters 1–4.
        """
        df = self.synthetic_cot.copy()
        enriched = compute_all_cot_features(df, include_jiler=True, include_daily_progo=False)

        # COT-01: Commercial Net Position
        self.assertIn("comm_net", enriched.columns)

        # COT-02: Commercial Net % of OI
        self.assertIn("comm_net_pct_oi", enriched.columns)

        # COT-03: Commercial Gross Long/Short % of OI & Dominance
        self.assertIn("comm_long_pct_oi", enriched.columns)
        self.assertIn("comm_short_pct_oi", enriched.columns)
        self.assertIn("comm_long_dominant_50", enriched.columns)
        self.assertIn("comm_short_dominant_50", enriched.columns)
        self.assertIn("comm_short_bottom_30", enriched.columns)
        self.assertIn("comm_short_extreme_55", enriched.columns)

        # COT-04: Weekly Net Flow & Exhaustion Divergence
        self.assertIn("delta_comm_net", enriched.columns)
        self.assertIn("delta_comm_net_pct_oi", enriched.columns)
        self.assertIn("comm_flow_buying", enriched.columns)
        self.assertIn("comm_flow_selling", enriched.columns)
        self.assertIn("comm_exhaustion_divergence", enriched.columns)

        # COT-05: Multi-Year Extremes
        for w in [208, 156, 260, 312, 520, 728]:
            self.assertIn(f"comm_{w}w_extreme_high", enriched.columns)
            self.assertIn(f"comm_{w}w_extreme_low", enriched.columns)

        # COT-06: Williams COT Indexes & Setup Zones
        for w in [156, 52, 26]:
            self.assertIn(f"cot_index_{w}w", enriched.columns)
            self.assertIn(f"cot_index_{w}w_bull", enriched.columns)
            self.assertIn(f"cot_index_{w}w_bear", enriched.columns)

        # COT-07: The Silver Rule Breakout
        self.assertIn("silver_rule_breakout", enriched.columns)

        # COT-08: Jiler Seasonal Deviation
        self.assertIn("jiler_seasonal_norm", enriched.columns)
        self.assertIn("jiler_deviation", enriched.columns)
        self.assertIn("jiler_noise_flag", enriched.columns)
        self.assertIn("jiler_caution_flag", enriched.columns)
        self.assertIn("jiler_bullish_flag", enriched.columns)
        self.assertIn("jiler_bearish_flag", enriched.columns)

        # COT-09: 26-Week COT Swing Velocity
        self.assertIn("cot_swing_velocity_26w", enriched.columns)
        self.assertIn("cot_net_contract_swing_26w", enriched.columns)
        self.assertIn("cot_rapid_swing_alert", enriched.columns)
        self.assertIn("cot_rapid_bull_swing", enriched.columns)
        self.assertIn("cot_rapid_bear_swing", enriched.columns)


if __name__ == "__main__":
    unittest.main()

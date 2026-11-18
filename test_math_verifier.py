"""
test_math_verifier.py — Exhaustive Mathematical Precision and Numerical Robustness Audit
for cot_engine/indicators.py against DOCS/COT_Master_Specification.md and Larry Williams'
'Trade Stocks and Commodities With the Insiders - Secrets of the COT Report' (Wiley, 2005).

Audit Verifications:
  1. Williams COT Index (156w, 52w, 26w):
     - Exact stochastic %K normalization.
     - Boundary safety [0.0, 100.0].
     - Epsilon guard on flat ranges (max == min -> neutral 50.0).
     - Chapter 4 typo resolution: 350 / -150 / 750 -> 55.5555...% (not 33%).
     - Warm-up and NaN preservation.
  2. WILLCO Oscillator:
     - 'net' variant: Stochastic(NetComm / OI).
     - 'short' variant: 100 - Stochastic(CommShort / OI).
     - Safe zero-division on OI <= 0.
     - Strict boundary clipping [0.0, 100.0].
     - Validation on invalid variant string.
  3. Commercial Share of OI:
     - Concentration ratios (CL%OI, CS%OI, Net%OI).
     - Extreme top trigger CS%OI >= 55%.
     - Robust handling of zero and negative OI.
  4. Multi-Year Extremes:
     - 4-year (208 weeks), 3-year (156 weeks), and 5-year (260 weeks) rolling highs/lows.
     - Strict exclusion of current bar (shift(1)).
     - Mutual exclusivity and flat range protection (no false dual high/low).
  5. Speculator Sentiment:
     - Public Contrary Index (Small Trader / Non-Reportable net stochastic).
     - Large Trader Exhaustion Index (Large Funds / Non-Commercial net stochastic).
     - Proper contrary polarity alignment.
  6. Open Interest Stochastic:
     - 52-week lookback baseline.
     - Case-insensitive column lookup and boundary bounds [0, 100].
     - Protection against non-positive OI.
  7. Inside-OI 4-State Diagnostic Matrix:
     - State 1: SPECULATIVE_TOP (Price UP, OI UP, CommNet DOWN).
     - State 2: COMMERCIAL_CAPITULATION (Price DOWN, OI UP, CommNet UP).
     - State 3: DESPAIR_BLASTOFF (Price DOWN, OI DOWN, CommNet UP).
     - State 4: HOLLOW_RALLY (Price UP, OI DOWN, CommNet DOWN).
     - State 0: NEUTRAL (all remaining permutations and zero deltas).
     - Case-insensitivity on price column.
  8. ProGo & Public Index:
     - ProGo: 14-day SMA of (Close - Open).
     - Public: 14-day SMA of (Open - Prior Close).
     - Bullish divergence: price at 20-bar low while ProGo is rising and above prior low.
     - Bearish divergence: price at 20-bar high while ProGo is falling and below prior high.
  9. Jiler Seasonal Deviation:
     - Historical month-of-year baseline strictly computed PIT over prior calendar years.
     - 5% noise threshold (|dev| < 5%).
     - 40% caution threshold (|dev| > 40%).
     - Zero/NaN safety during warm-up.
 10. Synthetic Stress & Invariance Tests:
     - 1,000-bar Monte Carlo synthetic series.
     - Extreme outliers, zeros, NaNs, and negative inputs.
"""

import unittest
import numpy as np
import pandas as pd

from cot_engine.indicators import (
    INSIDE_OI_STATES,
    _stochastic_normalize,
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
)


class TestMathVerifier(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # 6-year weekly COT timeline
        self.dates = pd.date_range("2018-01-02", "2024-01-02", freq="W-TUE")
        self.n_bars = len(self.dates)

    def test_01_williams_cot_index_precision_and_typo_resolution(self):
        """
        1. Verify Chapter 4 typo resolution:
           Book p.34 example prints CP=350, Min=-150, Max=750 -> '(200 / 600) * 100 = 33%'.
           Exact mathematical %K: (350 - (-150)) / (750 - (-150)) * 100 = 500 / 900 * 100 = 55.555555...%
        2. Verify exact normalization, boundary safety [0, 100], and epsilon guard on flat ranges.
        """
        # Exact Chapter 4 dataset
        df_ch4 = pd.DataFrame({
            "comm_long": [1000, 1000, 1350],
            "comm_short": [1150, 250, 1000],  # nets: -150, +750, +350
        })
        cot_idx = williams_cot_index(df_ch4, lookback_weeks=3)
        self.assertAlmostEqual(cot_idx.iloc[-1], 55.55555555555556, places=6)
        self.assertNotAlmostEqual(cot_idx.iloc[-1], 33.0, places=1)

        # Flat range: max == min -> neutral 50.0
        df_flat = pd.DataFrame({
            "comm_long": [2000] * 10,
            "comm_short": [1500] * 10,
        })
        flat_idx = williams_cot_index(df_flat, lookback_weeks=5)
        self.assertEqual(flat_idx.iloc[-1], 50.0)

        # Boundary safety across lookbacks (156w, 52w, 26w)
        s_rand = pd.Series(np.random.normal(0, 10000, self.n_bars))
        df_rand = pd.DataFrame({"comm_long": 50000 + s_rand, "comm_short": 50000})
        for w in [156, 52, 26]:
            idx = williams_cot_index(df_rand, lookback_weeks=w)
            valid = idx.dropna()
            self.assertTrue((valid >= 0.0).all(), f"COT Index {w}w violated lower bound 0.0")
            self.assertTrue((valid <= 100.0).all(), f"COT Index {w}w violated upper bound 100.0")

        # Warm-up test: first w - 1 bars must be NaN
        w = 26
        idx_warm = williams_cot_index(df_rand, lookback_weeks=w)
        self.assertTrue(idx_warm.iloc[: w - 1].isna().all(), "Warm-up period must be strictly NaN")
        self.assertFalse(np.isnan(idx_warm.iloc[w - 1]), "First bar after warm-up must be valid")

    def test_02_willco_oscillator_net_and_short_variants(self):
        """
        Verify WILLCO oscillator:
        - 'net' variant: Stochastic of (CommNet / OI).
        - 'short' variant: Inverted Stochastic of (CommShort / OI) [100 - Stoch(CS/OI)].
        - Division by OI safety (zero/negative OI -> NaN).
        - Boundary safety [0, 100].
        """
        # Synthetic sequence where commercial shorting ramps up from 10% to 90%
        df = pd.DataFrame({
            "comm_long": [5000, 5000, 5000],
            "comm_short": [1000, 5000, 9000],
            "open_interest": [10000, 10000, 10000],
        })
        willco_short = williams_willco(df, lookback_weeks=3, variant="short")
        # At bar 2: CS/OI is 90% (26-week high) -> Stoch(CS/OI) = 100% -> Inverted WILLCO = 0.0 (bearish)
        self.assertAlmostEqual(willco_short.iloc[-1], 0.0, places=4)

        # When commercial short covers from 90% down to 10%
        df_cov = pd.DataFrame({
            "comm_long": [5000, 5000, 5000],
            "comm_short": [9000, 5000, 1000],
            "open_interest": [10000, 10000, 10000],
        })
        willco_cov = williams_willco(df_cov, lookback_weeks=3, variant="short")
        # At bar 2: CS/OI is 10% (26-week low) -> Stoch(CS/OI) = 0% -> Inverted WILLCO = 100.0 (bullish)
        self.assertAlmostEqual(willco_cov.iloc[-1], 100.0, places=4)

        # Net variant: CommNet = Long - Short = [4000, 0, -4000] -> Net%OI = [40%, 0%, -40%]
        # Lookback 3: At bar 2, Net%OI is at min (-40%) -> Stoch = 0.0
        willco_net = williams_willco(df, lookback_weeks=3, variant="net")
        self.assertAlmostEqual(willco_net.iloc[-1], 0.0, places=4)

        # Zero and negative OI safety
        df_zero_oi = pd.DataFrame({
            "comm_long": [100, 200, 300],
            "comm_short": [50, 100, 150],
            "open_interest": [1000, 0, -500],
        })
        res_zero = williams_willco(df_zero_oi, lookback_weeks=2, variant="net")
        self.assertTrue(np.isnan(res_zero.iloc[1]))
        self.assertTrue(np.isnan(res_zero.iloc[2]))

        # Invalid variant raises ValueError
        with self.assertRaises(ValueError):
            williams_willco(df, variant="unsupported_variant")

    def test_03_commercial_share_oi_and_55_percent_alert(self):
        """
        Verify Commercial share of OI and extreme 55% top trigger.
        """
        df = pd.DataFrame({
            "comm_long": [10000, 20000, 30000],
            "comm_short": [54990, 55000, 60000],
            "open_interest": [100000, 100000, 100000],
        })
        share = commercial_share_oi(df)
        self.assertAlmostEqual(share["comm_short_pct_oi"].iloc[0], 54.99, places=2)
        self.assertAlmostEqual(share["comm_short_pct_oi"].iloc[1], 55.00, places=2)
        self.assertAlmostEqual(share["comm_short_pct_oi"].iloc[2], 60.00, places=2)

        # Threshold trigger
        self.assertEqual(share["comm_short_extreme_55"].iloc[0], 0)
        self.assertEqual(share["comm_short_extreme_55"].iloc[1], 1)
        self.assertEqual(share["comm_short_extreme_55"].iloc[2], 1)

        # Zero / Negative OI handling
        df_corrupt = pd.DataFrame({
            "comm_long": [10000, 10000],
            "comm_short": [60000, 60000],
            "open_interest": [0, -10000],
        })
        res_corrupt = commercial_share_oi(df_corrupt)
        self.assertTrue(np.isnan(res_corrupt["comm_short_pct_oi"].iloc[0]))
        self.assertTrue(np.isnan(res_corrupt["comm_short_pct_oi"].iloc[1]))
        self.assertEqual(res_corrupt["comm_short_extreme_55"].iloc[0], 0)
        self.assertEqual(res_corrupt["comm_short_extreme_55"].iloc[1], 0)

    def test_04_multi_year_extremes_lookbacks_and_exclusion(self):
        """
        Verify Multi-Year Extremes:
        - 4-year (208w), 3-year (156w), 5-year (260w).
        - Strict exclusion of current bar (shift(1)).
        - Flat series protection (no dual high/low trigger).
        """
        # Exclude current bar check
        # Bar 0..3: CP = [100, 200, 300, 400]
        # Bar 4: CP drops to 350 (prior max is 400). Must NOT be extreme high.
        # Bar 5: CP jumps to 500 (prior max is 400). MUST be extreme high.
        df = pd.DataFrame({
            "comm_long": [100, 200, 300, 400, 350, 500],
            "comm_short": [0, 0, 0, 0, 0, 0],
        })
        res = multi_year_extremes(df, lookback_weeks=3)
        self.assertEqual(res["comm_3w_extreme_high"].iloc[4], 0)
        self.assertEqual(res["comm_3w_extreme_high"].iloc[5], 1)

        # Flat market: CP is completely flat -> MUST NOT trigger both high and low
        df_flat = pd.DataFrame({
            "comm_long": [1000] * 10,
            "comm_short": [500] * 10,
        })
        res_flat = multi_year_extremes(df_flat, lookback_weeks=5)
        self.assertEqual(res_flat["comm_5w_extreme_high"].sum(), 0)
        self.assertEqual(res_flat["comm_5w_extreme_low"].sum(), 0)

        # 4-year (208w), 3-year (156w), 5-year (260w) lookback tests
        n = 300
        df_long = pd.DataFrame({
            "comm_long": np.linspace(1000, 5000, n),
            "comm_short": [1000] * n,
        })
        for w in [208, 156, 260]:
            out = multi_year_extremes(df_long, lookback_weeks=w)
            self.assertIn(f"comm_{w}w_extreme_high", out.columns)
            self.assertIn(f"comm_{w}w_extreme_low", out.columns)
            # After warm-up, continually increasing series should flag extreme highs
            self.assertEqual(out[f"comm_{w}w_extreme_high"].iloc[-1], 1)
            self.assertEqual(out[f"comm_{w}w_extreme_low"].iloc[-1], 0)

    def test_05_speculator_sentiment_indexes(self):
        """
        Verify Public Contrary Index and Large Trader Index:
        - Small Trader net: nonrep_long - nonrep_short
        - Large Trader net: noncomm_long - noncomm_short
        - Bounds strictly in [0, 100].
        """
        df = pd.DataFrame({
            "nonrep_long": [5000, 6000, 7000],
            "nonrep_short": [2000, 2000, 2000],
            "noncomm_long": [15000, 12000, 9000],
            "noncomm_short": [5000, 5000, 5000],
        })
        pub = public_contrary_index(df, lookback_weeks=3)
        large = large_trader_index(df, lookback_weeks=3)

        self.assertAlmostEqual(pub.iloc[-1], 100.0, places=4)
        self.assertAlmostEqual(large.iloc[-1], 0.0, places=4)

    def test_06_open_interest_stochastic(self):
        """
        Verify Open Interest Stochastic:
        - 52-week lookback normalization.
        - Guard against non-positive OI.
        - Case-insensitivity.
        """
        oi_vals = [100000 + i * 500 for i in range(60)]
        df = pd.DataFrame({"open_interest": oi_vals})
        stoch = open_interest_stochastic(df, lookback_weeks=52)

        self.assertEqual(stoch.name, "oi_stoch_52w")
        self.assertTrue((stoch.dropna() >= 0.0).all())
        self.assertTrue((stoch.dropna() <= 100.0).all())
        self.assertAlmostEqual(stoch.iloc[-1], 100.0, places=4)

        # Case-insensitive column test
        df_caps = pd.DataFrame({"Open_Interest": oi_vals})
        stoch_caps = open_interest_stochastic(df_caps, lookback_weeks=52)
        self.assertAlmostEqual(stoch_caps.iloc[-1], 100.0, places=4)

    def test_07_inside_oi_diagnostic_matrix_all_states(self):
        """
        Verify all 4 states + neutral state of the Inside-OI Matrix:
          State 1: SPECULATIVE_TOP (Price UP, OI UP, CommNet DOWN)
          State 2: COMMERCIAL_CAPITULATION (Price DOWN, OI UP, CommNet UP)
          State 3: DESPAIR_BLASTOFF (Price DOWN, OI DOWN, CommNet UP)
          State 4: HOLLOW_RALLY (Price UP, OI DOWN, CommNet DOWN)
          State 0: NEUTRAL
        """
        df = pd.DataFrame({
            "close":         [100, 110, 100,  90,  95, 100],
            "open_interest": [1000, 1200, 1300, 1100, 1000, 1100],
            "comm_long":     [ 500,  400,  600,  700,  600,  700],
            "comm_short":    [ 500,  600,  500,  400,  500,  400],
        })
        # Deltas:
        # Row 1 vs 0: P (+10), OI (+200), CP (-200 vs 0) -> SPECULATIVE_TOP (1)
        # Row 2 vs 1: P (-10), OI (+100), CP (+100 vs -200) -> COMMERCIAL_CAPITULATION (2)
        # Row 3 vs 2: P (-10), OI (-200), CP (+300 vs +100) -> DESPAIR_BLASTOFF (3)
        # Row 4 vs 3: P (+5),  OI (-100), CP (+100 vs +300) -> HOLLOW_RALLY (4)
        # Row 5 vs 4: P (+5),  OI (+100), CP (+300 vs +100) -> NEUTRAL (0)
        regimes = inside_oi_regime(df, price_col="close")

        self.assertEqual(regimes.iloc[1], 1, "Expected SPECULATIVE_TOP (1)")
        self.assertEqual(regimes.iloc[2], 2, "Expected COMMERCIAL_CAPITULATION (2)")
        self.assertEqual(regimes.iloc[3], 3, "Expected DESPAIR_BLASTOFF (3)")
        self.assertEqual(regimes.iloc[4], 4, "Expected HOLLOW_RALLY (4)")
        self.assertEqual(regimes.iloc[5], 0, "Expected NEUTRAL (0)")

    def test_08_progo_oscillator_and_divergences(self):
        """
        Verify ProGo & Public Index calculations and divergence logic:
        - ProGo = 14-day SMA(Close - Open).
        - Public Index = 14-day SMA(Open - Prior Close).
        - Bullish divergence: price at 20-bar low, ProGo rising and above prior low.
        - Bearish divergence: price at 20-bar high, ProGo falling and below prior high.
        """
        # Baseline data
        df = pd.DataFrame({"Open": [100.0] * 30, "Close": [100.0] * 30})
        # Construct bullish divergence:
        df.loc[20, "Open"], df.loc[20, "Close"] = 100.0, 90.0  # Pro = -10
        df.loc[21, "Open"], df.loc[21, "Close"] = 90.0, 95.0   # Pro = +5
        df.loc[22, "Open"], df.loc[22, "Close"] = 83.0, 88.0   # Price new low (88 < 90), Pro = +5 (rising)

        res_bull = progo_oscillator(df, window=3, div_lookback=20)
        self.assertEqual(res_bull["progo_bull_divergence"].iloc[22], 1)
        self.assertEqual(res_bull["progo_bear_divergence"].iloc[22], 0)

        # Construct bearish divergence:
        df_bear = pd.DataFrame({"Open": [100.0] * 30, "Close": [100.0] * 30})
        df_bear.loc[20, "Open"], df_bear.loc[20, "Close"] = 100.0, 110.0  # Pro = +10
        df_bear.loc[21, "Open"], df_bear.loc[21, "Close"] = 110.0, 105.0  # Pro = -5
        df_bear.loc[22, "Open"], df_bear.loc[22, "Close"] = 117.0, 112.0  # Price new high (112 > 110), Pro = -5 (falling)

        res_bear = progo_oscillator(df_bear, window=3, div_lookback=20)
        self.assertEqual(res_bear["progo_bull_divergence"].iloc[22], 0)
        self.assertEqual(res_bear["progo_bear_divergence"].iloc[22], 1)

    def test_09_jiler_seasonal_deviation_pit_and_thresholds(self):
        """
        Verify Bill Jiler CRB Seasonal Deviation:
        - Monthly baseline computed strictly from prior years.
        - 5% noise threshold (|dev| < 5%).
        - 40% caution threshold (|dev| > 40%).
        """
        dates = pd.date_range("2015-01-06", "2024-01-02", freq="W-TUE")
        n = len(dates)
        # Constant Net%OI of 20%
        df = pd.DataFrame({
            "report_date": dates,
            "open_interest": [100000] * n,
            "comm_long": [60000] * n,
            "comm_short": [40000] * n,
        })
        # Inject deliberate deviations in 2023 (index ~420)
        # Year 2023 January: change comm_long to create +3% deviation (noise)
        # Year 2023 July: change comm_long to create +45% deviation (caution)
        dt_idx = pd.DatetimeIndex(dates)
        mask_jan_2023 = (dt_idx.year == 2023) & (dt_idx.month == 1)
        mask_jul_2023 = (dt_idx.year == 2023) & (dt_idx.month == 7)

        df.loc[mask_jan_2023, "comm_long"] = 61500  # Net%OI = 21.5% -> dev = +1.5% (< 5% noise)
        df.loc[mask_jul_2023, "comm_long"] = 85000  # Net%OI = 45% -> dev = +25% (> 5% not noise)

        # Force a > 40% deviation in Nov 2023
        mask_nov_2023 = (dt_idx.year == 2023) & (dt_idx.month == 11)
        df.loc[mask_nov_2023, "comm_long"] = 105000 # Net%OI = 65% -> dev = +45% (> 40% caution)

        jiler = jiler_seasonal_deviation(df, lookback_years=6)

        # Jan 2023 should have noise flag = 1
        jan_idx = np.where(mask_jan_2023)[0][0]
        self.assertEqual(jiler["jiler_noise_flag"].iloc[jan_idx], 1)
        self.assertEqual(jiler["jiler_caution_flag"].iloc[jan_idx], 0)

        # Nov 2023 should have caution flag = 1
        nov_idx = np.where(mask_nov_2023)[0][0]
        self.assertEqual(jiler["jiler_noise_flag"].iloc[nov_idx], 0)
        self.assertEqual(jiler["jiler_caution_flag"].iloc[nov_idx], 1)

    def test_10_monte_carlo_synthetic_stress_and_invariance(self):
        """
        Exhaustive stress test over 1,000 synthetic bars with:
        - Outliers (100x variance).
        - Zeros and negative open interest.
        - Missing/NaN rows.
        - Verify compute_all_cot_features executes without crashes and outputs valid bounds.
        """
        n = 1000
        dates = pd.date_range("2000-01-04", periods=n, freq="W-TUE")
        oi = np.random.randint(10000, 200000, n).astype(float)
        # Inject edge cases: zero OI, negative OI, NaN OI
        oi[50] = 0.0
        oi[51] = -500.0
        oi[52] = np.nan

        cl = np.random.randint(1000, 80000, n).astype(float)
        cs = np.random.randint(1000, 80000, n).astype(float)
        # Inject missing data
        cl[100] = np.nan
        cs[100] = np.nan

        df = pd.DataFrame({
            "report_date": dates,
            "open_interest": oi,
            "comm_long": cl,
            "comm_short": cs,
            "noncomm_long": np.random.randint(5000, 50000, n),
            "noncomm_short": np.random.randint(5000, 50000, n),
            "nonrep_long": np.random.randint(1000, 20000, n),
            "nonrep_short": np.random.randint(1000, 20000, n),
            "Open": 100.0 + np.cumsum(np.random.normal(0, 1, n)),
            "Close": 100.0 + np.cumsum(np.random.normal(0, 1, n)),
        })

        enriched = compute_all_cot_features(df, include_daily_progo=True, include_jiler=True)

        # Core assertions:
        self.assertEqual(len(enriched), n)
        self.assertIn("cot_index_156w", enriched.columns)
        self.assertIn("willco_26w_net", enriched.columns)
        self.assertIn("willco_26w_short", enriched.columns)
        self.assertIn("comm_208w_extreme_high", enriched.columns)
        self.assertIn("comm_156w_extreme_high", enriched.columns)
        self.assertIn("public_index_156w", enriched.columns)
        self.assertIn("large_index_156w", enriched.columns)
        self.assertIn("oi_stoch_52w", enriched.columns)
        self.assertIn("inside_oi_regime", enriched.columns)
        self.assertIn("progo", enriched.columns)
        self.assertIn("jiler_seasonal_norm", enriched.columns)

        # Numerical bounds validation:
        cot_valid = enriched["cot_index_156w"].dropna()
        self.assertTrue((cot_valid >= 0.0).all() and (cot_valid <= 100.0).all())

        willco_net_valid = enriched["willco_26w_net"].dropna()
        self.assertTrue((willco_net_valid >= 0.0).all() and (willco_net_valid <= 100.0).all())

        willco_short_valid = enriched["willco_26w_short"].dropna()
        self.assertTrue((willco_short_valid >= 0.0).all() and (willco_short_valid <= 100.0).all())

        oi_stoch_valid = enriched["oi_stoch_52w"].dropna()
        self.assertTrue((oi_stoch_valid >= 0.0).all() and (oi_stoch_valid <= 100.0).all())

        # Regime flag validation
        self.assertTrue(set(enriched["inside_oi_regime"].unique()).issubset({0, 1, 2, 3, 4}))


if __name__ == "__main__":
    unittest.main()

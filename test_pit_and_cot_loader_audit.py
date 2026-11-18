"""
test_pit_and_cot_loader_audit.py — Comprehensive Audit Test Suite for CFTC ETL & Point-in-Time Accuracy
Audited against DOCS/COT_Master_Specification.md.

Tests:
  1. Real CFTC 2024 Data Ingestion from cftc.gov (deacot2024.zip).
  2. CFTC Accounting Balance Identities across all real contracts:
     - rep_long == comm_long + noncomm_long + noncomm_spread
     - rep_short == comm_short + noncomm_short + noncomm_spread
     - nonrep_long == open_interest - rep_long
     - nonrep_short == open_interest - rep_short
     - total longs == total shorts == open_interest
     - Net_comm + Net_noncomm + Net_nonrep == 0.
  3. Synthetic COT Accounting Identity Verification:
     - Guarantees exact 0 error on synthetic generation.
  4. Point-In-Time (PIT) Merger Audit (merge_cot_with_price):
     - Tuesday snapshot -> Friday 15:30 ET publication release.
     - Verify NO price bar before Friday 15:30 ET receives the new Tuesday COT data.
     - Verify Friday post-15:30 ET bars DO receive the newly released COT data.
     - Verify Monday open bars (09:30 ET) receive the newly released Friday COT data.
  5. Timezone Awareness and Leakage Audit:
     - Test UTC price series (e.g. 19:00 UTC = 14:00 EST vs 20:30 UTC = 15:30 EST).
     - Confirms 1.5-hr to 5-hr timezone lookahead bug is resolved.
     - Test timezone-naive price series.
  6. Holiday Weekend & Non-Tuesday Snapshot PIT Handling:
     - Thanksgiving weekend (release slides to Monday 15:30 ET).
     - Good Friday (release slides to Monday 15:30 ET).
     - Monday holiday snapshot (e.g. 2023-07-03 Monday snapshot -> Friday release, not Thursday).
  7. MARKET_MAP Coverage & Substring Matching Accuracy:
     - All 22 canonical market symbols across Commodities, Financials, Currencies, Equities.
     - Substring collision tests (Ultra Bond vs 30Y Bond, Euro FX vs EUR Cross Rates).
     - Modern vs Legacy contract name coverage (WTI-PHYSICAL, UST BOND, NAT GAS NYME).
"""

import unittest
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

from cot_engine import (
    MARKET_MAP,
    download_cftc_year,
    load_cftc_history,
    filter_market_cot,
    merge_cot_with_price,
    generate_synthetic_cot,
    validate_cftc_identities,
    compute_cftc_release_time,
)


class TestPITAandCOTLoaderAudit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load or download real CFTC 2024 dataset once for tests
        cls.cftc_2024 = download_cftc_year(2024, report_type="futures_only")

    def test_01_real_cftc_2024_data_ingestion(self):
        """
        Verify downloading, parsing, and caching real CFTC 2024 annual zip data (deacot2024.zip).
        """
        self.assertIsNotNone(self.cftc_2024)
        self.assertGreater(len(self.cftc_2024), 10000)
        self.assertIn("market_name", self.cftc_2024.columns)
        self.assertIn("report_date", self.cftc_2024.columns)
        self.assertIn("available_at", self.cftc_2024.columns)
        self.assertIn("open_interest", self.cftc_2024.columns)
        self.assertIn("comm_long", self.cftc_2024.columns)
        self.assertIn("comm_short", self.cftc_2024.columns)
        self.assertIn("cftc_code", self.cftc_2024.columns)

        # Verify date range covers full 2024 year
        report_dates = self.cftc_2024["report_date"].dropna()
        self.assertEqual(report_dates.min().year, 2024)
        self.assertEqual(report_dates.max().year, 2024)
        self.assertEqual(len(report_dates.unique()), 53)

    def test_02_cftc_accounting_identities_real_data(self):
        """
        Verify CFTC accounting balance identities across all real 2024 contracts:
          - rep_long = comm_long + noncomm_long + noncomm_spread
          - nonrep_long = OI - rep_long
          - total longs == total shorts == OI
          - Net_comm + Net_noncomm + Net_nonrep == 0.
        Per §1.3: identity violations must be <= 1 contract (due to conversion factor rounding).
        """
        df = self.cftc_2024

        # 1. Reportable positions
        rep_long_diff = (df["rep_long"] - (df["comm_long"] + df["noncomm_long"] + df["noncomm_spread"])).abs()
        self.assertLessEqual(rep_long_diff.max(), 1.0)

        rep_short_diff = (df["rep_short"] - (df["comm_short"] + df["noncomm_short"] + df["noncomm_spread"])).abs()
        self.assertLessEqual(rep_short_diff.max(), 1.0)

        # 2. Non-reportable positions
        nonrep_long_diff = (df["nonrep_long"] - (df["open_interest"] - df["rep_long"])).abs()
        self.assertLessEqual(nonrep_long_diff.max(), 1.0)

        nonrep_short_diff = (df["nonrep_short"] - (df["open_interest"] - df["rep_short"])).abs()
        self.assertLessEqual(nonrep_short_diff.max(), 1.0)

        # 3. Total longs == total shorts == open_interest
        total_longs = df["rep_long"] + df["nonrep_long"]
        total_shorts = df["rep_short"] + df["nonrep_short"]
        self.assertLessEqual((total_longs - df["open_interest"]).abs().max(), 1.0)
        self.assertLessEqual((total_shorts - df["open_interest"]).abs().max(), 1.0)
        self.assertLessEqual((total_longs - total_shorts).abs().max(), 1.0)

        # 4. Net sum across groups == 0 (spreading is symmetric)
        net_comm = df["comm_long"] - df["comm_short"]
        net_noncomm = df["noncomm_long"] - df["noncomm_short"]
        net_nonrep = df["nonrep_long"] - df["nonrep_short"]
        net_sum = (net_comm + net_noncomm + net_nonrep).abs()
        self.assertLessEqual(net_sum.max(), 2.0)

        # Ensure validate_cftc_identities accepts 100% of rows
        validated = validate_cftc_identities(df, max_tolerance=1.0)
        self.assertEqual(len(validated), len(df))

    def test_03_synthetic_cot_exact_accounting_identities(self):
        """
        Verify that synthetic COT generator satisfies CFTC accounting identities with EXACT 0.0 error.
        """
        syn = generate_synthetic_cot(start_date="2020-01-07", end_date="2024-12-31")

        # Identity 1: Reportable sum
        self.assertTrue((syn["rep_long"] == syn["comm_long"] + syn["noncomm_long"] + syn["noncomm_spread"]).all())
        self.assertTrue((syn["rep_short"] == syn["comm_short"] + syn["noncomm_short"] + syn["noncomm_spread"]).all())

        # Identity 2: Non-reportable complement of OI
        self.assertTrue((syn["nonrep_long"] == syn["open_interest"] - syn["rep_long"]).all())
        self.assertTrue((syn["nonrep_short"] == syn["open_interest"] - syn["rep_short"]).all())

        # Identity 3: Total longs == Total shorts == Open Interest
        total_longs = syn["rep_long"] + syn["nonrep_long"]
        total_shorts = syn["rep_short"] + syn["nonrep_short"]
        self.assertTrue((total_longs == syn["open_interest"]).all())
        self.assertTrue((total_shorts == syn["open_interest"]).all())

        # Identity 4: Net positions sum to zero
        net_comm = syn["comm_long"] - syn["comm_short"]
        net_noncomm = syn["noncomm_long"] - syn["noncomm_short"]
        net_nonrep = syn["nonrep_long"] - syn["nonrep_short"]
        self.assertTrue(((net_comm + net_noncomm + net_nonrep) == 0).all())

        # Ensure validate_cftc_identities passes with 0 tolerance
        validated_syn = validate_cftc_identities(syn, max_tolerance=0.0)
        self.assertEqual(len(validated_syn), len(syn))

    def test_04_pit_merger_intraday_and_monday_open(self):
        """
        Verify PIT merger strictly prevents lookahead:
          - Tuesday snapshot (2024-01-09) -> Released Friday (2024-01-12 15:30 ET).
          - NO price bar before Friday 15:30 ET sees the new Tuesday data.
          - Friday post-15:30 ET bars DO see the new Tuesday data.
          - Monday open bar (2024-01-15 09:30 ET) correctly sees Friday's newly released data.
        """
        cot = pd.DataFrame({
            "report_date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-09")],
            "available_at": [
                pd.Timestamp("2024-01-05 15:30:00"),  # Prior week release
                pd.Timestamp("2024-01-12 15:30:00"),  # Current week release
            ],
            "comm_long": [1000, 5000],
            "comm_short": [2000, 1000],
            "open_interest": [10000, 15000],
        })

        bars = pd.DataFrame({
            "Close": [2000.0, 2005.0, 2010.0, 2015.0, 2020.0],
        }, index=pd.to_datetime([
            "2024-01-12 09:30:00",  # Friday morning regular open -> MUST see Prior (1000)
            "2024-01-12 15:29:59",  # 1 second before release -> MUST see Prior (1000)
            "2024-01-12 15:30:00",  # Exact release timestamp -> Sees Current (5000)
            "2024-01-12 16:00:00",  # Friday settlement close -> Sees Current (5000)
            "2024-01-15 09:30:00",  # Monday regular open -> Sees Current (5000)
        ]))

        merged = merge_cot_with_price(bars, cot)

        self.assertEqual(merged.loc["2024-01-12 09:30:00", "comm_long"], 1000)
        self.assertEqual(merged.loc["2024-01-12 15:29:59", "comm_long"], 1000)
        self.assertEqual(merged.loc["2024-01-12 15:30:00", "comm_long"], 5000)
        self.assertEqual(merged.loc["2024-01-12 16:00:00", "comm_long"], 5000)
        self.assertEqual(merged.loc["2024-01-15 09:30:00", "comm_long"], 5000)

    def test_05_pit_merger_timezone_leakage_prevention(self):
        """
        Verify timezone-aware price series (e.g. UTC) do NOT leak data:
          - Release is at 15:30 US/Eastern.
          - In winter (EST = UTC-5), 15:30 EST == 20:30 UTC.
          - Price bar at 19:00 UTC (14:00 EST) MUST NOT see the 15:30 EST release!
          - Price bar at 20:30 UTC (15:30 EST) and 21:00 UTC (16:00 EST) MUST see it!
        """
        cot = pd.DataFrame({
            "report_date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-09")],
            "available_at": [
                pd.Timestamp("2024-01-05 15:30:00"),  # Prior
                pd.Timestamp("2024-01-12 15:30:00"),  # Current
            ],
            "comm_long": [1000, 5000],
        })

        # Price series in UTC
        bars_utc = pd.DataFrame({
            "Close": [2000.0, 2010.0, 2020.0, 2030.0],
        }, index=pd.to_datetime([
            "2024-01-12 19:00:00",  # 14:00 EST (before release) -> MUST see Prior (1000)
            "2024-01-12 20:29:00",  # 15:29 EST (before release) -> MUST see Prior (1000)
            "2024-01-12 20:30:00",  # 15:30 EST (release time)    -> Sees Current (5000)
            "2024-01-12 21:00:00",  # 16:00 EST (after release)   -> Sees Current (5000)
        ]).tz_localize("UTC"))

        merged = merge_cot_with_price(bars_utc, cot)

        self.assertEqual(merged.iloc[0]["comm_long"], 1000)
        self.assertEqual(merged.iloc[1]["comm_long"], 1000)
        self.assertEqual(merged.iloc[2]["comm_long"], 5000)
        self.assertEqual(merged.iloc[3]["comm_long"], 5000)

    def test_06_holiday_release_pit_accuracy(self):
        """
        Verify holiday edge cases:
          1. Thanksgiving 2024: report Tuesday 2024-11-26 -> release slides past Friday to Monday 2024-12-02 15:30 ET.
             - Friday 2024-11-29 bars MUST NOT see the new report.
             - Monday 2024-12-02 09:30 ET open MUST NOT see the new report.
             - Monday 2024-12-02 16:00 ET settlement and Tuesday 2024-12-03 open DO see it.
          2. Good Friday 2024 (2024-03-29): report Tuesday 2024-03-26 -> release slides to Monday 2024-04-01 15:30 ET.
          3. Monday snapshot (2023-07-03): report on Monday due to July 4 holiday -> release on Friday 2023-07-07, NOT Thursday.
        """
        # 1. Thanksgiving release computation
        tg_release = compute_cftc_release_time(date(2024, 11, 26))
        self.assertEqual(tg_release, datetime(2024, 12, 2, 15, 30))

        # 2. Good Friday release computation
        gf_release = compute_cftc_release_time(date(2024, 3, 26))
        self.assertEqual(gf_release, datetime(2024, 4, 1, 15, 30))

        # 3. Monday snapshot (July 4th week 2023)
        j4_release = compute_cftc_release_time(date(2023, 7, 3))
        self.assertEqual(j4_release, datetime(2023, 7, 7, 15, 30))

        # Test PIT merger during Thanksgiving holiday week
        cot_tg = pd.DataFrame({
            "report_date": [pd.Timestamp("2024-11-19"), pd.Timestamp("2024-11-26")],
            "available_at": [
                compute_cftc_release_time(date(2024, 11, 19)),  # Friday Nov 22 15:30
                compute_cftc_release_time(date(2024, 11, 26)),  # Monday Dec 02 15:30 (holiday delayed!)
            ],
            "comm_long": [1000, 9000],
        })

        bars_tg = pd.DataFrame({
            "Close": [100.0, 101.0, 102.0, 103.0],
        }, index=pd.to_datetime([
            "2024-11-29 16:00:00",  # Friday after Thanksgiving -> MUST see Prior (1000)
            "2024-12-02 09:30:00",  # Monday morning open       -> MUST see Prior (1000)
            "2024-12-02 16:00:00",  # Monday evening close      -> Sees Current (9000)
            "2024-12-03 09:30:00",  # Tuesday morning open      -> Sees Current (9000)
        ]))

        merged_tg = merge_cot_with_price(bars_tg, cot_tg)
        self.assertEqual(merged_tg.loc["2024-11-29 16:00:00", "comm_long"], 1000)
        self.assertEqual(merged_tg.loc["2024-12-02 09:30:00", "comm_long"], 1000)
        self.assertEqual(merged_tg.loc["2024-12-02 16:00:00", "comm_long"], 9000)
        self.assertEqual(merged_tg.loc["2024-12-03 09:30:00", "comm_long"], 9000)

    def test_07_market_map_coverage_across_all_22_symbols(self):
        """
        Verify that filter_market_cot successfully resolves ALL 22 canonical symbols
        in real CFTC 2024 data across commodities, financials, currencies, and equity indexes.
        """
        symbols = [
            "GC=F", "SI=F", "HG=F", "CL=F", "NG=F",
            "ZC=F", "ZW=F", "ZS=F", "KC=F",
            "ZB=F", "ZN=F", "ZF=F", "ZT=F",
            "ES=F", "NQ=F", "YM=F",
            "6J=F", "6B=F", "6E=F", "6A=F", "6C=F", "6S=F",
        ]

        for sym in symbols:
            filtered = filter_market_cot(self.cftc_2024, sym)
            self.assertEqual(
                len(filtered),
                53,
                f"Symbol {sym} failed to match exactly 53 weekly reports in 2024 data (matched {len(filtered)})",
            )
            # Verify primary market has substantial open interest
            self.assertGreater(filtered["open_interest"].mean(), 10000, f"Symbol {sym} has suspiciously low OI")

    def test_08_market_map_substring_collision_prevention(self):
        """
        Verify that substring matching does not misidentify distinct contracts:
          - ZB=F (30Y Bond) must NOT select ULTRA UST BOND (UB=F).
          - 6E=F (Euro FX) must NOT select EURO FX Cross Rates (EUR/GBP or EUR/JPY).
          - 6J=F (Japanese Yen) must NOT select EURO FX/JAPANESE YEN XRATE.
          - CL=F must select primary benchmark WTI-PHYSICAL, not regional differentials or swap spreads.
          - NG=F must select primary benchmark NAT GAS NYME, not E-Mini or index spreads.
        """
        # ZB=F Treasury Bond
        zb = filter_market_cot(self.cftc_2024, "ZB=F")
        self.assertTrue("UST BOND - CHICAGO BOARD OF TRADE" in zb["market_name"].iloc[0])
        self.assertFalse("ULTRA" in zb["market_name"].iloc[0])

        # 6E=F Euro FX
        euro = filter_market_cot(self.cftc_2024, "6E=F")
        self.assertEqual(euro["market_name"].iloc[0], "EURO FX - CHICAGO MERCANTILE EXCHANGE")

        # 6J=F Japanese Yen
        yen = filter_market_cot(self.cftc_2024, "6J=F")
        self.assertEqual(yen["market_name"].iloc[0], "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE")

        # CL=F Crude Oil
        crude = filter_market_cot(self.cftc_2024, "CL=F")
        self.assertEqual(crude["market_name"].iloc[0], "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE")

        # NG=F Natural Gas
        ng = filter_market_cot(self.cftc_2024, "NG=F")
        self.assertEqual(ng["market_name"].iloc[0], "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE")


if __name__ == "__main__":
    unittest.main()

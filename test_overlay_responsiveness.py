"""
Test Suite: Sidebar Overlay Responsiveness and Decoupled View Mode Logic
Validates that sidebar checkbox toggles directly and reliably update the ChartSpec
without being blocked by view modes or forced on when unchecked.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data_engine import fetch_data
from features import add_features
from ml.timesfm_engine import get_timesfm_engine
from tv_chart import (
    ChartSpec,
    add_harmonic_swing_wave,
    add_turning_points_to_chart,
    add_multi_scenario_forecast,
    add_forward_rv_envelope,
    add_volume_profile_to_chart,
    set_macro_catalyst_badge,
    series_points,
    SOLID as LS_SOLID, DOTTED as LS_DOTTED, DASHED as LS_DASHED
)
from config.market_catalog import get_price_decimals, format_price

def build_test_spec(
    df,
    df_feat,
    timesfm_res,
    symbol="NG=F",
    interval="5m",
    view_mode="🌊 Swing Wave & Cycles",
    show_future_forecast=True,
    show_swing_wave=True,
    show_multi_scenario=True,
    show_turning_points=True,
    show_rv_tunnel=False,
    show_poc_line=True,
    show_deal_markers=False,
    show_sma20=True,
    show_sma50=True,
    show_sma200=False,
    show_bb=True,
):
    """
    Mirrors the exact decoupled chart construction logic in app.py render_dashboard().
    """
    last_close = float(df["Close"].iloc[-1])
    spec = ChartSpec()
    spec.set_candles(df.tail(60))
    spec.auto_configure_precision(symbol, last_close)

    # Technical Overlays
    def _overlay(col, color, width, title, style=LS_SOLID):
        if col in df_feat.columns:
            spec.line(series_points(df.tail(60).index, df_feat.tail(60)[col]), color, width, style, title, legend=True)

    if show_sma20:
        _overlay("SMA_20", "#FFD54F", 1.2, "SMA 20")
    if show_sma50:
        _overlay("SMA_50", "#FF9800", 1.2, "SMA 50")
    if show_sma200:
        _overlay("SMA_200", "#7E57C2", 1.5, "SMA 200")
    if show_bb and "BB_High" in df_feat.columns and "BB_Low" in df_feat.columns:
        _overlay("BB_High", "rgba(148, 163, 184, 0.55)", 1, "BB Upper", LS_DASHED)
        _overlay("BB_Low", "rgba(148, 163, 184, 0.55)", 1, "BB Lower", LS_DASHED)

    # Institutional Volume Profile
    if show_poc_line:
        add_volume_profile_to_chart(spec, df.tail(60), n_bins=24)

    is_wave_mode = view_mode in ("🌊 Swing Wave & Cycles", "⚡ All Overlays")
    is_corridor_mode = view_mode in ("🎯 Execution Corridors", "⚡ All Overlays")
    is_cloud_mode = view_mode in ("🔮 Probabilistic Quantile Cloud", "⚡ All Overlays")

    n_proj = 15
    future_dates = [df.index[-1] + pd.Timedelta(minutes=5 * (i + 1)) for i in range(n_proj)]
    forecast_span = [df.index[-1]] + future_dates

    # 1. Harmonic Swing Wave
    if (show_swing_wave or is_wave_mode) and timesfm_res:
        wave_raw = timesfm_res.get("swing_wave") or timesfm_res.get("forecast")
        if wave_raw:
            wave_vals = [last_close] + list(wave_raw)
            add_harmonic_swing_wave(spec, forecast_span, wave_vals, title="Harmonic Swing Wave")

    # 2. Breakout Corridor
    if is_corridor_mode or (view_mode == "⚡ All Overlays"):
        spec.add_breakout_corridor(forecast_span, entry=last_close, take_profit=last_close * 1.03, stop_loss=last_close * 0.98)

    # 3. Multi-Horizon Forecast Quantiles
    if (show_future_forecast or is_cloud_mode) and timesfm_res and timesfm_res.get("q50"):
        q50_path = [last_close] + list(timesfm_res["q50"])
        q90_path = [last_close] + list(timesfm_res["q90"]) if timesfm_res.get("q90") else None
        q10_path = [last_close] + list(timesfm_res["q10"]) if timesfm_res.get("q10") else None
        spec.add_forecast_quantiles(forecast_span, q50=q50_path, q10=q10_path, q90=q90_path)

    # 4. Projected Turning Points
    if (show_turning_points or view_mode == "⚡ All Overlays") and timesfm_res:
        tp_dict_c = timesfm_res.get("turning_points") or {}
        tp_points = []
        if tp_dict_c.get("peak"):
            pk = tp_dict_c["peak"]
            tp_points.append(dict(time=future_dates[0], price=float(pk.get("price", last_close)), kind="peak"))
        if tp_dict_c.get("trough"):
            tr = tp_dict_c["trough"]
            tp_points.append(dict(time=future_dates[-1], price=float(tr.get("price", last_close)), kind="trough"))
        if tp_points:
            add_turning_points_to_chart(spec, tp_points)

    # 5. Multi-Scenario Paths
    if (show_multi_scenario or view_mode == "⚡ All Overlays") and timesfm_res:
        scenarios_c = timesfm_res.get("scenarios") or {}
        scenario_trajectories = {}
        if scenarios_c.get("bullish_breakout", {}).get("trajectory"):
            scenario_trajectories["bullish"] = [last_close] + list(scenarios_c["bullish_breakout"]["trajectory"])
        if scenarios_c.get("base", {}).get("trajectory"):
            scenario_trajectories["base"] = [last_close] + list(scenarios_c["base"]["trajectory"])
        if scenario_trajectories:
            add_multi_scenario_forecast(spec, forecast_span, scenario_trajectories)

    # 6. Forward RV Tunnel
    if (show_rv_tunnel or is_cloud_mode or view_mode == "⚡ All Overlays") and timesfm_res:
        rv_data = timesfm_res.get("rv_envelope") or {}
        if rv_data.get("upper_1s") and rv_data.get("lower_1s"):
            spec.add_volatility_envelope(
                times=forecast_span,
                upper_2s=[last_close] + list(rv_data.get("upper_2s", [])),
                upper_1s=[last_close] + list(rv_data.get("upper_1s", [])),
                lower_1s=[last_close] + list(rv_data.get("lower_1s", [])),
                lower_2s=[last_close] + list(rv_data.get("lower_2s", [])),
                center=[last_close] + list(timesfm_res.get("q50", [])),
                forward_rv=timesfm_res.get("forward_rv_mean"),
                name="Forward RV Tunnel",
            )
        else:
            add_forward_rv_envelope(
                spec=spec,
                times=forecast_span,
                center=[last_close] + list(timesfm_res.get("q50", [])),
                forward_rv=timesfm_res.get("forward_rv_mean", 0.015),
                name="Forward RV Tunnel",
            )

    return spec

class TestOverlayResponsiveness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.symbol = "NG=F"
        cls.interval = "5m"
        raw_df = fetch_data(cls.symbol, period="5d", interval=cls.interval)
        assert raw_df is not None and len(raw_df) > 50
        cls.df = raw_df
        cls.df_feat = add_features(raw_df)
        eng = get_timesfm_engine()
        cls.timesfm_res = eng.forecast(cls.df_feat, cls.symbol, cls.interval, 15).to_dict()

    def test_all_overlays_disabled(self):
        """When all toggles are OFF and view_mode is Custom/Bare, spec should have zero overlay structures."""
        spec = build_test_spec(
            self.df, self.df_feat, self.timesfm_res,
            view_mode="Custom",
            show_future_forecast=False,
            show_swing_wave=False,
            show_multi_scenario=False,
            show_turning_points=False,
            show_rv_tunnel=False,
            show_poc_line=False,
            show_sma20=False,
            show_sma50=False,
            show_sma200=False,
            show_bb=False,
        )
        self.assertEqual(len(spec.forecast_bands), 0)
        self.assertEqual(len(spec.swing_waves), 0)
        self.assertEqual(len(spec.breakout_corridors), 0)
        self.assertEqual(len(spec.turning_points), 0)
        self.assertEqual(len(spec.forecast_scenarios), 0)
        self.assertEqual(len(spec.volatility_envelopes), 0)
        self.assertIsNone(spec.volume_profile)
        self.assertEqual(len(spec.lines), 0)
        print("  [PASS] All overlays disabled -> ChartSpec is completely clean.")

    def test_forecast_cone_toggle_responsiveness(self):
        """Testing forecast cone toggle independently."""
        # 1. Enabled
        spec_on = build_test_spec(
            self.df, self.df_feat, self.timesfm_res,
            view_mode="Custom",
            show_future_forecast=True,
            show_swing_wave=False,
        )
        self.assertGreaterEqual(len(spec_on.forecast_bands), 1, "Forecast cone was not populated when show_future_forecast=True")
        
        # 2. Disabled
        spec_off = build_test_spec(
            self.df, self.df_feat, self.timesfm_res,
            view_mode="Custom",
            show_future_forecast=False,
            show_swing_wave=False,
        )
        self.assertEqual(len(spec_off.forecast_bands), 0, "Forecast cone remained when show_future_forecast=False")
        print("  [PASS] Forecast cone (q10-q90) toggles on/off cleanly.")

    def test_swing_wave_toggle_responsiveness(self):
        """Testing dedicated swing wave toggle."""
        # 1. Enabled
        spec_on = build_test_spec(
            self.df, self.df_feat, self.timesfm_res,
            view_mode="Custom",
            show_swing_wave=True,
        )
        self.assertGreaterEqual(len(spec_on.swing_waves), 1, "Swing wave missing when show_swing_wave=True")

        # 2. Disabled
        spec_off = build_test_spec(
            self.df, self.df_feat, self.timesfm_res,
            view_mode="Custom",
            show_swing_wave=False,
        )
        self.assertEqual(len(spec_off.swing_waves), 0, "Swing wave remained when show_swing_wave=False")
        print("  [PASS] Harmonic swing wave toggles on/off cleanly.")

    def test_turning_points_and_scenarios_toggle_responsiveness(self):
        """Verify turning points and scenarios can be toggled OFF independently even in default mode."""
        # 1. Turning points OFF
        spec_tp_off = build_test_spec(
            self.df, self.df_feat, self.timesfm_res,
            view_mode="Custom",
            show_turning_points=False,
            show_multi_scenario=True,
        )
        self.assertEqual(len(spec_tp_off.turning_points), 0)
        self.assertGreaterEqual(len(spec_tp_off.forecast_scenarios), 1)

        # 2. Scenarios OFF
        spec_sc_off = build_test_spec(
            self.df, self.df_feat, self.timesfm_res,
            view_mode="Custom",
            show_turning_points=True,
            show_multi_scenario=False,
        )
        self.assertGreaterEqual(len(spec_sc_off.turning_points), 1)
        self.assertEqual(len(spec_sc_off.forecast_scenarios), 0)
        print("  [PASS] Turning points & Multi-scenario paths toggle independently without cross-leakage.")

    def test_rv_tunnel_toggle_responsiveness(self):
        """Verify RV dispersion tunnel toggle."""
        spec_rv_on = build_test_spec(
            self.df, self.df_feat, self.timesfm_res,
            view_mode="Custom",
            show_rv_tunnel=True,
        )
        self.assertGreaterEqual(len(spec_rv_on.volatility_envelopes), 1)

        spec_rv_off = build_test_spec(
            self.df, self.df_feat, self.timesfm_res,
            view_mode="Custom",
            show_rv_tunnel=False,
        )
        self.assertEqual(len(spec_rv_off.volatility_envelopes), 0)
        print("  [PASS] Forward RV tunnel toggles on/off cleanly.")

    def test_technical_lines_toggle_responsiveness(self):
        """Verify SMA-20, SMA-50, SMA-200, and Bollinger toggles."""
        # SMA 20 ON only
        spec_sma20 = build_test_spec(
            self.df, self.df_feat, self.timesfm_res,
            view_mode="Custom",
            show_sma20=True, show_sma50=False, show_sma200=False, show_bb=False,
        )
        titles = [l.get("title") for l in spec_sma20.lines]
        self.assertIn("SMA 20", titles)
        self.assertNotIn("SMA 50", titles)
        self.assertNotIn("SMA 200", titles)
        self.assertNotIn("BB Upper", titles)

        # Bollinger ON only
        spec_bb = build_test_spec(
            self.df, self.df_feat, self.timesfm_res,
            view_mode="Custom",
            show_sma20=False, show_sma50=False, show_sma200=False, show_bb=True,
        )
        titles_bb = [l.get("title") for l in spec_bb.lines]
        self.assertNotIn("SMA 20", titles_bb)
        self.assertIn("BB Upper", titles_bb)
        self.assertIn("BB Lower", titles_bb)
        print("  [PASS] Technical indicators (SMAs, Bollinger) toggle with 100% precision.")

    def test_volume_profile_poc_toggle_responsiveness(self):
        """Verify Institutional Volume Profile & POC Line toggle."""
        spec_poc_on = build_test_spec(
            self.df, self.df_feat, self.timesfm_res,
            view_mode="Custom",
            show_poc_line=True,
        )
        self.assertIsNotNone(spec_poc_on.volume_profile)

        spec_poc_off = build_test_spec(
            self.df, self.df_feat, self.timesfm_res,
            view_mode="Custom",
            show_poc_line=False,
        )
        self.assertIsNone(spec_poc_off.volume_profile)
        print("  [PASS] Volume profile & POC line toggle on/off cleanly.")

    def test_all_overlays_enabled(self):
        """When all overlays are enabled, all visual layers must be populated simultaneously."""
        spec_all = build_test_spec(
            self.df, self.df_feat, self.timesfm_res,
            view_mode="⚡ All Overlays",
            show_future_forecast=True,
            show_swing_wave=True,
            show_multi_scenario=True,
            show_turning_points=True,
            show_rv_tunnel=True,
            show_poc_line=True,
            show_sma20=True,
            show_sma50=True,
            show_sma200=True,
            show_bb=True,
        )
        self.assertGreaterEqual(len(spec_all.forecast_bands), 1)
        self.assertGreaterEqual(len(spec_all.swing_waves), 1)
        self.assertGreaterEqual(len(spec_all.breakout_corridors), 1)
        self.assertGreaterEqual(len(spec_all.turning_points), 1)
        self.assertGreaterEqual(len(spec_all.forecast_scenarios), 1)
        self.assertGreaterEqual(len(spec_all.volatility_envelopes), 1)
        self.assertIsNotNone(spec_all.volume_profile)
        self.assertGreaterEqual(len(spec_all.lines), 4)

        payload = spec_all.to_json()
        import json
        json_str = json.dumps(payload, allow_nan=False)
        self.assertGreater(len(json_str), 10000)
        print("  [PASS] All overlays enabled -> All structures populated and serialized cleanly with zero NaNs.")

if __name__ == "__main__":
    print("======================================================================")
    print("RUNNING OVERLAY RESPONSIVENESS & DECOUPLING TEST SUITE")
    print("======================================================================")
    unittest.main()

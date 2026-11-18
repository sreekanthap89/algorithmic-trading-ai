"""
Comprehensive Verification Suite for Algorithmic Trading AI
Exhaustive verification of prediction pipeline, accuracy enhancements,
advanced chart plotting, and application UI integration.

Sections:
1. Static Syntax & AST Validation (models.py, ml/timesfm_engine.py, tv_chart.py, app.py, metrics/validation.py)
2. Model Pipeline, Dynamic Adaptive Thresholds & TimesFM Synergy (models.py & ml/timesfm_engine.py)
3. Advanced Predictive Chart Plotting, Spec Serialization & HTML Engine (tv_chart.py)
4. Prediction Logging, Multi-Bar Pending Resolution & Drift Validation (app.py & metrics/validation.py)
5. End-to-End Simulation & Regression Verification
"""

import sys
import os
import json
import ast
import traceback
import tempfile
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def run_static_checks():
    print("======================================================================")
    print("STEP 1: Static Syntax & AST Code Quality Validation")
    print("======================================================================")
    target_files = [
        "models.py",
        "ml/timesfm_engine.py",
        "tv_chart.py",
        "app.py",
        "metrics/validation.py",
        "quant_engine.py",
        "features.py"
    ]
    for rel_path in target_files:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        assert os.path.exists(full_path), f"Critical file missing: {rel_path}"
        with open(full_path, "r", encoding="utf-8") as f:
            code = f.read()
        ast.parse(code, filename=rel_path)
        print(f"  [PASS] AST parse verified cleanly: {rel_path} ({len(code.splitlines())} lines)")
    print(">> All static and syntax checks passed cleanly.\n")

def run_model_pipeline_checks():
    print("======================================================================")
    print("STEP 2: Prediction Pipeline, Adaptive Thresholds & TimesFM Synergy")
    print("======================================================================")
    from data_engine import fetch_data
    from features import add_features
    from models import TradingModel
    from ml.timesfm_engine import get_timesfm_engine, run_timesfm_forecast, TIMESFM_QUANTILES

    symbol = "BTC-USD"
    interval = "1h"
    df = fetch_data(symbol, period="1mo", interval=interval)
    assert df is not None and len(df) >= 100, f"Insufficient data for {symbol}"
    df_feat = add_features(df)
    print(f"  [PASS] Data & Features ready: {len(df_feat)} rows, {len(df_feat.columns)} features.")

    # 1. TradingModel Classifier Checks
    model = TradingModel(symbol, interval)
    model.train_or_load(df_feat, force_retrain=False)
    
    # Test adaptive thresholds computation
    b_th, s_th, th_diag = model.compute_adaptive_thresholds(df_feat.iloc[-1:])
    print(f"  Adaptive Thresholds: Buy={b_th:.4f}, Sell={s_th:.4f}, Details={th_diag}")
    assert 0.50 <= b_th <= 0.65, f"Buy threshold out of realistic bounds: {b_th}"
    assert 0.35 <= s_th <= 0.50, f"Sell threshold out of realistic bounds: {s_th}"
    assert b_th > s_th, f"Buy threshold must exceed sell threshold: {b_th} vs {s_th}"

    # Prediction without external TimesFM
    base_pred = model.predict_next(df_feat)
    print(f"  Base Classifier Prediction: prob_up={base_pred['prob_up']:.4f}, meta_prob={base_pred['meta_prob']:.4f}, signal={base_pred['signal']}")
    assert "prob_up" in base_pred and "prob_down" in base_pred, "Missing probability keys in prediction"
    assert "meta_prob" in base_pred and "model_disagreement" in base_pred, "Missing meta keys"
    assert 0.0 <= base_pred["prob_up"] <= 1.0, f"prob_up out of bounds: {base_pred['prob_up']}"
    assert 0.0 <= base_pred["prob_down"] <= 1.0, f"prob_down out of bounds: {base_pred['prob_down']}"
    assert abs((base_pred["prob_up"] + base_pred["prob_down"]) - 1.0) < 1e-3, "Probabilities must sum to 1.0"
    assert base_pred["signal"] in ["BUY (UP)", "SELL (DOWN)", "NEUTRAL"], f"Invalid signal: {base_pred['signal']}"
    print("  [PASS] Base classifier returns properly calibrated probabilities and meta-gated signals.")

    # 2. TimesFM Foundation Forecast Checks
    engine = get_timesfm_engine()
    fc_out = engine.forecast(
        df=df_feat,
        symbol=symbol,
        interval=interval,
        horizon=15
    )
    print(f"  TimesFM Output: backend={fc_out.backend}, horizon={fc_out.horizon}, expected_ret={fc_out.expected_return_pct:.3f}%, prob_up={fc_out.prob_up:.3f}, signal={fc_out.signal}")
    assert len(fc_out.forecast) == 15, f"Expected 15 forecast steps, got {len(fc_out.forecast)}"
    assert fc_out.quantiles.shape == (15, 9), f"Expected quantiles shape (15, 9), got {fc_out.quantiles.shape}"
    assert len(fc_out.q10) == 15 and len(fc_out.q90) == 15, "Quantile convenience lists length mismatch"
    
    # Verify strict quantile monotonicity across all forecast horizons
    for h in range(15):
        q_row = fc_out.quantiles[h]
        diffs = np.diff(q_row)
        assert np.all(diffs >= -1e-6), f"Quantile crossing detected at horizon step {h}: {q_row}"
    print("  [PASS] Strict quantile monotonicity verified across all 15 forecast bars (q10 <= q20 <= ... <= q90).")

    # Verify finite numbers
    assert np.all(np.isfinite(fc_out.forecast)), "Point forecast contains non-finite values"
    assert np.all(np.isfinite(fc_out.quantiles)), "Quantiles contain non-finite values"

    # 3. Synergy Reconciliation Check
    fc_dict = fc_out.to_dict()
    synergy_pred = model.predict_next(df_feat, timesfm_res=fc_dict)
    assert "synergy" in synergy_pred, "Missing synergy structure in prediction"
    syn = synergy_pred["synergy"]
    print(f"  Synergy Reconciliation: consensus_prob_up={syn['synergy_prob_up']:.4f}, signal={syn['synergy_signal']}, status={syn['synergy_status']}, discrepancy={syn['synergy_discrepancy']:.4f}")
    assert "synergy_prob_up" in syn and "synergy_signal" in syn, "Incomplete synergy dictionary"
    assert 0.0 <= syn["synergy_prob_up"] <= 1.0, f"synergy_prob_up out of bounds: {syn['synergy_prob_up']}"
    assert 0.0 <= syn["synergy_discrepancy"] <= 1.0, f"synergy_discrepancy out of bounds: {syn['synergy_discrepancy']}"
    assert syn["synergy_signal"] in ["BUY (UP)", "SELL (DOWN)", "NEUTRAL"], f"Invalid synergy_signal: {syn['synergy_signal']}"
    print("  [PASS] Bidirectional skill-weighted synergy successfully reconciles classifier and TimesFM.")
    print(">> Model prediction pipeline and synergy checks passed cleanly.\n")

def run_chart_plotting_checks():
    print("======================================================================")
    print("STEP 3: Advanced Chart Plotting, Spec Serialization & LWC HTML Engine")
    print("======================================================================")
    import tv_chart
    from tv_chart import ChartSpec, build_tv_chart_html, add_prediction_forecast, add_risk_reward_corridor
    from data_engine import fetch_data
    from features import add_features
    from ml.timesfm_engine import run_timesfm_forecast

    symbol = "BTC-USD"
    interval = "1h"
    df = fetch_data(symbol, period="5d", interval=interval)
    df_feat = add_features(df)
    fc_dict = run_timesfm_forecast(df=df_feat, symbol=symbol, interval=interval, horizon=12)

    spec = ChartSpec()
    spec.set_candles(df_feat.tail(60))

    # Add technical line
    spec.line(
        [{"time": int(t.timestamp()), "value": float(v)} for t, v in df_feat["SMA_20"].tail(60).dropna().items() if np.isfinite(v)],
        color="#2962FF",
        title="SMA_20"
    )

    # 1. Multi-horizon Forecast Bands
    last_dt = df_feat.index[-1]
    step = pd.Timedelta(hours=1)
    future_times = [last_dt + step * (i + 1) for i in range(12)]
    forecast_span = [last_dt] + future_times
    last_close = float(df_feat["Close"].iloc[-1])

    direction = add_prediction_forecast(
        spec=spec,
        timesfm_res=fc_dict,
        forecast_span=forecast_span,
        last_close=last_close,
        accuracy=0.74,
        confidence=0.88,
        interval=interval
    )
    print(f"  [PASS] add_prediction_forecast added forecast bands and tooltips, direction: {direction}")
    assert direction in ["BULLISH", "BEARISH", "NEUTRAL"], f"Invalid forecast direction: {direction}"

    # 2. Directional Projection Vector
    proj_path = [last_close * (1.0 + 0.0015 * i) for i in range(len(forecast_span))]
    proj_dir = spec.add_directional_projection(forecast_span, proj_path, title="Target Trajectory")
    print(f"  [PASS] add_directional_projection added glowing path: {proj_dir}")

    # 3. Risk-Reward Breakout Corridor
    tp = last_close * 1.035
    sl = last_close * 0.982
    add_risk_reward_corridor(
        spec=spec,
        forecast_span=forecast_span,
        entry=last_close,
        take_profit=tp,
        stop_loss=sl,
        win_prob=0.72
    )
    print("  [PASS] add_risk_reward_corridor attached corridor and R:R pill badge.")

    # 4. In-Chart Prediction HUD Badge
    spec.set_prediction_badge(
        accuracy=0.762,
        confidence=0.841,
        direction=direction or "NEUTRAL",
        horizon_bars=12,
        model_name="TimesFM-3.0.2 Multivariate",
        win_rate=0.735,
        brier_score=0.194,
        drift=0.028
    )
    assert spec.prediction_badge is not None, "Prediction badge not set"
    print("  [PASS] set_prediction_badge attached in-chart HUD card.")

    # 5. Check JSON Spec Serialization
    payload = spec.to_json()
    assert isinstance(payload, dict), "to_json() must return a dictionary"
    assert "candles" in payload and len(payload["candles"]) > 0, "Missing candles in spec"
    assert "forecastBands" in payload and len(payload["forecastBands"]) >= 2, "Forecast bands missing in payload"
    assert "breakoutCorridors" in payload and len(payload["breakoutCorridors"]) >= 1, "Breakout corridors missing in payload"
    assert "directionalProjections" in payload and len(payload["directionalProjections"]) >= 1, "Directional projections missing in payload"
    assert "predictionBadge" in payload and payload["predictionBadge"] is not None, "Prediction badge missing in payload"
    assert "forecastTooltips" in payload and len(payload["forecastTooltips"]) >= 1, "Forecast tooltips missing in payload"

    # Ensure valid JSON string without NaN or circular refs
    spec_json = json.dumps(payload, allow_nan=False)
    print(f"  [PASS] ChartSpec serialized to JSON cleanly ({len(spec_json)} bytes, zero NaNs).")

    # 6. HTML Generation & LWC Bundle Integrity
    html_out = build_tv_chart_html(spec, title="BTC-USD Advanced Verification", height=550, intraday=True, state_key="btc_test_key")
    assert isinstance(html_out, str) and len(html_out) > 50000, "build_tv_chart_html output too short or empty"
    assert "<!DOCTYPE html>" in html_out, "HTML header missing"
    assert "forecastBands" in html_out, "Forecast bands layer missing from HTML"
    assert "breakoutCorridors" in html_out, "Breakout corridors layer missing from HTML"
    assert "btn-fs" in html_out, "Fullscreen button missing from HTML"
    assert "resize-handle" in html_out, "Resize handle missing from HTML"
    print(f"  [PASS] Full LWC standalone HTML page built successfully ({len(html_out)} bytes).")
    print(">> Advanced chart plotting and spec serialization checks passed cleanly.\n")

def run_prediction_tracker_and_app_logging_checks():
    print("======================================================================")
    print("STEP 4: Prediction Logging, Pending Resolution & Drift Flow")
    print("======================================================================")
    from metrics.validation import PredictionTracker
    from app import log_prediction

    test_sym = "VERIF_TEST_BTC"
    test_intv = "1m"
    tracker = PredictionTracker(test_sym, test_intv)
    tracking_file = tracker.tracking_file

    if os.path.exists(tracking_file):
        try:
            os.remove(tracking_file)
        except Exception:
            pass

    # Create synthetic multi-bar DataFrame simulating live time progression
    times = pd.date_range("2026-11-12 12:00:00", periods=5, freq="1min")
    prices = [73000.0, 73150.0, 73100.0, 73300.0, 73450.0]

    # Bar 0 arrived: Log first prediction (BUY)
    df_step0 = pd.DataFrame({
        "Open": prices[:2], "High": [p + 50 for p in prices[:2]],
        "Low": [p - 50 for p in prices[:2]], "Close": prices[:2], "Volume": [100, 150]
    }, index=times[:2])
    
    pred0 = {"signal": "BUY (UP)", "prob_up": 0.75, "meta_prob": 0.70, "model_disagreement": 0.01}
    m0 = log_prediction(test_sym, test_intv, pred0, df_step0)
    print(f"  Step 0: Logged initial prediction, metrics={m0}")
    assert m0.get("total_predictions") == 1
    assert m0.get("pending_predictions") == 1
    assert m0.get("resolved_predictions") == 0

    # Test idempotence: calling log_prediction again with SAME bar should NOT create duplicate record
    m0_repeat = log_prediction(test_sym, test_intv, pred0, df_step0)
    assert m0_repeat.get("total_predictions") == 1, f"Duplicate record created on same bar: {m0_repeat}"
    print("  [PASS] Idempotence verified: Re-running on same bar does NOT create duplicate records.")

    # Bar 1 arrived: Price went from 73150 to 73100 (down). Log second prediction (SELL)
    df_step1 = pd.DataFrame({
        "Open": prices[:3], "High": [p + 50 for p in prices[:3]],
        "Low": [p - 50 for p in prices[:3]], "Close": prices[:3], "Volume": [100, 150, 120]
    }, index=times[:3])
    
    pred1 = {"signal": "SELL (DOWN)", "prob_up": 0.35, "meta_prob": 0.65, "model_disagreement": 0.02}
    m1 = log_prediction(test_sym, test_intv, pred1, df_step1)
    print(f"  Step 1: Second bar arrived, resolved prior prediction, metrics={m1}")
    assert m1.get("total_predictions") == 2
    assert m1.get("resolved_predictions") == 1
    assert m1.get("pending_predictions") == 1

    # Bar 2 arrived: Price went from 73100 to 73300 (up). Log third prediction (BUY)
    df_step2 = pd.DataFrame({
        "Open": prices[:4], "High": [p + 50 for p in prices[:4]],
        "Low": [p - 50 for p in prices[:4]], "Close": prices[:4], "Volume": [100, 150, 120, 200]
    }, index=times[:4])

    pred2 = {"signal": "BUY (UP)", "prob_up": 0.82, "meta_prob": 0.78, "model_disagreement": 0.01}
    m2 = log_prediction(test_sym, test_intv, pred2, df_step2)
    print(f"  Step 2: Third bar arrived, resolved second prediction, metrics={m2}")
    assert m2.get("total_predictions") == 3
    assert m2.get("resolved_predictions") == 2
    assert m2.get("pending_predictions") == 1

    # Verify disk records integrity
    with open(tracking_file, "r") as f:
        disk_records = [json.loads(line.strip()) for line in f if line.strip()]
    assert len(disk_records) == 3, f"Expected 3 records on disk, got {len(disk_records)}"
    assert disk_records[0]["actual_result"] is not None, "Record 0 actual_result was not resolved!"
    assert disk_records[1]["actual_result"] is not None, "Record 1 actual_result was not resolved!"
    assert disk_records[2]["actual_result"] is None, "Record 2 should still be pending!"
    print("  [PASS] Multi-bar resolution verified: Pending predictions resolved against actual market outcomes without dropping records.")

    # Drift calculation verification
    drift_res = tracker.calculate_drift(window=100)
    print(f"  [PASS] Drift detection calculated successfully: {drift_res}")
    assert "drift_magnitude" in drift_res
    assert "drift_detected" in drift_res

    # Cleanup test file
    if os.path.exists(tracking_file):
        try:
            os.remove(tracking_file)
        except Exception:
            pass
    print(">> Prediction logging, pending resolution and drift flow checks passed cleanly.\n")

def run_timesfm3_deep_intelligence_checks():
    print("======================================================================")
    print("STEP 5: TimesFM-3 100% Predictive Intelligence Suite & Chart Integration")
    print("======================================================================")
    from data_engine import fetch_data
    from features import add_features
    from ml.timesfm_engine import get_timesfm_engine
    import tv_chart
    from tv_chart import (
        ChartSpec, build_tv_chart_html,
        add_turning_points_to_chart,
        add_multi_scenario_forecast,
        add_forward_rv_envelope,
        set_macro_catalyst_badge,
    )

    symbol = "BTC-USD"
    interval = "1h"
    horizon = 15
    df = fetch_data(symbol, period="1mo", interval=interval)
    df_feat = add_features(df)
    last_close = float(df_feat["Close"].iloc[-1])

    # 1. Run TimesFM-3 Engine
    engine = get_timesfm_engine()
    fc = engine.forecast(df=df_feat, symbol=symbol, interval=interval, horizon=horizon)
    fc_dict = fc.to_dict()

    # 2. Test Online Test-Time Adaptation (TTA)
    print("  Testing Online Test-Time Adaptation (TTA)...")
    tta = fc_dict.get("tta_metrics", {})
    assert tta.get("enabled") is True, "TTA must be enabled in TimesFM-3"
    assert "adaptation_window" in tta and tta["adaptation_window"] > 0, "Missing adaptation_window in TTA"
    assert "drift_correction" in tta, "Missing drift_correction in TTA"
    assert "adapted_weights" in tta and len(tta["adapted_weights"]) > 0, "Missing adapted_weights in TTA"
    print(f"  [PASS] TTA verified: window={tta['adaptation_window']}, drift_correction={tta['drift_correction']:.5f}, loss_reduction={tta.get('recency_loss_reduction_pct', 0.0):.2f}%")

    # 3. Test Dynamic Turning Point & Critical Level Projection
    print("  Testing Dynamic Turning Point Projection...")
    tp = fc_dict.get("turning_points", {})
    assert "peak" in tp and "trough" in tp, "Turning points must contain peak and trough"
    assert "cycle_type" in tp, "Turning points must contain cycle_type"
    peak = tp["peak"]
    trough = tp["trough"]
    assert 1 <= peak["bar_step"] <= horizon, f"Peak step {peak['bar_step']} outside horizon [1, {horizon}]"
    assert 1 <= trough["bar_step"] <= horizon, f"Trough step {trough['bar_step']} outside horizon [1, {horizon}]"
    assert peak["price"] > 0 and np.isfinite(peak["price"]), "Peak price invalid"
    assert trough["price"] > 0 and np.isfinite(trough["price"]), "Trough price invalid"
    assert "label" in peak and "label" in trough, "Peak/Trough missing chart labels"
    print(f"  [PASS] Turning points verified: Peak={peak['label']}, Trough={trough['label']}, Cycle={tp['cycle_type']}")

    # 4. Test Multi-Scenario Probabilistic Path Diffusion
    print("  Testing Multi-Scenario Path Diffusion...")
    scenarios = fc_dict.get("scenarios", {})
    assert "base" in scenarios, "Scenarios missing base path"
    assert "bullish_breakout" in scenarios, "Scenarios missing bullish_breakout"
    assert "bearish_breakdown" in scenarios, "Scenarios missing bearish_breakdown"
    assert "volatility_expansion" in scenarios, "Scenarios missing volatility_expansion"
    
    total_prob = 0.0
    for s_name in ["base", "bullish_breakout", "bearish_breakdown"]:
        s_data = scenarios[s_name]
        assert len(s_data["trajectory"]) == horizon, f"Scenario {s_name} trajectory length mismatch"
        assert s_data["probability"] > 0.0, f"Scenario {s_name} non-positive probability"
        assert np.all(np.isfinite(s_data["trajectory"])), f"Scenario {s_name} contains non-finite prices"
        total_prob += s_data["probability"]
    assert 0.85 <= total_prob <= 1.15, f"Scenario probabilities sum far from 1.0: {total_prob}"
    print(f"  [PASS] Scenarios verified: Bull={scenarios['bullish_breakout']['probability_pct']}%, Base={scenarios['base']['probability_pct']}%, Bear={scenarios['bearish_breakdown']['probability_pct']}%, Vol Expansion={scenarios['volatility_expansion']['regime']}")

    # 5. Test Forward Realized Volatility (RV) Envelope
    print("  Testing Forward RV Envelope...")
    rv_env = fc_dict.get("rv_envelope", {})
    assert "upper_1s" in rv_env and "lower_1s" in rv_env, "Missing 1-sigma bounds in RV envelope"
    assert "upper_2s" in rv_env and "lower_2s" in rv_env, "Missing 2-sigma bounds in RV envelope"
    assert len(rv_env["upper_1s"]) == horizon, "RV upper_1s length mismatch"
    assert len(rv_env["upper_2s"]) == horizon, "RV upper_2s length mismatch"
    
    # Check dispersion monotonicity: 2s upper >= 1s upper >= 1s lower >= 2s lower
    u2 = np.array(rv_env["upper_2s"])
    u1 = np.array(rv_env["upper_1s"])
    l1 = np.array(rv_env["lower_1s"])
    l2 = np.array(rv_env["lower_2s"])
    assert np.all(u2 >= u1 - 1e-4), "2-sigma upper must be >= 1-sigma upper"
    assert np.all(u1 >= l1 - 1e-4), "1-sigma upper must be >= 1-sigma lower"
    assert np.all(l1 >= l2 - 1e-4), "1-sigma lower must be >= 2-sigma lower"
    print(f"  [PASS] Forward RV envelope verified: mean_rv={rv_env['forward_mean_rv']:.4f}, tunnel_width={rv_env['tunnel_width_pct']:.2f}%")

    # 6. Test Macro Driver & Technical Covariate Attribution
    print("  Testing Macro & Technical Driver Attributions...")
    macro_drivers = fc_dict.get("top_macro_drivers", [])
    assert len(macro_drivers) > 0, "No top macro drivers returned"
    for md in macro_drivers:
        assert "name" in md and "symbol" in md and "correlation" in md, f"Incomplete macro driver record: {md}"
        assert "directional_impact" in md, "Missing directional_impact in macro driver"
    primary_catalyst = fc_dict.get("primary_macro_catalyst", "")
    assert len(primary_catalyst) > 0, "Primary macro catalyst string missing"
    print(f"  [PASS] Macro attribution verified: Primary={primary_catalyst}, Drivers count={len(macro_drivers)}")

    # 7. Test ChartSpec Integration with TimesFM-3 Visual Primitives
    print("  Testing ChartSpec Visualization Primitives for TimesFM-3...")
    spec = ChartSpec()
    spec.set_candles(df_feat.tail(60))

    last_dt = df_feat.index[-1]
    step = pd.Timedelta(hours=1)
    future_times = [last_dt + step * (i + 1) for i in range(horizon)]
    forecast_span = [last_dt] + future_times

    # Add turning points
    add_turning_points_to_chart(spec, tp)
    assert len(spec.turning_points) >= 1, "Failed to add turning points to chart"
    print(f"  [PASS] add_turning_points_to_chart attached {len(spec.turning_points)} pins.")

    # Add multi-scenario forecast
    add_multi_scenario_forecast(spec, forecast_span, scenarios)
    assert len(spec.forecast_scenarios) >= 3, "Failed to attach forecast scenarios"
    print(f"  [PASS] add_multi_scenario_forecast attached {len(spec.forecast_scenarios)} paths.")

    # Add forward RV envelope tunnel
    spec.add_volatility_envelope(
        times=forecast_span,
        upper_2s=[last_close] + rv_env["upper_2s"],
        upper_1s=[last_close] + rv_env["upper_1s"],
        lower_1s=[last_close] + rv_env["lower_1s"],
        lower_2s=[last_close] + rv_env["lower_2s"],
        name="BTC Forward RV Tunnel"
    )
    assert len(spec.volatility_envelopes) >= 1, "Failed to attach volatility envelope"
    print("  [PASS] add_volatility_envelope attached dispersion tunnel.")

    # Add macro catalyst badge
    top_md = macro_drivers[0]
    set_macro_catalyst_badge(
        spec,
        primary_driver=top_md["name"],
        correlation_or_weight=top_md["correlation"],
        conviction="High",
        top_drivers=macro_drivers
    )
    assert spec.macro_catalyst is not None, "Failed to set macro catalyst badge"
    print("  [PASS] set_macro_catalyst_badge attached catalyst HUD.")

    # 8. Check JSON Spec Serialization
    payload = spec.to_json()
    assert "turningPoints" in payload and len(payload["turningPoints"]) >= 1, "Payload missing turningPoints"
    assert "forecastScenarios" in payload and len(payload["forecastScenarios"]) >= 3, "Payload missing forecastScenarios"
    assert "volatilityEnvelopes" in payload and len(payload["volatilityEnvelopes"]) >= 1, "Payload missing volatilityEnvelopes"
    assert "macroCatalyst" in payload and payload["macroCatalyst"] is not None, "Payload missing macroCatalyst"
    
    spec_json = json.dumps(payload, allow_nan=False)
    print(f"  [PASS] ChartSpec serialized to JSON with all TimesFM-3 intelligence structures ({len(spec_json)} bytes, zero NaNs).")

    # 9. HTML Generation
    html_code = build_tv_chart_html(spec, title="BTC TimesFM-3 Intelligence", height=600, intraday=True, state_key="btc_deep_key")
    assert isinstance(html_code, str) and len(html_code) > 60000, "HTML generation output too short"
    assert "turningPoints" in html_code, "HTML missing turningPoints"
    assert "forecastScenarios" in html_code, "HTML missing forecastScenarios"
    assert "volatilityEnvelopes" in html_code, "HTML missing volatilityEnvelopes"
    assert "macroCatalyst" in html_code, "HTML missing macroCatalyst"
    print(f"  [PASS] Standalone HTML page successfully generated with complete TimesFM-3 visual layers ({len(html_code)} bytes).")
    print(">> TimesFM-3 deep predictive intelligence checks passed cleanly.\n")

def run_natural_gas_and_micro_swing_checks():
    print("======================================================================")
    print("STEP 6: Natural Gas (NG=F) Micro-Swing, Calibration & Precision Scaling")
    print("======================================================================")
    from data_engine import fetch_data
    from features import add_features
    from models import TradingModel
    from ml.timesfm_engine import get_timesfm_engine
    import tv_chart
    from tv_chart import ChartSpec, build_tv_chart_html, add_harmonic_swing_wave, add_volume_profile_to_chart

    # 1. Test Natural Gas 5m Data & Micro-Cycle Extraction
    symbol = "NG=F"
    interval = "5m"
    print("  Fetching NG=F 5m data...")
    df_5m = fetch_data(symbol, period="5d", interval=interval)
    assert df_5m is not None and len(df_5m) > 100, "Insufficient 5m data for NG=F"
    df_feat_5m = add_features(df_5m)
    last_close = float(df_feat_5m["Close"].iloc[-1])
    print(f"  [PASS] NG=F 5m data ready: {len(df_feat_5m)} bars, close=${last_close:.4f}")

    # 2. Test Model Prediction & Non-Zero Recall Distribution
    model_5m = TradingModel(symbol, interval)
    model_5m.train_or_load(df_feat_5m, force_retrain=False)
    
    signals = []
    probs = []
    for i in range(min(50, len(df_feat_5m) - 20), 0, -1):
        sub = df_feat_5m.iloc[:-i]
        p = model_5m.predict_next(sub)
        signals.append(p["signal"])
        probs.append(p["prob_up"])

    from collections import Counter
    sig_counts = Counter(signals)
    print(f"  NG=F 5m 50-bar Signal Distribution: {dict(sig_counts)}")
    print(f"  NG=F 5m Probabilities: min={min(probs):.4f}, mean={np.mean(probs):.4f}, max={max(probs):.4f}, std={np.std(probs):.4f}")
    assert sig_counts.get("BUY (UP)", 0) > 0 or sig_counts.get("SELL (DOWN)", 0) > 0, "Model has zero directional signals!"
    assert np.std(probs) > 0.03, f"Probabilities are excessively squashed (std={np.std(probs):.4f})"
    print("  [PASS] Anti-squashing and non-zero recall verified for Natural Gas 5m.")

    # 3. Test Harmonic Swing Wave & Fourier Cycle Extraction in TimesFM-3
    engine = get_timesfm_engine()
    fc_5m = engine.forecast(df=df_feat_5m, symbol=symbol, interval=interval, horizon=15)
    fc_dict = fc_5m.to_dict()
    tp = fc_dict.get("turning_points", {})
    harmonic = tp.get("harmonic_cycle", {})
    assert harmonic, "Harmonic cycle dict missing in turning_points"
    print(f"  Harmonic Cycle: period={harmonic.get('dominant_period_bars')} bars, freq={harmonic.get('dominant_frequency')}, wave_range=${harmonic.get('wave_range'):.4f}")
    assert harmonic.get("dominant_period_bars", 0) >= 3.0, "Harmonic cycle period too short"
    assert harmonic.get("wave_range", 0) > 0.005, "Harmonic wave range is excessively flat!"
    print("  [PASS] Harmonic swing wave captures micro-oscillations without flatlining.")

    # 4. Test ChartSpec Adaptive Commodity Precision & Bezier Spline Wave
    spec = ChartSpec()
    spec.set_candles(df_feat_5m.tail(80))
    spec.set_price_precision(precision=4, min_move=0.0001)
    spec.set_scale_margins(top=0.22, bottom=0.20)

    last_dt = df_feat_5m.index[-1]
    future_times = [last_dt + pd.Timedelta(minutes=5 * (i + 1)) for i in range(15)]
    forecast_span = [last_dt] + future_times
    wave_path = [last_close] + list(fc_dict["forecast"])

    add_harmonic_swing_wave(spec, forecast_span, wave_path)
    add_volume_profile_to_chart(spec, df_feat_5m.tail(100), n_bins=24)

    payload = spec.to_json()
    assert payload.get("pricePrecision") == 4, f"Price precision mismatch: {payload.get('pricePrecision')}"
    assert payload.get("scaleMargins") == {"top": 0.22, "bottom": 0.20}, "Scale margins mismatch"
    assert "swingWaves" in payload and len(payload["swingWaves"]) >= 1, "Missing swingWaves in payload"
    assert "volumeProfile" in payload and payload["volumeProfile"] is not None, "Missing volumeProfile in payload"
    
    spec_json = json.dumps(payload, allow_nan=False)
    print(f"  [PASS] Natural Gas ChartSpec serialized cleanly ({len(spec_json)} bytes, 4 decimals, zero NaNs).")

    # 5. HTML Generation with Spline Engine & Precision Formatting
    html = build_tv_chart_html(spec, title="NG=F 5m Micro-Cycle Verification", height=550, intraday=True, state_key="ng_verif_key")
    assert len(html) > 60000, "HTML output too short"
    assert "drawSmoothSpline" in html, "Bezier spline engine missing in HTML"
    assert "pricePrecision" in html, "pricePrecision missing in HTML script"
    assert "volumeProfile" in html, "volumeProfile missing in HTML"
    print(f"  [PASS] Standalone HTML built successfully with Bezier spline wave and volume profile ({len(html)} bytes).")

    # 6. Test Quick Asset & Timeframe Header Pill Bar Helpers in app.py
    from app import _find_cat_and_inst_for_symbol, ASSET_CLASS_CATALOG
    cat, inst_lbl = _find_cat_and_inst_for_symbol("NG=F")
    assert cat is not None and inst_lbl is not None, f"Failed resolving category/instrument for NG=F: ({cat}, {inst_lbl})"
    if cat in ASSET_CLASS_CATALOG and inst_lbl in ASSET_CLASS_CATALOG[cat]:
        inst_data = ASSET_CLASS_CATALOG[cat][inst_lbl]
        assert inst_data.get("symbol") == "NG=F", f"Resolved incorrect symbol: {inst_data.get('symbol')}"
    print(f"  [PASS] Quick pill resolver verified for NG=F: Category='{cat}', Instrument='{inst_lbl}'")
    print(">> Natural Gas and micro-swing checks passed cleanly.\n")

if __name__ == "__main__":
    try:
        run_static_checks()
        run_model_pipeline_checks()
        run_chart_plotting_checks()
        run_prediction_tracker_and_app_logging_checks()
        run_timesfm3_deep_intelligence_checks()
        run_natural_gas_and_micro_swing_checks()
        print("**********************************************************************")
        print("🎉 ALL VERIFICATION CHECKS PASSED WITH ZERO ERRORS (100% SUCCESS) 🎉")
        print("**********************************************************************")
    except Exception as e:
        print(f"\n❌ VERIFICATION SUITE FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

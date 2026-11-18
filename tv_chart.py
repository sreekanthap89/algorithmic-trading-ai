"""
tv_chart.py — TradingView Lightweight Charts™ renderer for Streamlit.

The charting library is loaded DIRECTLY from the local source checkout in
``Tradingview/lightweight-charts`` (its built standalone bundle in ``dist/``)
and inlined into a Streamlit component, so no CDN / network is required.

Build the bundle once (only needed after updating the library source):

    cd Tradingview/lightweight-charts
    npx -y pnpm@10.34.5 install --filter lightweight-charts --ignore-scripts
    NODE_ENV=production npx -y pnpm@10.34.5 run tsc
    NODE_ENV=production npx -y pnpm@10.34.5 run rollup
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────
# Library loading
# ─────────────────────────────────────────────────────────────

LWC_ROOT = Path(__file__).resolve().parent / "Tradingview" / "lightweight-charts"
LWC_DIST = LWC_ROOT / "dist"
_BUNDLE_CANDIDATES = (
    "lightweight-charts.standalone.production.js",
    "lightweight-charts.standalone.development.js",
)

# Lightweight Charts LineStyle enum
SOLID, DOTTED, DASHED, LARGE_DASHED, SPARSE_DOTTED = 0, 1, 2, 3, 4

UP_COLOR = "#089981"
DOWN_COLOR = "#F23645"


class LightweightChartsNotBuilt(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_bundle_cached(path: str, mtime: float) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_lwc_bundle() -> str:
    """Return the standalone (IIFE) bundle source from the local Tradingview folder."""
    for name in _BUNDLE_CANDIDATES:
        p = LWC_DIST / name
        if p.exists():
            return _load_bundle_cached(str(p), p.stat().st_mtime)
    raise LightweightChartsNotBuilt(
        f"Lightweight Charts bundle not found in {LWC_DIST}. Build it with:\n"
        "  cd Tradingview/lightweight-charts && "
        "npx -y pnpm@10.34.5 install --filter lightweight-charts --ignore-scripts && "
        "NODE_ENV=production npx -y pnpm@10.34.5 run tsc && "
        "NODE_ENV=production npx -y pnpm@10.34.5 run rollup"
    )


# ─────────────────────────────────────────────────────────────
# Data conversion helpers
# ─────────────────────────────────────────────────────────────

def to_time(ts: Any) -> int:
    """
    Convert a timestamp to the UTC-seconds value Lightweight Charts expects.
    Tz-aware stamps keep their exchange wall-clock time (LWC renders in UTC).
    """
    if isinstance(ts, (int, float, np.integer, np.floating)):
        v = float(ts)
        if v > 1e11:  # milliseconds
            return int(v // 1000)
        if v > 1e8:   # already epoch seconds
            return int(v)
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return int(t.value // 1_000_000_000)


def _ok(v: Any) -> bool:
    try:
        return v is not None and not math.isnan(float(v)) and not math.isinf(float(v))
    except (TypeError, ValueError):
        return False


def _clean_points(points: Iterable[Tuple[Any, Any]]) -> List[Dict[str, float]]:
    """Sort by time, drop NaNs and duplicate times (LWC requires strictly ascending)."""
    by_time: Dict[int, float] = {}
    for t, v in points:
        if _ok(v):
            by_time[to_time(t)] = float(v)
    return [{"time": t, "value": by_time[t]} for t in sorted(by_time)]


def series_points(index: Sequence, values: Sequence) -> List[Dict[str, float]]:
    return _clean_points(zip(index, values))


class ChartSpec:
    """Collects everything the JS side needs to render one chart."""

    def __init__(self) -> None:
        self.candles: List[Dict[str, float]] = []
        self.lines: List[Dict[str, Any]] = []
        self.baselines: List[Dict[str, Any]] = []
        self.histograms: List[Dict[str, Any]] = []
        self.price_lines: List[Dict[str, Any]] = []
        self.markers: List[Dict[str, Any]] = []
        self._candle_times: set = set()

        # Advanced Predictive Analytics Plotting
        self.forecast_bands: List[Dict[str, Any]] = []
        self.directional_projections: List[Dict[str, Any]] = []
        self.breakout_corridors: List[Dict[str, Any]] = []
        self.prediction_badge: Optional[Dict[str, Any]] = None
        self.forecast_tooltips: Dict[int, Dict[str, Any]] = {}
        self._future_times: set = set()

        # TimesFM-3 Advanced Intelligence Plotting
        self.turning_points: List[Dict[str, Any]] = []
        self.forecast_scenarios: List[Dict[str, Any]] = []
        self.volatility_envelopes: List[Dict[str, Any]] = []
        self.macro_catalyst: Optional[Dict[str, Any]] = None

        # Harmonic Swing Waves & Adaptive Precision
        self.swing_waves: List[Dict[str, Any]] = []
        self.price_precision: Optional[int] = None
        self.price_min_move: Optional[float] = None
        self.scale_margins: Optional[Dict[str, float]] = None

        # Translucent Volume Profile
        self.volume_profile: Optional[Dict[str, Any]] = None
        self.intraday: bool = True

        # View Mode & Dynamic Client-Side Overlay Configuration
        self.view_mode: str = "🌊 Swing Wave & Cycles"
        self.default_overlays: Dict[str, bool] = {}

    def set_view_mode(self, mode: str, default_overlays: Optional[Dict[str, bool]] = None) -> 'ChartSpec':
        """
        Sets the active view mode (e.g. Swing Wave & Cycles, Execution Corridors,
        Probabilistic Quantile Cloud, All Overlays) and optional initial overlay visibility.
        """
        self.view_mode = str(mode)
        if default_overlays is not None:
            self.default_overlays = dict(default_overlays)
        return self

    # ── primitives ────────────────────────────────────────────
    def set_candles(self, df: pd.DataFrame) -> None:
        rows: Dict[int, Dict[str, float]] = {}
        for ts, o, h, l, c in zip(df.index, df["Open"], df["High"], df["Low"], df["Close"]):
            if all(_ok(x) for x in (o, h, l, c)):
                t = to_time(ts)
                rows[t] = {"time": t, "open": float(o), "high": float(h),
                           "low": float(l), "close": float(c)}
        self.candles = [rows[t] for t in sorted(rows)]
        self._candle_times = set(rows)

    def line(self, points: List[Dict[str, float]], color: str, width: float = 1.0,
             style: int = SOLID, title: str = "", pane: int = 0, legend: bool = False,
             last_value: bool = False, price_lines: Optional[List[Dict]] = None,
             tag: str = "") -> None:
        if len(points) < 1:
            return
        self.lines.append(dict(data=points, color=color, width=width, style=style,
                               title=title, pane=pane, legend=legend,
                               lastValue=last_value, priceLines=price_lines or [],
                               tag=tag))

    def polyline(self, pts: Sequence[Tuple[Any, float]], color: str, width: float = 1.5,
                 style: int = SOLID, title: str = "", legend: bool = False) -> None:
        self.line(_clean_points(pts), color, width, style, title, legend=legend)

    def hline(self, t0: Any, t1: Any, price: float, color: str, width: float = 1.2,
              style: int = DASHED, title: str = "") -> None:
        self.polyline([(t0, price), (t1, price)], color, width, style, title)

    def zone(self, times: Sequence, base: float, value: float, color: str, fill: str) -> None:
        """Shaded band between `base` and `value` over `times` (TP / SL zones)."""
        pts = _clean_points((t, value) for t in times)
        if pts:
            self.baselines.append(dict(data=pts, base=float(base), line=color, fill=fill))

    def histogram(self, data: List[Dict], title: str = "", pane: int = 1,
                  volume: bool = False) -> None:
        if data:
            self.histograms.append(dict(data=data, title=title, pane=pane, volume=volume))

    def price_line(self, price: float, color: str, title: str, style: int = DASHED,
                   width: int = 1, tag: str = "") -> None:
        if _ok(price):
            self.price_lines.append(dict(price=float(price), color=color, title=title,
                                         style=style, width=width, tag=tag))

    def marker(self, ts: Any, price: Optional[float], text: str, color: str,
               shape: str = "circle", position: str = "atPriceMiddle") -> None:
        t = to_time(ts)
        if t not in self._candle_times and t not in self._future_times:
            # Allow markers on future bars if registered in future times
            self._future_times.add(t)
        m = dict(time=t, position=position, shape=shape, color=color, text=text)
        if position.startswith("atPrice"):
            if not _ok(price):
                return
            m["price"] = float(price)
        self.markers.append(m)

    # ── Advanced Predictive Analytics & Plotting ──────────────
    def add_forecast_band(
        self,
        times: Sequence[Any],
        lower: Sequence[float],
        upper: Sequence[float],
        fill_color: str = "rgba(56, 189, 248, 0.12)",
        line_color: Optional[str] = "rgba(56, 189, 248, 0.40)",
        line_width: float = 1.0,
        line_style: int = DOTTED,
        name: str = "Confidence Band",
        gradient: bool = True,
    ) -> None:
        """
        Multi-horizon forecast confidence band/cloud between lower and upper quantiles
        (e.g. q10..q90 outer cloud or q30..q70 inner core).
        Renders smooth shaded polygon fill on the chart canvas plus boundary bounding lines.
        """
        lower_pts = _clean_points(zip(times, lower))
        upper_pts = _clean_points(zip(times, upper))
        if not lower_pts or not upper_pts:
            return

        for pt in upper_pts:
            self._future_times.add(pt["time"])
        for pt in lower_pts:
            self._future_times.add(pt["time"])

        # Add lines for LWC auto-scale and crosshair tracking
        if line_color and line_width > 0:
            self.lines.append(dict(
                data=upper_pts, color=line_color, width=line_width, style=line_style,
                title=f"{name} Upper", pane=0, legend=False, lastValue=False, priceLines=[]
            ))
            self.lines.append(dict(
                data=lower_pts, color=line_color, width=line_width, style=line_style,
                title=f"{name} Lower", pane=0, legend=False, lastValue=False, priceLines=[]
            ))

        self.forecast_bands.append(dict(
            name=name,
            upper=upper_pts,
            lower=lower_pts,
            fillColor=fill_color,
            lineColor=line_color or fill_color,
            lineWidth=line_width,
            lineStyle=line_style,
            gradient=gradient,
        ))

    def forecast_band(
        self,
        times: Sequence[Any],
        lower: Sequence[float],
        upper: Sequence[float],
        fill_color: str = "rgba(56, 189, 248, 0.12)",
        line_color: Optional[str] = "rgba(56, 189, 248, 0.40)",
        line_width: float = 1.0,
        line_style: int = DOTTED,
        name: str = "Confidence Band",
    ) -> None:
        """Alias for add_forecast_band."""
        self.add_forecast_band(times, lower, upper, fill_color, line_color, line_width, line_style, name)

    def add_directional_projection(
        self,
        times: Sequence[Any],
        values: Sequence[float],
        title: str = "Forecast Trajectory",
        bullish_color: str = "#00E676",
        bearish_color: str = "#FF5252",
        neutral_color: str = "#38BDF8",
        width: float = 2.4,
        style: int = SOLID,
        glow: bool = True,
        terminal_marker: bool = True,
    ) -> str:
        """
        Directional projection trajectory with dynamic color theme (Bullish / Bearish / Neutral).
        Returns the determined direction ('BULLISH', 'BEARISH', or 'NEUTRAL').
        """
        pts = _clean_points(zip(times, values))
        if len(pts) < 2:
            return "NEUTRAL"

        for pt in pts:
            self._future_times.add(pt["time"])

        start_val = pts[0]["value"]
        end_val = pts[-1]["value"]
        diff_pct = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0.0

        if diff_pct > 0.05:
            direction = "BULLISH"
            theme_color = bullish_color
            glow_color = "rgba(0, 230, 118, 0.45)"
        elif diff_pct < -0.05:
            direction = "BEARISH"
            theme_color = bearish_color
            glow_color = "rgba(255, 82, 82, 0.45)"
        else:
            direction = "NEUTRAL"
            theme_color = neutral_color
            glow_color = "rgba(56, 189, 248, 0.45)"

        # Add native LWC line
        self.line(
            pts,
            color=theme_color,
            width=width,
            style=style,
            title=f"{title} ({direction.capitalize()})",
            legend=True,
            last_value=True,
        )

        # Terminal target badge
        if terminal_marker and len(pts) >= 2:
            shape = "arrowUp" if direction == "BULLISH" else ("arrowDown" if direction == "BEARISH" else "circle")
            pos = "atPriceTop" if direction == "BEARISH" else "atPriceBottom"
            tgt_text = f"🎯 Target {end_val:.2f} ({diff_pct:+.1f}%)"
            self.markers.append(dict(
                time=pts[-1]["time"],
                price=end_val,
                position=pos,
                shape=shape,
                color=theme_color,
                text=tgt_text,
            ))

        self.directional_projections.append(dict(
            points=pts,
            title=title,
            direction=direction,
            strokeColor=theme_color,
            glowColor=glow_color,
            width=width,
            style=style,
            glow=glow,
            diffPct=diff_pct,
            startVal=start_val,
            endVal=end_val,
        ))
        return direction

    def add_breakout_corridor(
        self,
        times: Sequence[Any],
        entry: float,
        take_profit: float,
        stop_loss: float,
        label: str = "AI Trade Corridor",
        tp_label: Optional[str] = None,
        sl_label: Optional[str] = None,
        entry_label: Optional[str] = None,
        tp_fill: str = "rgba(0, 230, 118, 0.12)",
        sl_fill: str = "rgba(255, 82, 82, 0.12)",
        tp_border: str = "rgba(0, 230, 118, 0.70)",
        sl_border: str = "rgba(255, 82, 82, 0.70)",
        entry_color: str = "#29B6F6",
        show_rr_pill: bool = True,
        win_prob: Optional[float] = None,
    ) -> None:
        """
        Enhanced risk-reward breakout corridor with take profit, stop loss, and entry levels.
        Renders transparent shaded zones, breakout corridors, and central R:R metric badge.
        """
        if not _ok(entry) or not _ok(take_profit) or not _ok(stop_loss) or not times:
            return

        clean_times = [to_time(t) for t in times]
        for t in clean_times:
            self._future_times.add(t)

        start_time = clean_times[0]
        end_time = clean_times[-1]

        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        rr_ratio = (reward / risk) if risk > 0 else 1.0
        tp_pct = ((take_profit - entry) / entry * 100) if entry > 0 else 0.0
        sl_pct = (abs(stop_loss - entry) / entry * 100) if entry > 0 else 0.0

        # Native LWC price lines
        self.price_line(entry, entry_color, entry_label or f"⚡ ENTRY {entry:.2f}", DASHED, 1)
        self.price_line(
            take_profit,
            "#00E676",
            tp_label or f"🎯 TP {take_profit:.2f} (+{tp_pct:.1f}%)",
            DASHED,
            2,
        )
        self.price_line(
            stop_loss,
            "#FF5252",
            sl_label or f"🛑 SL {stop_loss:.2f} (-{sl_pct:.1f}%)",
            DASHED,
            2,
        )

        # Native LWC zones (BaselineSeries fallback)
        self.zone(times, entry, take_profit, tp_border, tp_fill)
        self.zone(times, entry, stop_loss, sl_border, sl_fill)

        # Breakout corridor metadata for canvas overlay
        self.breakout_corridors.append(dict(
            entry=float(entry),
            takeProfit=float(take_profit),
            stopLoss=float(stop_loss),
            startTime=start_time,
            endTime=end_time,
            label=label,
            tpLabel=tp_label or f"🎯 TARGET (TP +{tp_pct:.1f}%)",
            slLabel=sl_label or f"🛑 STOP LOSS (-{sl_pct:.1f}%)",
            entryLabel=entry_label or f"ENTRY {entry:.2f}",
            tpFill=tp_fill,
            slFill=sl_fill,
            tpBorder=tp_border,
            slBorder=sl_border,
            entryColor=entry_color,
            rrRatio=rr_ratio,
            tpPct=tp_pct,
            slPct=sl_pct,
            winProb=win_prob,
            showRRPill=show_rr_pill,
        ))

    def set_prediction_badge(
        self,
        accuracy: Optional[float] = None,
        confidence: Optional[float] = None,
        direction: Optional[str] = None,
        horizon_bars: Optional[int] = None,
        model_name: str = "TimesFM-3 + Ensemble",
        brier_score: Optional[float] = None,
        drift: Optional[float] = None,
        win_rate: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Configure the prediction accuracy & confidence HUD badge for the chart."""
        self.prediction_badge = dict(
            accuracy=accuracy,
            confidence=confidence,
            direction=direction,
            horizon_bars=horizon_bars,
            model_name=model_name,
            brier_score=brier_score,
            drift=drift,
            win_rate=win_rate,
            metrics=metrics or {},
        )

    def add_forecast_tooltip(self, ts: Any, data: Dict[str, Any]) -> None:
        """Register custom forecast tooltip data for a specific future bar timestamp."""
        t = to_time(ts)
        self._future_times.add(t)
        self.forecast_tooltips[t] = data

    def add_forecast_tooltips(self, tooltips: Dict[Any, Dict[str, Any]]) -> None:
        """Register multiple custom forecast tooltips."""
        for ts, d in tooltips.items():
            self.add_forecast_tooltip(ts, d)

    def add_forecast_quantiles(
        self,
        times: Sequence[Any],
        q50: Sequence[float],
        q10: Optional[Sequence[float]] = None,
        q90: Optional[Sequence[float]] = None,
        q30: Optional[Sequence[float]] = None,
        q70: Optional[Sequence[float]] = None,
        bullish_color: str = "#00E676",
        bearish_color: str = "#FF5252",
        neutral_color: str = "#38BDF8",
        outer_band_fill: str = "rgba(56, 189, 248, 0.12)",
        inner_band_fill: str = "rgba(129, 140, 248, 0.22)",
        title: str = "TimesFM-3 Forecast",
        confidence: Optional[float] = None,
        accuracy: Optional[float] = None,
        interval_label: str = "",
    ) -> str:
        """
        High-level all-in-one predictive plotting:
        - Outer 80% confidence cloud (q10..q90)
        - Inner 40% confidence core (q30..q70)
        - Dynamic directional trajectory path for q50 (Bullish vs Bearish)
        - Populates interactive future bar tooltips with quantile spreads and expected moves
        """
        clean_times = [to_time(t) for t in times]
        for t in clean_times:
            self._future_times.add(t)

        # 1. Outer confidence band (q10..q90)
        if q10 is not None and q90 is not None:
            self.add_forecast_band(
                times,
                q10,
                q90,
                fill_color=outer_band_fill,
                line_color="rgba(56, 189, 248, 0.40)",
                line_width=1.0,
                line_style=DOTTED,
                name="80% CI (q10–q90)",
            )

        # 2. Inner core band (q30..q70)
        if q30 is not None and q70 is not None:
            self.add_forecast_band(
                times,
                q30,
                q70,
                fill_color=inner_band_fill,
                line_color="rgba(129, 140, 248, 0.55)",
                line_width=1.1,
                line_style=DASHED,
                name="40% Core (q30–q70)",
            )

        # 3. Directional projection for q50
        direction = self.add_directional_projection(
            times,
            q50,
            title=title,
            bullish_color=bullish_color,
            bearish_color=bearish_color,
            neutral_color=neutral_color,
            width=2.5,
            style=SOLID,
            glow=True,
            terminal_marker=True,
        )

        # 4. Generate future bar tooltips
        base_val = float(q50[0]) if len(q50) > 0 else 1.0
        n_bars = len(clean_times)
        for i in range(1, n_bars):
            t = clean_times[i]
            cur_q50 = float(q50[i]) if i < len(q50) else None
            chg = ((cur_q50 - base_val) / base_val * 100) if (cur_q50 is not None and base_val != 0) else 0.0
            step_dir = "BULLISH" if chg > 0.02 else ("BEARISH" if chg < -0.02 else "NEUTRAL")

            tt_data = {
                "step": i,
                "horizon_label": f"+{i} bars {interval_label}".strip(),
                "expected_price": cur_q50,
                "change_pct": chg,
                "direction": step_dir,
                "q50": cur_q50,
                "q10": float(q10[i]) if (q10 is not None and i < len(q10)) else None,
                "q90": float(q90[i]) if (q90 is not None and i < len(q90)) else None,
                "q30": float(q30[i]) if (q30 is not None and i < len(q30)) else None,
                "q70": float(q70[i]) if (q70 is not None and i < len(q70)) else None,
                "confidence": confidence,
                "accuracy": accuracy,
            }
            self.forecast_tooltips[t] = tt_data

        return direction

    # ── TimesFM-3 Advanced Predictive Intelligence ─────────────
    def add_turning_point(
        self,
        ts: Any,
        price: float,
        kind: str = "peak",
        label: Optional[str] = None,
        step: Optional[int] = None,
        color: Optional[str] = None,
    ) -> None:
        """
        Plot a projected turning point (Peak or Trough) on a future forecast bar.
        - Peak: local maximum target, typically gold/cyan with arrowDown atPriceTop.
        - Trough: local minimum target, typically purple/coral with arrowUp atPriceBottom.
        """
        if not _ok(price):
            return
        t = to_time(ts)
        self._future_times.add(t)

        is_peak = str(kind).lower() in ("peak", "high", "top", "max")
        default_color = "#FFD700" if is_peak else "#E040FB"
        tp_color = color or default_color
        kind_str = "Peak" if is_peak else "Trough"
        step_str = f" (t+{step})" if step is not None else ""
        disp_label = label or f"{'▲' if is_peak else '▼'} {kind_str} {price:.2f}{step_str}"

        # Native LWC marker
        self.markers.append(dict(
            time=t,
            price=float(price),
            position="atPriceTop" if is_peak else "atPriceBottom",
            shape="arrowDown" if is_peak else "arrowUp",
            color=tp_color,
            text=disp_label,
        ))

        # Stored for canvas rendering
        self.turning_points.append(dict(
            time=t,
            price=float(price),
            kind="peak" if is_peak else "trough",
            label=disp_label,
            step=step,
            color=tp_color,
        ))

    def add_turning_points(self, points: Any) -> None:
        """Add multiple projected turning points. Accepts a sequence of dicts or a dictionary of turning points."""
        if isinstance(points, dict):
            pts_list = []
            for k in ("peak", "trough"):
                if k in points and isinstance(points[k], dict):
                    item = dict(points[k])
                    item.setdefault("kind", k)
                    pts_list.append(item)
            if not pts_list:
                pts_list = [v for v in points.values() if isinstance(v, dict)]
            points = pts_list

        for p in points:
            if not isinstance(p, dict):
                continue
            self.add_turning_point(
                ts=p.get("time") or p.get("ts") or p.get("target_time"),
                price=p.get("price") or p.get("value"),
                kind=p.get("kind", "peak"),
                label=p.get("label"),
                step=p.get("step") or p.get("bar_step"),
                color=p.get("color"),
            )

    def add_forecast_scenarios(
        self,
        times: Sequence[Any],
        scenarios: Dict[str, Any],
        styles: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Render distinct probabilistic scenario paths (Bullish Breakout, Base Path, Bearish Breakdown, Vol Expansion).
        - Bullish Breakout: neon lime (#00E676), dashed
        - Base Forecast: sky blue (#38BDF8), solid
        - Bearish Breakdown: vibrant coral (#FF5252), dashed
        - Volatility Expansion: amber (#FFD700), dotted
        """
        default_styles = {
            "bullish_breakout": dict(color="#00E676", style=DASHED, width=2.2, title="🚀 Bull Breakout"),
            "bullish": dict(color="#00E676", style=DASHED, width=2.2, title="🚀 Bull Breakout"),
            "bull": dict(color="#00E676", style=DASHED, width=2.2, title="🚀 Bull Breakout"),
            "breakout": dict(color="#00E676", style=DASHED, width=2.2, title="🚀 Bull Breakout"),
            "base": dict(color="#38BDF8", style=SOLID, width=2.5, title="🔮 Base Projection (q50)"),
            "base_path": dict(color="#38BDF8", style=SOLID, width=2.5, title="🔮 Base Projection (q50)"),
            "q50": dict(color="#38BDF8", style=SOLID, width=2.5, title="🔮 Base Projection (q50)"),
            "bearish_breakdown": dict(color="#FF5252", style=DASHED, width=2.2, title="💥 Bear Breakdown"),
            "bearish": dict(color="#FF5252", style=DASHED, width=2.2, title="💥 Bear Breakdown"),
            "bear": dict(color="#FF5252", style=DASHED, width=2.2, title="💥 Bear Breakdown"),
            "breakdown": dict(color="#FF5252", style=DASHED, width=2.2, title="💥 Bear Breakdown"),
            "volatility_expansion": dict(color="#FFD700", style=DOTTED, width=1.8, title="⚡ Vol Expansion"),
            "vol": dict(color="#FFD700", style=DOTTED, width=1.8, title="⚡ Vol Expansion"),
        }
        styles = styles or {}

        for name, vals in scenarios.items():
            raw_vals = vals.get("trajectory") if isinstance(vals, dict) else vals
            if raw_vals is None or not hasattr(raw_vals, "__iter__"):
                continue
            val_list = list(raw_vals)
            # If length is horizon while times is horizon + 1 (including last_close)
            if len(times) == len(val_list) + 1:
                times_to_use = times[1:]
            else:
                times_to_use = times[:len(val_list)]
            pts = _clean_points(zip(times_to_use, val_list))
            if len(pts) < 2:
                continue

            for pt in pts:
                self._future_times.add(pt["time"])

            norm_key = name.lower().replace("-", "_").replace(" ", "_")
            cfg = {**default_styles.get(norm_key, dict(color="#AB47BC", style=DASHED, width=2.0, title=name.capitalize())), **styles.get(name, {})}

            # Add to native LWC lines
            self.line(
                pts,
                color=cfg["color"],
                width=cfg.get("width", 2.0),
                style=cfg.get("style", DASHED),
                title=cfg.get("title", name),
                legend=True,
                last_value=True,
            )

            # Store for canvas trajectory rendering
            self.forecast_scenarios.append(dict(
                name=name,
                title=cfg.get("title", name),
                points=pts,
                color=cfg["color"],
                style=cfg.get("style", DASHED),
                width=cfg.get("width", 2.0),
            ))

    def add_volatility_envelope(
        self,
        times: Sequence[Any],
        upper_2s: Optional[Sequence[float]] = None,
        upper_1s: Optional[Sequence[float]] = None,
        lower_1s: Optional[Sequence[float]] = None,
        lower_2s: Optional[Sequence[float]] = None,
        center: Optional[Sequence[float]] = None,
        forward_rv: Optional[Any] = None,
        color_2s: str = "rgba(168, 85, 247, 0.08)",
        color_1s: str = "rgba(168, 85, 247, 0.16)",
        border_color: str = "rgba(168, 85, 247, 0.55)",
        name: str = "Forward RV Tunnel",
    ) -> None:
        """
        Forward Realized Volatility (RV) Envelope:
        Renders 1-sigma and 2-sigma forward volatility dispersion channels across the forecast horizon.
        Can receive explicit quantile paths or auto-derive from center + forward_rv.
        """
        n_pts = len(times)
        if n_pts < 2:
            return

        # Auto-derive sigma bounds if forward_rv and center are given
        if center is not None and forward_rv is not None and upper_2s is None:
            c_list = list(center)
            if isinstance(forward_rv, (int, float)):
                rv_pct = float(forward_rv)
                u2, u1, l1, l2 = [], [], [], []
                for i, c in enumerate(c_list):
                    scale = rv_pct * math.sqrt(i + 1)
                    u2.append(c * (1.0 + 2.0 * scale))
                    u1.append(c * (1.0 + 1.0 * scale))
                    l1.append(c * (1.0 - 1.0 * scale))
                    l2.append(c * (1.0 - 2.0 * scale))
                upper_2s, upper_1s, lower_1s, lower_2s = u2, u1, l1, l2
            elif isinstance(forward_rv, (list, tuple, Sequence)):
                rv_seq = list(forward_rv)
                u2, u1, l1, l2 = [], [], [], []
                for i, c in enumerate(c_list):
                    scale = float(rv_seq[i]) if i < len(rv_seq) else float(rv_seq[-1])
                    u2.append(c * (1.0 + 2.0 * scale))
                    u1.append(c * (1.0 + 1.0 * scale))
                    l1.append(c * (1.0 - 1.0 * scale))
                    l2.append(c * (1.0 - 2.0 * scale))
                upper_2s, upper_1s, lower_1s, lower_2s = u2, u1, l1, l2

        if upper_2s is None or lower_2s is None:
            return

        u2_pts = _clean_points(zip(times, upper_2s))
        l2_pts = _clean_points(zip(times, lower_2s))
        u1_pts = _clean_points(zip(times, upper_1s)) if upper_1s is not None else []
        l1_pts = _clean_points(zip(times, lower_1s)) if lower_1s is not None else []

        if len(u2_pts) < 2 or len(l2_pts) < 2:
            return

        for pt in u2_pts + l2_pts + u1_pts + l1_pts:
            self._future_times.add(pt["time"])

        # Add outer boundary rails to self.lines for LWC auto-scale
        self.lines.append(dict(
            data=u2_pts, color=border_color, width=1.0, style=DOTTED,
            title=f"{name} (+2σ)", pane=0, legend=False, lastValue=False, priceLines=[]
        ))
        self.lines.append(dict(
            data=l2_pts, color=border_color, width=1.0, style=DOTTED,
            title=f"{name} (-2σ)", pane=0, legend=False, lastValue=False, priceLines=[]
        ))

        self.volatility_envelopes.append(dict(
            name=name,
            upper2s=u2_pts,
            lower2s=l2_pts,
            upper1s=u1_pts,
            lower1s=l1_pts,
            color2s=color_2s,
            color1s=color_1s,
            borderColor=border_color,
        ))

    def set_macro_catalyst(
        self,
        primary_driver: str,
        correlation_or_weight: Optional[float] = None,
        conviction: str = "High",
        top_drivers: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Configure the in-chart Macro Catalyst & Model Conviction pill badge.
        e.g. primary_driver='DX-Y (-0.68)', conviction='High'.
        """
        self.macro_catalyst = dict(
            primary_driver=primary_driver,
            correlation_or_weight=correlation_or_weight,
            conviction=conviction,
            top_drivers=top_drivers or [],
        )

    # ── Harmonic Swing Waves & Precision Scaling ───────────────
    def set_price_precision(self, precision: int, min_move: Optional[float] = None) -> None:
        """
        Explicitly configure chart price decimal precision and minimum tick move.
        Essential for commodities (4 decimals on Natural Gas, e.g. $3.2780).
        """
        self.price_precision = int(precision)
        self.price_min_move = float(min_move) if min_move is not None else float(10 ** (-int(precision)))

    def set_scale_margins(self, top: float = 0.22, bottom: float = 0.20) -> None:
        """
        Configure top and bottom vertical scale margins (essential for low-priced commodities like Natural Gas).
        Prevents small price moves ($0.02) from looking flat or squashed vertically.
        """
        self.scale_margins = {"top": float(top), "bottom": float(bottom)}

    def auto_configure_precision(self, symbol: str = "", ref_price: Optional[float] = None) -> None:
        """
        Dynamically configures price precision and vertical scale margins based on symbol and price level.
        - Forex (EUR/USD, GBP/USD): 5 decimals, min_move 0.00001, margins 0.20/0.18
        - Natural Gas (NG=F): 4 decimals, min_move 0.0001, margins 0.22/0.20
        - Silver (SI=F): 3 decimals, min_move 0.001, margins 0.20/0.18
        - Crude Oil (CL=F): 2 decimals, min_move 0.01, margins 0.18/0.15
        - Gold (GC=F), Bitcoin (BTC-USD), Stocks: 2 decimals, min_move 0.01, margins 0.15/0.12
        """
        clean_sym = symbol.upper().replace("=X", "").replace("=F", "")
        if ref_price is None and self.candles:
            ref_price = self.candles[-1]["close"]
        p = float(ref_price or 100.0)

        # Forex
        if any(fx in clean_sym for fx in ("EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")) or (p < 2.0 and p > 0.5):
            self.set_price_precision(5, 0.00001)
            self.set_scale_margins(0.20, 0.18)
        # Natural Gas
        elif "NG" in clean_sym or (p < 5.0 and p >= 1.0):
            self.set_price_precision(4, 0.0001)
            self.set_scale_margins(0.22, 0.20)
        # Silver
        elif "SI" in clean_sym or ("SILVER" in clean_sym) or (p < 50.0 and p >= 15.0 and "CL" not in clean_sym):
            self.set_price_precision(3, 0.001)
            self.set_scale_margins(0.20, 0.18)
        # Crude Oil / Low-Medium
        elif "CL" in clean_sym or p < 100.0:
            self.set_price_precision(2, 0.01)
            self.set_scale_margins(0.18, 0.15)
        # Gold / Crypto / Equities
        else:
            self.set_price_precision(2, 0.01)
            self.set_scale_margins(0.15, 0.12)

    def add_swing_wave(
        self,
        times: Sequence[Any],
        values: Sequence[float],
        title: str = "Harmonic Swing Wave",
        smooth: bool = True,
        show_crest_tags: bool = True,
        cycle_frequency: Optional[float] = None,
        crest_color: str = "#00E5FF",
        trough_color: str = "#FF3D71",
        width: float = 2.8,
    ) -> None:
        """
        Plot a harmonic micro-cycle swing wave trajectory with smooth spline curvature,
        glowing multi-phase styling, and visual amplitude crest / trough tags.
        """
        pts = _clean_points(zip(times, values))
        if len(pts) < 2:
            return

        for pt in pts:
            self._future_times.add(pt["time"])

        start_p = pts[0]["value"]
        crests: List[Dict[str, Any]] = []
        troughs: List[Dict[str, Any]] = []

        # Extract local peaks (crests) and troughs (dips) along the wave
        for i in range(1, len(pts) - 1):
            prev_v = pts[i - 1]["value"]
            curr_v = pts[i]["value"]
            next_v = pts[i + 1]["value"]
            t_val = pts[i]["time"]
            try:
                time_dt = pd.to_datetime(t_val, unit="s") if isinstance(t_val, (int, float)) and t_val > 100000000 else pd.to_datetime(t_val)
                time_str = time_dt.strftime("%H:%M") if getattr(self, "intraday", True) else time_dt.strftime("%b %d")
                time_suffix = f" at {time_str}"
            except Exception:
                time_suffix = f" (t+{i})"

            if curr_v > prev_v and curr_v >= next_v:
                diff = curr_v - start_p
                pct = (diff / start_p * 100) if start_p else 0.0
                fmt_diff = f"+${abs(diff):.4f}" if start_p < 20 else f"+${abs(diff):.2f}"
                crests.append({
                    "time": pts[i]["time"],
                    "price": curr_v,
                    "step": i,
                    "diff": diff,
                    "pct": pct,
                    "label": f"▲ {fmt_diff} ({pct:+.2f}%) Crest{time_suffix}",
                })
            elif curr_v < prev_v and curr_v <= next_v:
                diff = curr_v - start_p
                pct = (diff / start_p * 100) if start_p else 0.0
                fmt_diff = f"-${abs(diff):.4f}" if start_p < 20 else f"-${abs(diff):.2f}"
                troughs.append({
                    "time": pts[i]["time"],
                    "price": curr_v,
                    "step": i,
                    "diff": diff,
                    "pct": pct,
                    "label": f"▼ {fmt_diff} ({pct:+.2f}%) Dip{time_suffix}",
                })

        # Add to native LWC lines for price auto-scaling
        self.line(
            pts,
            color=crest_color,
            width=width,
            style=SOLID,
            title=title,
            legend=True,
            last_value=True,
        )

        self.swing_waves.append(dict(
            points=pts,
            title=title,
            smooth=smooth,
            showCrestTags=show_crest_tags,
            crestColor=crest_color,
            troughColor=trough_color,
            width=width,
            cycleFrequency=cycle_frequency,
            crests=crests,
            troughs=troughs,
        ))

    def add_volume_profile(
        self,
        bins: Sequence[Dict[str, Any]],
        poc: Optional[float] = None,
        vah: Optional[float] = None,
        val: Optional[float] = None,
        align: str = "right",
        opacity: float = 0.28,
        show_labels: bool = True,
    ) -> None:
        """
        Attach a translucent volume profile overlay to the chart canvas.
        bins: list of dicts with keys 'price' and 'volume' (optional 'buy_vol', 'sell_vol').
        poc: Point of Control price level (highest volume level).
        vah: Value Area High (e.g. 70% volume envelope top).
        val: Value Area Low (e.g. 70% volume envelope bottom).
        align: 'right' (anchored to price scale) or 'left'.
        """
        clean_bins = []
        for b in bins:
            p = float(b.get("price", b.get("price_mid", 0.0)))
            v = float(b.get("volume", 0.0))
            if p > 0 and v >= 0:
                clean_bins.append({
                    "price": p,
                    "volume": v,
                    "buy_vol": float(b.get("buy_vol", v * 0.5)),
                    "sell_vol": float(b.get("sell_vol", v * 0.5)),
                })

        self.volume_profile = {
            "bins": clean_bins,
            "poc": float(poc) if poc is not None else None,
            "vah": float(vah) if vah is not None else None,
            "val": float(val) if val is not None else None,
            "align": align,
            "opacity": float(opacity),
            "showLabels": bool(show_labels),
        }

    def compute_volume_profile(
        self,
        n_bins: int = 24,
        align: str = "right",
        opacity: float = 0.28,
        value_area_pct: float = 0.70,
    ) -> None:
        """
        Automatically calculate and attach a horizontal volume profile from existing candle data.
        """
        if not self.candles:
            return

        min_p = min(c["low"] for c in self.candles)
        max_p = max(c["high"] for c in self.candles)
        if min_p >= max_p or n_bins < 4:
            return

        bin_width = (max_p - min_p) / n_bins
        bins_data = [{"price": min_p + (i + 0.5) * bin_width, "volume": 0.0, "buy_vol": 0.0, "sell_vol": 0.0} for i in range(n_bins)]

        for c in self.candles:
            vol = float(c.get("volume", abs(c["high"] - c["low"]) * 1000.0 or 1.0))
            mid = (c["high"] + c["low"]) / 2.0
            idx = int((mid - min_p) / bin_width)
            idx = max(0, min(n_bins - 1, idx))
            bins_data[idx]["volume"] += vol
            if c["close"] >= c["open"]:
                bins_data[idx]["buy_vol"] += vol * 0.6
                bins_data[idx]["sell_vol"] += vol * 0.4
            else:
                bins_data[idx]["buy_vol"] += vol * 0.4
                bins_data[idx]["sell_vol"] += vol * 0.6

        poc_bin = max(bins_data, key=lambda b: b["volume"])
        poc = poc_bin["price"]

        total_vol = sum(b["volume"] for b in bins_data)
        va_target = total_vol * value_area_pct
        poc_idx = bins_data.index(poc_bin)
        curr_vol = poc_bin["volume"]
        low_idx = poc_idx
        high_idx = poc_idx
        while curr_vol < va_target and (low_idx > 0 or high_idx < n_bins - 1):
            next_low_vol = bins_data[low_idx - 1]["volume"] if low_idx > 0 else -1
            next_high_vol = bins_data[high_idx + 1]["volume"] if high_idx < n_bins - 1 else -1
            if next_high_vol >= next_low_vol and high_idx < n_bins - 1:
                high_idx += 1
                curr_vol += bins_data[high_idx]["volume"]
            elif low_idx > 0:
                low_idx -= 1
                curr_vol += bins_data[low_idx]["volume"]
            else:
                break

        val = bins_data[low_idx]["price"]
        vah = bins_data[high_idx]["price"]

        self.add_volume_profile(
            bins=bins_data,
            poc=poc,
            vah=vah,
            val=val,
            align=align,
            opacity=opacity,
            show_labels=True,
        )

    def to_json(self) -> Dict[str, Any]:
        return dict(
            candles=self.candles,
            lines=self.lines,
            baselines=self.baselines,
            histograms=self.histograms,
            priceLines=self.price_lines,
            markers=sorted(self.markers, key=lambda m: m["time"]),
            forecastBands=self.forecast_bands,
            directionalProjections=self.directional_projections,
            breakoutCorridors=self.breakout_corridors,
            predictionBadge=self.prediction_badge,
            forecastTooltips=self.forecast_tooltips,
            turningPoints=self.turning_points,
            forecastScenarios=self.forecast_scenarios,
            volatilityEnvelopes=self.volatility_envelopes,
            macroCatalyst=self.macro_catalyst,
            swingWaves=self.swing_waves,
            volumeProfile=self.volume_profile,
            pricePrecision=self.price_precision,
            priceMinMove=self.price_min_move,
            scaleMargins=self.scale_margins,
            viewMode=self.view_mode,
            defaultOverlays=self.default_overlays,
        )


# ─────────────────────────────────────────────────────────────
# Pattern → drawing conversion (mirrors patterns.draw_* for Plotly)
# ─────────────────────────────────────────────────────────────

def add_patterns(spec: ChartSpec, detected: Dict[str, Dict], last_time: Any) -> None:
    if "wolfe_wave" in detected:
        p = detected["wolfe_wave"]
        p1, p2, p3, p4, p5, p6 = (p[k] for k in ("p1", "p2", "p3", "p4", "p5", "p6"))
        spec.polyline([p1, p3, p5], "#26A69A", 2, DOTTED, "🌊 Wolfe Wave", legend=True)
        spec.polyline([p2, p4, p6], "#EF5350", 2, DOTTED)
        spec.polyline([p1, p4, p6], "#FFD700", 1.8, DASHED)
        for lbl, pt, pos, col in [("1", p1, "atPriceBottom", "#87CEEB"),
                                  ("2", p2, "atPriceTop", "#87CEEB"),
                                  ("3", p3, "atPriceBottom", "#87CEEB"),
                                  ("4", p4, "atPriceTop", "#87CEEB"),
                                  ("5 Entry", p5, "atPriceBottom", "#00FF7F"),
                                  ("6 Target", p6, "atPriceTop", "#FFD700")]:
            spec.marker(pt[0], pt[1], lbl, col, "circle", pos)

    if "cup_handle" in detected:
        p = detected["cup_handle"]
        lr, ct, rm, ht, cur = (p[k] for k in ("left_rim", "cup_top", "right_min",
                                              "handle_top", "current"))
        spec.polyline([lr, ct, rm], "#64B5F6", 2.5, SOLID, "☕ Inv. Cup & Handle", legend=True)
        spec.polyline([rm, ht, cur], "#FFA726", 2.5)
        spec.hline(lr[0], cur[0], p["support"], "#EF5350", 1.5, DASHED)
        spec.hline(rm[0], cur[0], p["target"], "#26A69A", 1.5, DASHED)
        spec.marker(ct[0], ct[1], "Cup Top", "#64B5F6", "circle", "atPriceTop")
        spec.marker(cur[0], p["support"], "Breakout", "#FF5722", "arrowDown", "atPriceTop")
        spec.marker(cur[0], p["target"], "Target ↓", "#26A69A", "arrowDown", "atPriceBottom")

    if "bearish_bat" in detected:
        p = detected["bearish_bat"]
        X, A, B, C, D = p["X"], p["A"], p["B"], p["C"], p["D"]
        spec.polyline([X, A, B, C, D], "#EF9A9A", 1.8, SOLID, "🦇 Bearish Bat", legend=True)
        spec.polyline([X, D], "#FFA726", 1.4, DOTTED)
        spec.hline(X[0], last_time, p["stop"], "#EF5350", 1.5, DASHED, "Bat Stop-Loss")
        spec.hline(X[0], last_time, p["target"], "#26A69A", 1.5, DASHED, "Bat Target")
        d_label = (
            f"D Entry · XD {p['ratios']['XD_XA']:.3f}"
            if p.get("bars_ago", 999) <= 3
            else f"D · XD {p['ratios']['XD_XA']:.3f}"
        )
        for lbl, pt, pos, col in [("X", X, "atPriceTop", "#F48FB1"),
                                  ("A", A, "atPriceBottom", "#CE93D8"),
                                  ("B", B, "atPriceTop", "#90CAF9"),
                                  ("C", C, "atPriceBottom", "#A5D6A7"),
                                  (d_label, D, "atPriceTop", "#FFCC02")]:
            spec.marker(pt[0], pt[1], lbl, col, "circle", pos)

    if "falling_wedge" in detected:
        p = detected["falling_wedge"]
        h1, h2, l1, l2, apex = p["h1"], p["h2"], p["l1"], p["l2"], p["apex"]
        spec.polyline([h1, h2, apex], "#EF5350", 2, SOLID, "📐 Falling Wedge", legend=True)
        spec.polyline([l1, l2, apex], "#26A69A", 2)
        spec.hline(h1[0], apex[0], p["target"], "#FFD700", 1.5, DASHED, "FW Target")
        spec.hline(l1[0], apex[0], p["stop"], "#EF5350", 1.2, DOTTED, "FW Stop")
        if p.get("bars_ago", 999) <= 3:
            spec.marker(apex[0], apex[1], "Position Opening", "#00FF7F", "arrowUp", "atPriceBottom")
        else:
            spec.marker(apex[0], apex[1], "FW Apex", "#26A69A", "circle", "atPriceBottom")


def add_deal_markers(spec: ChartSpec, deal_signals: Dict[str, Any]) -> None:
    """
    Appends AI Buy/Sell Entry and Deal Close (TP / SL) markers to ChartSpec.
    """
    if not deal_signals or not isinstance(deal_signals, dict):
        return
    markers = deal_signals.get("markers", [])
    for m in markers:
        t = m.get("time")
        if t in spec._candle_times or t in spec._future_times:
            existing = any(
                em.get("time") == t and em.get("position") == m.get("position")
                for em in spec.markers
            )
            if not existing:
                spec.markers.append(m)


def add_prediction_forecast(
    spec: ChartSpec,
    timesfm_res: Dict[str, Any],
    forecast_span: Sequence[Any],
    last_close: float,
    accuracy: Optional[float] = None,
    confidence: Optional[float] = None,
    interval: str = "",
) -> Optional[str]:
    """
    Convenience helper to attach multi-horizon forecast quantiles and tooltips to a ChartSpec.
    Returns the forecast direction ('BULLISH', 'BEARISH', or 'NEUTRAL').
    """
    if not timesfm_res or not timesfm_res.get("q50"):
        return None
    q50_path = [last_close] + list(timesfm_res.get("q50", []))
    q90_path = [last_close] + list(timesfm_res.get("q90", [])) if "q90" in timesfm_res else None
    q10_path = [last_close] + list(timesfm_res.get("q10", [])) if "q10" in timesfm_res else None
    q70_path = [last_close] + list(timesfm_res.get("q70", [])) if "q70" in timesfm_res else None
    q30_path = [last_close] + list(timesfm_res.get("q30", [])) if "q30" in timesfm_res else None

    return spec.add_forecast_quantiles(
        forecast_span,
        q50=q50_path,
        q10=q10_path,
        q90=q90_path,
        q30=q30_path,
        q70=q70_path,
        confidence=confidence or timesfm_res.get("confidence"),
        accuracy=accuracy or timesfm_res.get("accuracy"),
        interval_label=interval,
    )


def add_risk_reward_corridor(
    spec: ChartSpec,
    forecast_span: Sequence[Any],
    entry: float,
    take_profit: float,
    stop_loss: float,
    label: str = "AI Trade Corridor",
    win_prob: Optional[float] = None,
) -> None:
    """
    Convenience helper to attach an enhanced risk-reward breakout corridor to a ChartSpec.
    """
    spec.add_breakout_corridor(
        times=forecast_span,
        entry=entry,
        take_profit=take_profit,
        stop_loss=stop_loss,
        label=label,
        win_prob=win_prob,
    )


def add_turning_points_to_chart(
    spec: ChartSpec,
    turning_points: Sequence[Dict[str, Any]],
) -> None:
    """Convenience helper to attach projected turning points (peaks and troughs) to a ChartSpec."""
    spec.add_turning_points(turning_points)


def add_multi_scenario_forecast(
    spec: ChartSpec,
    times: Sequence[Any],
    scenarios: Dict[str, Sequence[float]],
    styles: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Convenience helper to attach multi-scenario forecast paths (Bull/Base/Bear) to a ChartSpec."""
    spec.add_forecast_scenarios(times, scenarios, styles)


def add_forward_rv_envelope(
    spec: ChartSpec,
    times: Sequence[Any],
    center: Sequence[float],
    forward_rv: Any,
    name: str = "Forward RV Tunnel",
) -> None:
    """Convenience helper to attach a forward volatility dispersion tunnel to a ChartSpec."""
    spec.add_volatility_envelope(times, center=center, forward_rv=forward_rv, name=name)


def set_macro_catalyst_badge(
    spec: ChartSpec,
    primary_driver: str,
    correlation_or_weight: Optional[float] = None,
    conviction: str = "High",
    top_drivers: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Convenience helper to set macro catalyst metadata on a ChartSpec."""
    spec.set_macro_catalyst(primary_driver, correlation_or_weight, conviction, top_drivers)


def add_harmonic_swing_wave(
    spec: ChartSpec,
    times: Sequence[Any],
    values: Sequence[float],
    title: str = "Harmonic Swing Wave",
    smooth: bool = True,
    show_crest_tags: bool = True,
    crest_color: str = "#00E5FF",
    trough_color: str = "#FF3D71",
    width: float = 2.8,
) -> None:
    """Convenience helper to attach a harmonic swing wave with micro-swings to a ChartSpec."""
    spec.add_swing_wave(
        times=times,
        values=values,
        title=title,
        smooth=smooth,
        show_crest_tags=show_crest_tags,
        crest_color=crest_color,
        trough_color=trough_color,
        width=width,
    )


def add_volume_profile_to_chart(
    spec: ChartSpec,
    bins: Optional[Any] = None,
    poc: Optional[float] = None,
    vah: Optional[float] = None,
    val: Optional[float] = None,
    align: str = "right",
    opacity: float = 0.28,
    auto_compute: bool = False,
    n_bins: int = 24,
) -> None:
    """Convenience helper to attach a translucent volume profile to a ChartSpec."""
    if auto_compute or bins is None:
        spec.compute_volume_profile(n_bins=n_bins, align=align, opacity=opacity)
    elif isinstance(bins, (list, tuple)) and len(bins) > 0:
        spec.add_volume_profile(bins=bins, poc=poc, vah=vah, val=val, align=align, opacity=opacity)
    elif hasattr(bins, "empty"):
        spec.compute_volume_profile(n_bins=n_bins, align=align, opacity=opacity)
    else:
        spec.compute_volume_profile(n_bins=n_bins, align=align, opacity=opacity)


# ─────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  :root { --bg:#0D1117; --panel:rgba(19,23,34,.82); --border:rgba(255,255,255,.10); --txt:#D1D4DC; --muted:#787B86; }
  html,body { margin:0; padding:0; background:var(--bg); overflow:hidden;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  #wrap { position:relative; width:100%; height:__HEIGHT__px; border:1px solid var(--border);
    border-radius:10px; overflow:hidden; box-sizing:border-box; }
  #chart { position:absolute; inset:0; }
  #legend { position:absolute; top:8px; left:10px; z-index:5; pointer-events:none;
    display:flex; flex-wrap:wrap; align-items:center; gap:5px; font-size:11.5px; line-height:1.4; color:var(--txt); }
  .legend-badge {
    background:rgba(19, 23, 34, 0.88); border:1px solid rgba(255,255,255,0.12);
    border-radius:6px; padding:3px 8px; backdrop-filter:blur(8px);
    display:inline-flex; align-items:center; gap:4px; box-shadow:0 2px 8px rgba(0,0,0,0.3);
  }
  .legend-badge .sym { font-weight:700; font-size:12px; color:#fff; }
  .legend-badge .k { color:var(--muted); font-size:10.5px; font-weight:600; margin-left:3px; }
  .legend-badge .up { color:#00E676; font-weight:600; }
  .legend-badge .dn { color:#FF3D71; font-weight:600; }
  #legend .items { display:flex; flex-wrap:wrap; gap:4px 6px; }
  #legend .item-pill {
    background:rgba(19, 23, 34, 0.85); border:1px solid rgba(255,255,255,0.10);
    border-radius:5px; padding:2px 7px; backdrop-filter:blur(6px);
    display:inline-flex; align-items:center; gap:4px; font-size:10.5px;
  }
  #legend .dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:3px; }
  /* Tools & Markings styling */
  #tools { position:absolute; top:8px; right:78px; z-index:6; display:flex; gap:6px; align-items:center; }
  .tool-sep { width:1px; height:16px; background:rgba(255,255,255,0.12); margin:0 2px; }
  .tool-group { display:flex; gap:4px; }
  #tools button { background:var(--panel); color:var(--txt); border:1px solid var(--border);
    border-radius:6px; padding:4px 9px; font-size:12px; cursor:pointer; backdrop-filter:blur(6px);
    transition:all .15s ease; display:inline-flex; align-items:center; gap:4px; }
  #tools button:hover:not(:disabled) { border-color:#2962FF; color:#fff; transform:translateY(-1px); }
  #tools button:disabled { opacity:0.35; cursor:not-allowed; }
  #tools button.active { border-color:#2962FF; background:rgba(41,98,255,0.25); color:#fff; font-weight:600; }
  #err { color:#FF5252; padding:12px; font-size:13px; white-space:pre-wrap; }

  /* AI Overlays Dropdown Menu Styling (Zero-Refresh Instant Controls) */
  #ai-overlays-menu {
    position: absolute; top: 42px; right: 195px; z-index: 15;
    background: rgba(19, 23, 34, 0.96); border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 8px; padding: 10px 14px; box-shadow: 0 10px 32px rgba(0, 0, 0, 0.75);
    backdrop-filter: blur(12px); display: flex; flex-direction: column; gap: 5px;
    min-width: 250px; user-select: none;
  }
  #ai-overlays-menu.menu-hidden { display: none !important; }
  .overlay-menu-header {
    font-size: 10.5px; font-weight: 700; color: #29B6F6; text-transform: uppercase;
    letter-spacing: 0.6px; padding-bottom: 5px; border-bottom: 1px solid rgba(255, 255, 255, 0.10);
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;
  }
  .overlay-item {
    display: flex; align-items: center; gap: 8px; font-size: 11.5px; color: var(--txt);
    cursor: pointer; padding: 4px 6px; border-radius: 5px; transition: background .12s ease;
  }
  .overlay-item:hover {
    background: rgba(255, 255, 255, 0.08); color: #fff;
  }
  .overlay-item input[type="checkbox"] {
    accent-color: #2962FF; cursor: pointer; width: 14px; height: 14px; margin: 0;
  }
  .overlay-menu-divider {
    height: 1px; background: rgba(255, 255, 255, 0.08); margin: 3px 0;
  }

  /* Resizer handle bar at bottom of chart */
  #resize-handle {
    position: absolute; bottom: 0; left: 0; right: 0; height: 11px;
    background: rgba(19, 23, 34, 0.75); border-top: 1px solid rgba(255, 255, 255, 0.12);
    cursor: ns-resize; z-index: 20; display: flex; align-items: center; justify-content: center;
    user-select: none; transition: all .15s ease;
  }
  #resize-handle:hover, #resize-handle.resizing {
    background: rgba(41, 98, 255, 0.45); border-top-color: #2962FF;
  }
  .resize-grip {
    width: 36px; height: 3px; border-radius: 2px; background: rgba(255, 255, 255, 0.35);
  }
  #resize-handle:hover .resize-grip, #resize-handle.resizing .resize-grip {
    background: #fff; box-shadow: 0 0 6px rgba(41, 98, 255, 0.9);
  }

  /* Fullscreen & Pseudo-Fullscreen styling */
  :fullscreen, :-webkit-full-screen {
    width: 100vw !important; height: 100vh !important;
    background: #0D1117 !important; margin: 0 !important; padding: 0 !important; overflow: hidden !important;
  }
  :fullscreen #wrap, :-webkit-full-screen #wrap {
    width: 100vw !important; height: 100vh !important;
    border: none !important; border-radius: 0 !important;
  }
  :fullscreen #chart, :-webkit-full-screen #chart {
    width: 100vw !important; height: 100vh !important;
  }
  body.pseudo-fullscreen {
    position: fixed !important; inset: 0 !important; width: 100vw !important; height: 100vh !important;
    z-index: 999999 !important; background: #0D1117 !important;
  }
  body.pseudo-fullscreen #wrap {
    width: 100vw !important; height: 100vh !important;
    border: none !important; border-radius: 0 !important;
  }
  body.pseudo-fullscreen #chart {
    width: 100vw !important; height: 100vh !important;
  }

  /* Draw Palette */
  #draw-palette {
    position:absolute; top:42px; right:78px; z-index:8; display:flex; align-items:center; gap:6px;
    background:rgba(19, 23, 34, 0.95); border:1px solid rgba(255, 255, 255, 0.16);
    border-radius:8px; padding:6px 10px; box-shadow:0 8px 24px rgba(0,0,0,0.6);
    backdrop-filter:blur(8px); flex-wrap:nowrap;
  }
  #draw-palette.palette-hidden { display:none; }
  .palette-title { font-size:10px; color:var(--muted); text-transform:uppercase; font-weight:700; margin-right:2px; }
  .palette-btn {
    background:rgba(255,255,255,0.06); color:var(--txt); border:1px solid var(--border);
    border-radius:5px; padding:4px 7px; font-size:11px; cursor:pointer; transition:all .15s ease;
    display:inline-flex; align-items:center; gap:3px;
  }
  .palette-btn:hover { background:rgba(255,255,255,0.14); color:#fff; }
  .palette-btn.active { background:rgba(41,98,255,0.3); border-color:#2962FF; color:#fff; font-weight:600; }
  .pal-sep { width:1px; height:18px; background:rgba(255,255,255,0.15); margin:0 3px; }
  
  /* Color Dots */
  .color-dot {
    width:16px; height:16px; border-radius:50%; border:2px solid transparent;
    cursor:pointer; padding:0; transition:all .15s ease; outline:none;
  }
  .color-dot:hover { transform:scale(1.25); }
  .color-dot.active { border-color:#fff; box-shadow:0 0 6px rgba(255,255,255,0.9); transform:scale(1.25); }

  /* Size toggle */
  .size-btn {
    background:rgba(255,255,255,0.06); color:var(--txt); border:1px solid var(--border);
    border-radius:4px; padding:2px 6px; font-size:10px; cursor:pointer; font-weight:600;
  }
  .size-btn.active { background:#2962FF; color:#fff; border-color:#2962FF; }

  .palette-close {
    background:rgba(255,255,255,0.08); border:1px solid var(--border); border-radius:4px;
    color:var(--txt); cursor:pointer; padding:3px 8px; font-size:11px; margin-left:4px;
  }
  .palette-close:hover { background:#2962FF; color:#fff; }

  /* Draw Guide Banner */
  #draw-banner {
    position:absolute; top:12px; left:50%; transform:translateX(-50%); z-index:10;
    display:flex; align-items:center; gap:10px; background:rgba(21, 101, 192, 0.95);
    color:#fff; padding:6px 14px; border-radius:20px; font-size:12px; font-weight:500;
    box-shadow:0 4px 16px rgba(0,0,0,0.5); border:1px solid rgba(255,255,255,0.25);
  }
  #draw-banner.banner-hidden { display:none; }
  #banner-cancel {
    background:rgba(255,255,255,0.22); color:#fff; border:none; border-radius:10px;
    padding:2px 8px; font-size:11px; cursor:pointer;
  }
  #banner-cancel:hover { background:rgba(255,255,255,0.38); }

  /* Markings Modal / Drawer */
  #markings-modal {
    position:absolute; top:42px; right:78px; width:320px; max-height:380px; z-index:15;
    background:#131722; border:1px solid rgba(255, 255, 255, 0.18); border-radius:8px;
    box-shadow:0 12px 36px rgba(0,0,0,0.75); display:flex; flex-direction:column; overflow:hidden;
  }
  #markings-modal.modal-hidden { display:none; }
  .modal-header {
    display:flex; justify-content:space-between; align-items:center; padding:9px 12px;
    border-bottom:1px solid rgba(255,255,255,0.08); font-size:12px; font-weight:600; color:#fff;
  }
  .modal-close { background:none; border:none; color:var(--muted); cursor:pointer; font-size:14px; }
  .modal-close:hover { color:#fff; }
  .modal-body { overflow-y:auto; max-height:240px; padding:6px; }
  .modal-empty { padding:18px 10px; text-align:center; color:var(--muted); font-size:12px; }
  .mark-item {
    display:flex; align-items:center; justify-content:space-between; padding:6px 8px;
    margin-bottom:4px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
    border-radius:5px; font-size:11px;
  }
  .mark-badge { display:inline-flex; align-items:center; gap:5px; font-weight:600; }
  .mark-del { background:none; border:none; color:#FF5252; cursor:pointer; font-size:13px; padding:2px 5px; border-radius:3px; }
  .mark-del:hover { background:rgba(255,82,82,0.18); }
  .modal-footer {
    display:flex; justify-content:space-between; align-items:center; padding:8px 12px; border-top:1px solid rgba(255,255,255,0.08);
  }
  .btn-danger {
    background:rgba(255,82,82,0.15); color:#FF5252; border:1px solid rgba(255,82,82,0.4);
    border-radius:5px; padding:4px 10px; font-size:11px; cursor:pointer;
  }
  .btn-danger:hover { background:#FF5252; color:#fff; }

  /* Canvases: Persistent Markings Layer & Active Drawing Layer */
  #forecast-canvas {
    position:absolute; inset:0; width:100%; height:100%; z-index:4; pointer-events:none;
  }
  #markings-canvas {
    position:absolute; inset:0; width:100%; height:100%; z-index:5; pointer-events:none;
  }
  #draw-canvas {
    position:absolute; inset:0; width:100%; height:100%; z-index:6; pointer-events:none;
  }
  #draw-canvas.active { pointer-events:auto; cursor:crosshair; }
  #draw-canvas.eraser-mode { cursor:cell; }

  /* Prediction Accuracy & Confidence HUD Card */
  .pred-hud {
    position:absolute; top:38px; left:10px; z-index:7;
    background:rgba(19, 23, 34, 0.92); border:1px solid rgba(255, 255, 255, 0.14);
    border-radius:8px; padding:7px 11px; box-shadow:0 8px 24px rgba(0,0,0,0.6);
    backdrop-filter:blur(8px); font-size:11px; color:var(--txt);
    display:flex; flex-direction:column; gap:4px; transition:all .2s ease;
    user-select:none; max-width:380px;
  }
  .pred-hud.collapsed { padding:4px 8px; }
  .pred-hud.collapsed .pred-hud-body { display:none; }
  .pred-hud-header {
    display:flex; align-items:center; justify-content:space-between; gap:8px;
    font-weight:600; color:#fff;
  }
  .pred-hud-title { display:flex; align-items:center; gap:5px; font-size:11.5px; }
  .pred-hud-badge {
    padding:2px 6px; border-radius:4px; font-size:10px; font-weight:700; text-transform:uppercase;
  }
  .pred-hud-badge.up { background:rgba(0, 230, 118, 0.2); color:#00E676; border:1px solid rgba(0, 230, 118, 0.4); }
  .pred-hud-badge.dn { background:rgba(255, 82, 82, 0.2); color:#FF5252; border:1px solid rgba(255, 82, 82, 0.4); }
  .pred-hud-badge.nt { background:rgba(56, 189, 248, 0.2); color:#38BDF8; border:1px solid rgba(56, 189, 248, 0.4); }
  .pred-hud-toggle {
    background:none; border:none; color:var(--muted); cursor:pointer; font-size:12px; padding:0 3px;
  }
  .pred-hud-toggle:hover { color:#fff; }
  .pred-hud-body {
    display:flex; flex-wrap:wrap; gap:4px 10px; border-top:1px solid rgba(255,255,255,0.08);
    padding-top:4px; margin-top:2px;
  }
  .pred-hud-stat { display:inline-flex; align-items:center; gap:4px; }
  .pred-hud-stat .k { color:var(--muted); font-size:10.5px; }
  .pred-hud-stat .v { font-weight:600; color:#fff; }
  .pred-hud-stat .v.highlight { color:#00E676; }

  /* Interactive Crosshair Forecast Tooltip */
  .forecast-tooltip {
    position:absolute; z-index:12; pointer-events:none; opacity:0;
    background:rgba(13, 17, 23, 0.94); border:1px solid rgba(56, 189, 248, 0.45);
    border-radius:8px; padding:9px 12px; box-shadow:0 10px 30px rgba(0,0,0,0.8), 0 0 12px rgba(56, 189, 248, 0.25);
    backdrop-filter:blur(10px); font-size:11px; color:#D1D4DC;
    transition:opacity .12s ease; min-width:220px;
  }
  .forecast-tooltip.show { opacity:1; }
  .ft-head {
    display:flex; align-items:center; justify-content:space-between;
    margin-bottom:6px; padding-bottom:5px; border-bottom:1px solid rgba(255,255,255,0.1);
  }
  .ft-title { font-weight:700; color:#fff; font-size:12px; }
  .ft-dir {
    padding:1px 6px; border-radius:3px; font-weight:700; font-size:9.5px; text-transform:uppercase;
  }
  .ft-dir.up { background:rgba(0, 230, 118, 0.2); color:#00E676; }
  .ft-dir.dn { background:rgba(255, 82, 82, 0.2); color:#FF5252; }
  .ft-row { display:flex; justify-content:space-between; margin:3px 0; gap:8px; }
  .ft-label { color:#8B949E; }
  .ft-val { font-weight:600; color:#fff; }
  .ft-val.up { color:#00E676; }
  .ft-val.dn { color:#FF5252; }
  .ft-val.bold { font-size:12px; }
  .ft-val.highlight { color:#38BDF8; font-weight:700; }

  /* Macro Catalyst & Conviction Pill Badge */
  .macro-badge {
    position:absolute; top:38px; right:82px; z-index:7;
    background:rgba(19, 23, 34, 0.92); border:1px solid rgba(255, 215, 0, 0.35);
    border-radius:20px; padding:4px 12px; box-shadow:0 6px 20px rgba(0,0,0,0.5);
    backdrop-filter:blur(8px); font-size:11px; color:#fff;
    display:flex; align-items:center; gap:7px; user-select:none;
    transition:all .2s ease;
  }
  .macro-badge:hover { border-color:#FFD700; transform:translateY(-1px); }
  .macro-badge-icon { font-size:13px; }
  .macro-badge-driver { font-weight:700; color:#FFD700; }
  .macro-badge-conv {
    background:rgba(0, 230, 118, 0.2); color:#00E676; border:1px solid rgba(0, 230, 118, 0.4);
    border-radius:10px; padding:1px 6px; font-size:9.5px; font-weight:700; text-transform:uppercase;
  }

  /* Toast Notification */
  #draw-toast {
    position:absolute; bottom:24px; right:24px; z-index:20; background:rgba(30, 34, 45, 0.95);
    color:#fff; border:1px solid rgba(255,255,255,0.2); border-radius:6px; padding:6px 12px;
    font-size:11px; pointer-events:none; opacity:0; transform:translateY(6px);
    transition:all .2s ease; box-shadow:0 4px 16px rgba(0,0,0,0.5);
  }
  #draw-toast.show { opacity:1; transform:translateY(0); }
</style></head>
<body>
<div id="wrap">
  <div id="chart"></div>
  <div id="legend"></div>
  <div id="prediction-badge" class="pred-hud" style="display:none;"></div>
  <div id="macro-catalyst-badge" class="macro-badge" style="display:none;"></div>
  <div id="forecast-tooltip" class="forecast-tooltip"></div>
  <canvas id="forecast-canvas"></canvas>
  <canvas id="markings-canvas"></canvas>
  <canvas id="draw-canvas"></canvas>

  <div id="draw-banner" class="banner-hidden">
    <span id="banner-text">🖌️ Freehand Mode: Drag anywhere to draw · Click to mark</span>
    <button id="banner-cancel">Done [Esc]</button>
  </div>

  <div id="draw-palette" class="palette-hidden">
    <span class="palette-title">Tool:</span>
    <button class="palette-btn active" data-tool="pen" title="Freehand Pen / Brush (Drag anywhere freely)">🖌️ Pen</button>
    <button class="palette-btn" data-tool="pin" title="Click to drop a Marker Pin / Dot">📍 Pin</button>
    <button class="palette-btn" data-tool="arrow" title="Drag to draw an Arrow">➔ Arrow</button>
    <button class="palette-btn" data-tool="box" title="Drag to draw a Box / Zone">🔲 Box</button>
    <button class="palette-btn" data-tool="level" title="1-Click Horizontal Price Level">➖ Level</button>
    <button class="palette-btn" data-tool="eraser" title="Click any marking to erase it">🧹 Eraser</button>
    <div class="pal-sep"></div>
    <span class="palette-title">Color:</span>
    <button class="color-dot active" data-color="#FFD700" style="background:#FFD700;" title="Gold"></button>
    <button class="color-dot" data-color="#00E676" style="background:#00E676;" title="Green"></button>
    <button class="color-dot" data-color="#FF5252" style="background:#FF5252;" title="Red"></button>
    <button class="color-dot" data-color="#29B6F6" style="background:#29B6F6;" title="Cyan"></button>
    <button class="color-dot" data-color="#FFFFFF" style="background:#FFFFFF;" title="White"></button>
    <div class="pal-sep"></div>
    <span class="palette-title">Size:</span>
    <button class="size-btn active" data-size="2.5">2px</button>
    <button class="size-btn" data-size="4.5">4px</button>
    <button id="btn-close-palette" class="palette-close" title="Done drawing">✕ Done</button>
  </div>

  <div id="markings-modal" class="modal-hidden">
    <div class="modal-header">
      <span>Saved Markings (<b id="modal-symbol"></b>)</span>
      <button id="modal-close" class="modal-close">✕</button>
    </div>
    <div id="markings-list" class="modal-body"></div>
    <div class="modal-footer">
      <span style="font-size:11px;color:var(--muted);">Persists until removed</span>
      <button id="modal-clear-all" class="btn-danger">🗑️ Clear All</button>
    </div>
  </div>

  <div id="draw-toast"></div>

  <div id="ai-overlays-menu" class="menu-hidden">
    <div class="overlay-menu-header">
      <span>⚡ AI Predictive Overlays</span>
      <span style="font-size:9.5px;color:var(--muted);font-weight:normal;">Instant Canvas</span>
    </div>
    <label class="overlay-item"><input type="checkbox" id="ov-swing-wave" checked> 🌊 Harmonic Swing Wave</label>
    <label class="overlay-item"><input type="checkbox" id="ov-corridors" checked> 🎯 Breakout Corridors</label>
    <label class="overlay-item"><input type="checkbox" id="ov-forecast-cone" checked> 🔮 Forecast Quantile Cloud</label>
    <label class="overlay-item"><input type="checkbox" id="ov-turning-points" checked> 🔄 Projected Turning Points</label>
    <label class="overlay-item"><input type="checkbox" id="ov-scenarios" checked> 🔀 Multi-Scenario Paths</label>
    <label class="overlay-item"><input type="checkbox" id="ov-rv-tunnel" checked> 🌊 Forward RV Tunnel</label>
    <label class="overlay-item"><input type="checkbox" id="ov-volume-poc" checked> 🏛️ Volume POC & Profile</label>
    <label class="overlay-item"><input type="checkbox" id="ov-deal-markers" checked> 📍 Buy/Sell Deal Markers</label>
    <div class="overlay-menu-divider"></div>
    <div class="overlay-menu-header">
      <span>📈 Technical Indicators</span>
    </div>
    <label class="overlay-item"><input type="checkbox" id="ov-sma20" checked> 🟡 SMA 20</label>
    <label class="overlay-item"><input type="checkbox" id="ov-sma50" checked> 🟠 SMA 50</label>
    <label class="overlay-item"><input type="checkbox" id="ov-sma200"> 🟣 SMA 200</label>
    <label class="overlay-item"><input type="checkbox" id="ov-bb" checked> ⚪ Bollinger Bands</label>
    <div class="overlay-menu-divider"></div>
    <label class="overlay-item"><input type="checkbox" id="ov-hud" checked> 🔮 Prediction Accuracy HUD</label>
    <label class="overlay-item"><input type="checkbox" id="ov-macro" checked> 🏛️ Macro Catalyst Badge</label>
  </div>

  <div id="tools">
    <div class="tool-group">
      <button id="btn-draw-toggle" title="Freehand & Shape Drawing Tools">✏️ Draw ▾</button>
      <button id="btn-undo" title="Remove last marking (Ctrl+Z)" disabled>⌫ Undo</button>
      <button id="btn-clear" title="Clear all markings" disabled>🗑️ Clear</button>
      <button id="btn-list" title="View & manage all saved markings">📋 (<span id="mark-count">0</span>)</button>
    </div>
    <div class="tool-sep"></div>
    <button id="btn-ai-overlays" title="Toggle AI Forecast Overlays (Instant, zero chart refresh)">⚡ AI Overlays ▾</button>
    <button id="btn-hud" title="Toggle Forecast Prediction HUD">🔮 HUD</button>
    <button id="btn-macro" title="Toggle Macro Catalyst Badge">🏛️ Macro</button>
    <button id="btn-tz" title="Switch between Local timezone and UTC">🕒 Local</button>
    <button id="btn-fit" title="Fit all data">⤢ Fit</button>
    <button id="btn-live" title="Scroll to latest bar">⇥ Live</button>
    <button id="btn-shot" title="Download PNG">📷</button>
    <button id="btn-fs" title="Toggle Fullscreen Mode (or press Esc to exit)">⛶ Fullscreen</button>
  </div>
  <div id="resize-handle" title="Click & Drag up/down to dynamically resize chart height">
    <div class="resize-grip"></div>
  </div>
</div>
<script>__LWC_BUNDLE__</script>
<script>
(function () {
  const SPEC = __SPEC__;
  const LWC = window.LightweightCharts;
  const wrap = document.getElementById('wrap');
  const el = document.getElementById('chart');

  if (!LWC) {
    if (wrap) wrap.innerHTML = '<div id="err">TradingView Lightweight Charts library failed to load.</div>';
    return;
  }

  try {
    const lockX = SPEC.zoomMode === 'lockX', lockY = SPEC.zoomMode === 'lockY';

    // ── Timezone management (IST vs Local vs UTC) ──
    const tzConfigs = {
      'IST': { iana: 'Asia/Kolkata', label: 'IST' },
      'Local': { iana: undefined, label: 'Local' },
      'UTC': { iana: 'UTC', label: 'UTC' },
      'EST': { iana: 'America/New_York', label: 'EST' }
    };
    let currentTz = SPEC.tzMode || 'IST';
    if (!tzConfigs[currentTz]) currentTz = 'IST';

    function getTzConfig() {
      return tzConfigs[currentTz] || tzConfigs['IST'];
    }

    function formatTickTime(time, tickMarkType) {
      const ts = typeof time === 'number' ? time : (time.timestamp || time);
      const d = new Date(ts * 1000);
      const cfg = getTzConfig();
      const tzOpt = cfg.iana ? { timeZone: cfg.iana } : {};

      if (tickMarkType === 3 /* Time */ || tickMarkType === 4 /* TimeWithSeconds */) {
        return d.toLocaleTimeString('en-US', { ...tzOpt, hour: '2-digit', minute: '2-digit', hour12: false });
      }
      if (tickMarkType === 2 /* DayOfMonth */) {
        return d.toLocaleDateString('en-US', { ...tzOpt, day: 'numeric', month: 'short' });
      }
      if (tickMarkType === 1 /* Month */) {
        return d.toLocaleDateString('en-US', { ...tzOpt, month: 'short' });
      }
      if (tickMarkType === 0 /* Year */) {
        return d.toLocaleDateString('en-US', { ...tzOpt, year: 'numeric' });
      }
      return d.toLocaleTimeString('en-US', { ...tzOpt, hour: '2-digit', minute: '2-digit', hour12: false });
    }

    function formatLocalizationTime(time) {
      const ts = typeof time === 'number' ? time : (time.timestamp || time);
      const d = new Date(ts * 1000);
      const cfg = getTzConfig();
      const tzOpt = cfg.iana ? { timeZone: cfg.iana } : {};

      if (SPEC.intraday) {
        const dateStr = d.toLocaleDateString('en-US', { ...tzOpt, month: 'short', day: 'numeric' });
        const timeStr = d.toLocaleTimeString('en-US', { ...tzOpt, hour: '2-digit', minute: '2-digit', hour12: false });
        return `${dateStr}, ${timeStr} (${cfg.label})`;
      }
      return d.toLocaleDateString('en-US', { ...tzOpt, year: 'numeric', month: 'short', day: 'numeric' });
    }

    // ── Universal Adaptive Price Scaling & Precision for All Asset Classes ──
    const sym = (SPEC.filename || '').split('_')[0] || (SPEC.title ? SPEC.title.split(' ')[0] : '');
    const lastCandle = (SPEC.candles && SPEC.candles.length) ? SPEC.candles[SPEC.candles.length - 1] : null;
    const refPrice = lastCandle ? Math.abs(Number(lastCandle.close) || 100) : 100;

    const isForex = /^(EUR|GBP|USD|JPY|AUD|CAD|CHF|NZD|=X)/i.test(sym) || /EURUSD|GBPUSD|USDJPY/i.test(sym) || (refPrice < 2.0 && refPrice > 0.5);
    const isNatGas = /^NG/i.test(sym) || (refPrice < 6.0 && refPrice >= 1.0);
    const isSilver = /^SI/i.test(sym) || /SILVER/i.test(sym) || (refPrice < 50.0 && refPrice >= 15.0 && !/^CL/i.test(sym));
    const isLowDenom = isForex || isNatGas || isSilver || refPrice < 100;

    let mainDecimals = SPEC.pricePrecision != null ? SPEC.pricePrecision : (
      isForex ? 5 :
      (isNatGas ? 4 :
      (isSilver ? 3 :
      (refPrice < 1.0 ? 5 :
      (refPrice < 10.0 ? 4 :
      (refPrice < 50.0 ? 3 : 2)))))
    );
    let mainMinMove = SPEC.priceMinMove != null ? SPEC.priceMinMove : Number(Math.pow(10, -mainDecimals).toFixed(mainDecimals));
    const mainPriceFmt = { type: 'price', precision: mainDecimals, minMove: mainMinMove };

    const topScaleMargin = (SPEC.scaleMargins && SPEC.scaleMargins.top != null) ? SPEC.scaleMargins.top : (isLowDenom ? 0.20 : 0.15);
    const bottomScaleMargin = (SPEC.scaleMargins && SPEC.scaleMargins.bottom != null) ? SPEC.scaleMargins.bottom : (isLowDenom ? 0.18 : 0.12);

    const chart = LWC.createChart(el, {
      autoSize: true,
      layout: { background: { type: LWC.ColorType.Solid, color: '#0D1117' }, textColor: '#B2B5BE',
                fontSize: 11, panes: { separatorColor: 'rgba(255,255,255,0.08)', enableResize: true } },
      grid: { vertLines: { color: 'rgba(255,255,255,0.04)' }, horzLines: { color: 'rgba(255,255,255,0.05)' } },
      crosshair: { mode: LWC.CrosshairMode.Normal,
        vertLine: { color: 'rgba(255,255,255,0.35)', style: 1, labelBackgroundColor: '#2962FF' },
        horzLine: { color: 'rgba(255,255,255,0.35)', style: 1, labelBackgroundColor: '#2962FF' } },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,0.1)',
        scaleMargins: { top: topScaleMargin, bottom: bottomScaleMargin },
        autoScale: true,
      },
      timeScale: {
        borderColor: 'rgba(255,255,255,0.1)',
        timeVisible: SPEC.intraday,
        secondsVisible: false,
        rightOffset: 8,
        barSpacing: 7,
        tickMarkFormatter: (time, tickMarkType) => formatTickTime(time, tickMarkType),
      },
      localization: {
        timeFormatter: (time) => formatLocalizationTime(time),
      },
      handleScroll: lockX ? { mouseWheel: false, pressedMouseMove: false, horzTouchDrag: false, vertTouchDrag: true } : true,
      handleScale: lockX
        ? { mouseWheel: false, pinch: false, axisPressedMouseMove: { time: false, price: true } }
        : { axisPressedMouseMove: { time: true, price: !lockY } },
    });

    // ── Main candlesticks ──
    const candles = chart.addSeries(LWC.CandlestickSeries, {
      upColor: SPEC.up, downColor: SPEC.down, borderUpColor: SPEC.up, borderDownColor: SPEC.down,
      wickUpColor: SPEC.up, wickDownColor: SPEC.down, title: '',
      priceFormat: mainPriceFmt,
    });
    candles.setData(SPEC.candles);

    // ── TP / SL shaded zones ──
    (SPEC.baselines || []).forEach(z => {
      try {
        const s = chart.addSeries(LWC.BaselineSeries, {
          baseValue: { type: 'price', price: z.base },
          topLineColor: z.line, bottomLineColor: z.line, lineWidth: 1, lineStyle: 1,
          topFillColor1: z.fill, topFillColor2: z.fill, bottomFillColor1: z.fill, bottomFillColor2: z.fill,
          lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false,
          priceFormat: mainPriceFmt,
        });
        s.setData(z.data);
      } catch (e) { console.warn('zone', e); }
    });

    // ── Lines (overlays, patterns, forecast, sub-pane indicators) ──
    const legendItems = [];
    const legendSeries = [];
    const forecastLineSeries = [];
    const taggedLineSeries = {};
    (SPEC.lines || []).forEach(l => {
      try {
        const isMainPane = !l.pane || l.pane === 0;
        const s = chart.addSeries(LWC.LineSeries, {
          color: l.color, lineWidth: l.width, lineStyle: l.style, title: l.lastValue ? l.title : '',
          lastValueVisible: !!l.lastValue, priceLineVisible: false,
          crosshairMarkerVisible: !!l.legend, pointMarkersVisible: false,
          priceFormat: isMainPane ? mainPriceFmt : { type: 'price', precision: 2, minMove: 0.01 },
        }, l.pane || 0);
        s.setData(l.data);
        (l.priceLines || []).forEach(p => s.createPriceLine(p));
        if (l.legend) { legendItems.push(l); legendSeries.push(s); }
        if (l.tag === 'forecast' || (l.title && l.title.includes('TimesFM'))) {
          forecastLineSeries.push(s);
        }
        if (l.tag) {
          if (!taggedLineSeries[l.tag]) taggedLineSeries[l.tag] = [];
          taggedLineSeries[l.tag].push(s);
        }
      } catch (e) { console.warn('line', l.title, e); }
    });

    // ── Histograms (volume / MACD hist) ──
    (SPEC.histograms || []).forEach(h => {
      try {
        const s = chart.addSeries(LWC.HistogramSeries, {
          priceFormat: h.volume ? { type: 'volume' } : { type: 'price', precision: 4, minMove: 0.0001 },
          lastValueVisible: true, priceLineVisible: false, title: h.title,
        }, h.pane || 1);
        s.setData(h.data);
      } catch (e) { console.warn('hist', e); }
    });

    // ── Trade price lines (Entry / TP / SL / POC) ──
    const priceLineObjects = [];
    (SPEC.priceLines || []).forEach(p => {
      try {
        const pl = candles.createPriceLine({
          price: p.price, color: p.color, lineWidth: p.width, lineStyle: p.style,
          axisLabelVisible: true, title: p.title,
        });
        priceLineObjects.push({ pl: pl, spec: p, attached: true });
      } catch (e) { console.warn('priceLine', p.title, e); }
    });

    // ── Pattern & AI Trade Deal Markers (Candles & Future Bars) ──
    const allCandleMarkers = [];
    const allFutureMarkers = [];
    if ((SPEC.markers || []).length) {
      try {
        const candleTimeSet = new Set((SPEC.candles || []).map(c => c.time));

        SPEC.markers.forEach(m => {
          if (candleTimeSet.has(m.time)) {
            allCandleMarkers.push(m);
          } else {
            allFutureMarkers.push(m);
          }
        });
      } catch (e) { console.warn('markers error', e); }
    }

    // ── Pane sizing ──
    try { const panes = chart.panes(); if (panes.length > 1) { panes[0].setStretchFactor(3); panes[1].setStretchFactor(1); } } catch (e) {}

    // ── Elements & UI Handles ──
    const legend = document.getElementById('legend');
    const predBadgeEl = document.getElementById('prediction-badge');
    const macroBadgeEl = document.getElementById('macro-catalyst-badge');
    const forecastTooltipEl = document.getElementById('forecast-tooltip');
    const btnHud = document.getElementById('btn-hud');
    const btnMacro = document.getElementById('btn-macro');
    const btnAiOverlays = document.getElementById('btn-ai-overlays');
    const aiOverlaysMenu = document.getElementById('ai-overlays-menu');
    const forecastCanvas = document.getElementById('forecast-canvas');
    const forecastCtx = forecastCanvas ? forecastCanvas.getContext('2d') : null;

    const fmt = v => {
      if (v == null || typeof v === 'undefined' || isNaN(v)) return '—';
      try {
        return Number(v).toLocaleString(undefined, { maximumFractionDigits: mainDecimals, minimumFractionDigits: mainDecimals });
      } catch (e) {
        return String(v);
      }
    };
    const last = SPEC.candles[SPEC.candles.length - 1];

    function drawRoundRect(ctx, x, y, w, h, r) {
      if (typeof ctx.roundRect === 'function') {
        ctx.roundRect(x, y, w, h, r);
      } else {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
      }
    }

    function drawSmoothSpline(ctx, pts, tension) {
      if (!pts || pts.length < 2) return;
      if (pts.length === 2) {
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        ctx.lineTo(pts[1].x, pts[1].y);
        ctx.stroke();
        return;
      }
      const t = tension != null ? tension : 0.45;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);

      for (let i = 0; i < pts.length - 1; i++) {
        const p0 = i > 0 ? pts[i - 1] : pts[0];
        const p1 = pts[i];
        const p2 = pts[i + 1];
        const p3 = i < pts.length - 2 ? pts[i + 2] : p2;

        const cp1x = p1.x + (p2.x - p0.x) * t / 2;
        const cp1y = p1.y + (p2.y - p0.y) * t / 2;
        const cp2x = p2.x - (p3.x - p1.x) * t / 2;
        const cp2y = p2.y - (p3.y - p1.y) * t / 2;

        ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
      }
      ctx.stroke();
    }

    // ── Legend (OHLC on crosshair with Glassmorphism Badges) ──
    function renderLegend(bar, vals) {
      bar = bar || last;
      let html = '<div class="legend-badge">';
      html += `<span class="sym">${SPEC.title}</span>`;
      if (bar) {
        const cls = bar.close >= bar.open ? 'up' : 'dn';
        const chg = bar.open ? ((bar.close - bar.open) / bar.open * 100) : 0;
        let timeBadge = '';
        if (bar.time) {
          const bd = new Date(bar.time * 1000);
          const cfg = getTzConfig();
          const tzOpt = cfg.iana ? { timeZone: cfg.iana } : {};
          const timeStr = SPEC.intraday
            ? bd.toLocaleTimeString('en-US', { ...tzOpt, hour: '2-digit', minute: '2-digit', hour12: false })
            : bd.toLocaleDateString('en-US', { ...tzOpt, year: 'numeric', month: 'short', day: 'numeric' });
          timeBadge = ` <span class="k" style="color:#29B6F6;font-weight:600;">[${timeStr} ${cfg.label}]</span>`;
        }
        html += timeBadge
             + `<span class="k">O</span> <span class="${cls}">${fmt(bar.open)}</span>`
             + `<span class="k">H</span> <span class="${cls}">${fmt(bar.high)}</span>`
             + `<span class="k">L</span> <span class="${cls}">${fmt(bar.low)}</span>`
             + `<span class="k">C</span> <span class="${cls}">${fmt(bar.close)}</span>`
             + ` <span class="${cls}">(${chg >= 0 ? '+' : ''}${chg.toFixed(2)}%)</span>`;
      }
      html += '</div>';

      if (legendItems.length) {
        html += '<div class="items">' + legendItems.map((l, i) => {
          const v = vals ? vals[i] : null;
          return `<div class="item-pill"><span class="dot" style="background:${l.color}"></span><span>${l.title}</span>${v != null ? ' <b>' + fmt(v) + '</b>' : ''}</div>`;
        }).join('') + '</div>';
      }
      legend.innerHTML = html;
    }
    renderLegend();

    // ── Crosshair Move (Historical Bars & Interactive Future Forecast Tooltip) ──
    chart.subscribeCrosshairMove(param => {
      if (!param || !param.time) {
        if (forecastTooltipEl) forecastTooltipEl.classList.remove('show');
        renderLegend();
        return;
      }

      // Check if hovering over a future forecast bar
      const tKey = typeof param.time === 'number' ? param.time : (param.time.timestamp || param.time);
      const ftData = (SPEC.forecastTooltips && (SPEC.forecastTooltips[tKey] || SPEC.forecastTooltips[String(tKey)])) || null;

      if (ftData && forecastTooltipEl) {
        const isUp = ftData.direction === 'BULLISH' || (ftData.change_pct != null && ftData.change_pct >= 0);
        const dirCls = isUp ? 'up' : 'dn';
        const dirText = isUp ? '▲ BULLISH' : '▼ BEARISH';
        const chgSign = (ftData.change_pct != null && ftData.change_pct >= 0) ? '+' : '';
        const chgStr = ftData.change_pct != null ? `${chgSign}${Number(ftData.change_pct).toFixed(2)}%` : '';

        let rowsHtml = '';
        if (ftData.expected_price != null || ftData.q50 != null) {
          const expVal = ftData.expected_price != null ? ftData.expected_price : ftData.q50;
          rowsHtml += `<div class="ft-row"><span class="ft-label">Expected (q50):</span> <span class="ft-val bold ${dirCls}">${fmt(expVal)} <span style="font-size:10px;">(${chgStr})</span></span></div>`;
        }
        if (ftData.q10 != null && ftData.q90 != null) {
          rowsHtml += `<div class="ft-row"><span class="ft-label">80% Interval (q10–q90):</span> <span class="ft-val">${fmt(ftData.q10)} – ${fmt(ftData.q90)}</span></div>`;
        }
        if (ftData.q30 != null && ftData.q70 != null) {
          rowsHtml += `<div class="ft-row"><span class="ft-label">40% Core (q30–q70):</span> <span class="ft-val">${fmt(ftData.q30)} – ${fmt(ftData.q70)}</span></div>`;
        }
        if (ftData.confidence != null) {
          rowsHtml += `<div class="ft-row"><span class="ft-label">Model Confidence:</span> <span class="ft-val highlight">${(ftData.confidence * 100).toFixed(1)}%</span></div>`;
        }
        if (ftData.accuracy != null) {
          rowsHtml += `<div class="ft-row"><span class="ft-label">Model Accuracy:</span> <span class="ft-val">${(ftData.accuracy * 100).toFixed(1)}%</span></div>`;
        }

        forecastTooltipEl.innerHTML = `
          <div class="ft-head">
            <span class="ft-title">🔮 Forecast Step +${ftData.step || 1} <span style="font-weight:400;font-size:10.5px;color:#8B949E;">(${ftData.horizon_label || ''})</span></span>
            <span class="ft-dir ${dirCls}">${ftData.direction || dirText}</span>
          </div>
          <div class="ft-body">${rowsHtml}</div>
        `;

        if (param.point) {
          const wrapRect = wrap.getBoundingClientRect();
          let left = param.point.x + 18;
          let top = param.point.y - 36;
          if (left + 240 > wrapRect.width) left = Math.max(10, param.point.x - 245);
          if (top < 10) top = 10;
          if (top + 170 > wrapRect.height) top = Math.max(10, wrapRect.height - 180);
          forecastTooltipEl.style.left = left + 'px';
          forecastTooltipEl.style.top = top + 'px';
        }
        forecastTooltipEl.classList.add('show');

        // Update legend for forecast bar
        const expVal = ftData.expected_price != null ? ftData.expected_price : ftData.q50;
        let fcLegend = `<div class="legend-badge"><span class="sym">${SPEC.title}</span> `
          + `<span class="k" style="color:#38BDF8;font-weight:700;">[🔮 FORECAST +${ftData.step || 1}]</span> `
          + `<span class="k">Exp:</span> <span class="${dirCls}">${fmt(expVal)}</span> `
          + `<span class="${dirCls}">(${chgStr})</span>`;
        if (ftData.q10 != null && ftData.q90 != null) {
          fcLegend += ` <span class="k">80% CI:</span> <span>[${fmt(ftData.q10)} – ${fmt(ftData.q90)}]</span>`;
        }
        fcLegend += `</div>`;
        legend.innerHTML = fcLegend;
        return;
      }

      // Normal candle crosshair
      if (forecastTooltipEl) forecastTooltipEl.classList.remove('show');
      const bar = param.seriesData.get(candles);
      const vals = legendSeries.map(s => { const d = param.seriesData.get(s); return d ? d.value : null; });
      renderLegend(bar, vals);
    });

    // ── Refresh-proof zoom: persist the visible range per (symbol, interval, generation) ──
    const KEY = 'lwc_range::' + SPEC.stateKey;
    const store = (() => { try { return window.sessionStorage; } catch (e) { return null; } })();
    const ts = chart.timeScale();
    let restored = false;
    try {
      const saved = store && store.getItem(KEY);
      if (saved) { ts.setVisibleLogicalRange(JSON.parse(saved)); restored = true; }
    } catch (e) {}
    if (!restored) ts.fitContent();
    let saveTimer = null;
    ts.subscribeVisibleLogicalRangeChange(r => {
      if (!r || !store) return;
      clearTimeout(saveTimer);
      saveTimer = setTimeout(() => { try { store.setItem(KEY, JSON.stringify({ from: r.from, to: r.to })); } catch (e) {} }, 150);
    });

    // ── Active Overlay Visibility State Engine (Zero-Refresh Dynamic Control) ──
    const DEFAULT_OVERLAYS = {
      swingWave: true,
      corridors: true,
      forecastBands: true,
      turningPoints: true,
      scenarios: true,
      rvTunnel: true,
      volumeProfile: true,
      dealMarkers: true,
      hud: true,
      macro: true,
      sma20: true,
      sma50: true,
      sma200: false,
      bb: true,
    };

    const activeOverlays = Object.assign({}, DEFAULT_OVERLAYS, SPEC.defaultOverlays || {});

    // Check if lwc_synced_overlays exists in localStorage (fresh from Streamlit rerun)
    try {
      const synced = store && store.getItem('lwc_synced_overlays');
      if (synced) {
        const parsed = JSON.parse(synced);
        if (parsed && parsed.overlays && (Date.now() - parsed.ts < 5000)) {
          Object.assign(activeOverlays, parsed.overlays);
        }
      }
    } catch (e) {}

    // Restore saved user overlay toggles if present for this chart session
    const OVERLAY_STORE_KEY = 'lwc_overlays::' + SPEC.stateKey;
    try {
      const savedOverlays = store && store.getItem(OVERLAY_STORE_KEY);
      if (savedOverlays) {
        Object.assign(activeOverlays, JSON.parse(savedOverlays));
      }
    } catch (e) {}

    function saveOverlayState() {
      try {
        if (store) store.setItem(OVERLAY_STORE_KEY, JSON.stringify(activeOverlays));
      } catch (e) {}
    }

    function updateBadgesVisibility() {
      if (predBadgeEl) {
        predBadgeEl.style.display = (activeOverlays.hud !== false && SPEC.predictionBadge) ? 'flex' : 'none';
        if (btnHud) btnHud.classList.toggle('active', activeOverlays.hud !== false);
      }
      if (macroBadgeEl) {
        macroBadgeEl.style.display = (activeOverlays.macro !== false && SPEC.macroCatalyst) ? 'flex' : 'none';
        if (btnMacro) btnMacro.classList.toggle('active', activeOverlays.macro !== false);
      }
    }

    function updateMarkers() {
      try {
        const showMarkers = activeOverlays.dealMarkers !== false;
        const currentCandleMarkers = showMarkers ? allCandleMarkers : [];
        const currentFutureMarkers = showMarkers ? allFutureMarkers : [];

        if (typeof candles.setMarkers === 'function') {
          candles.setMarkers(currentCandleMarkers);
        } else if (typeof LWC !== 'undefined' && typeof LWC.createSeriesMarkers === 'function') {
          LWC.createSeriesMarkers(candles, currentCandleMarkers);
        }

        if (legendSeries.length) {
          const targetSeries = legendSeries[0];
          if (typeof targetSeries.setMarkers === 'function') {
            targetSeries.setMarkers(currentFutureMarkers);
          } else if (typeof LWC !== 'undefined' && typeof LWC.createSeriesMarkers === 'function') {
            LWC.createSeriesMarkers(targetSeries, currentFutureMarkers);
          }
        }
      } catch (e) {
        console.warn('updateMarkers error', e);
      }
    }

    function updateForecastLines() {
      const visible = activeOverlays.forecastBands !== false;
      forecastLineSeries.forEach(s => {
        try { s.applyOptions({ visible }); } catch (e) {}
      });
    }

    function updateIndicatorLines() {
      ['sma20', 'sma50', 'sma200', 'bb'].forEach(tag => {
        const seriesList = taggedLineSeries[tag] || [];
        const visible = activeOverlays[tag] !== false;
        seriesList.forEach(s => {
          try { s.applyOptions({ visible }); } catch (e) {}
        });
      });
    }

    function updatePriceLines() {
      priceLineObjects.forEach(item => {
        const tag = item.spec.tag;
        if (!tag) return;
        const shouldShow = activeOverlays[tag] !== false;
        if (shouldShow && !item.attached) {
          try {
            item.pl = candles.createPriceLine({
              price: item.spec.price, color: item.spec.color, lineWidth: item.spec.width,
              lineStyle: item.spec.style, axisLabelVisible: true, title: item.spec.title,
            });
            item.attached = true;
          } catch(e) {}
        } else if (!shouldShow && item.attached) {
          try {
            candles.removePriceLine(item.pl);
            item.attached = false;
          } catch(e) {}
        }
      });
    }

    const overlayMapping = {
      'ov-swing-wave': 'swingWave',
      'ov-corridors': 'corridors',
      'ov-forecast-cone': 'forecastBands',
      'ov-turning-points': 'turningPoints',
      'ov-scenarios': 'scenarios',
      'ov-rv-tunnel': 'rvTunnel',
      'ov-volume-poc': 'volumeProfile',
      'ov-deal-markers': 'dealMarkers',
      'ov-sma20': 'sma20',
      'ov-sma50': 'sma50',
      'ov-sma200': 'sma200',
      'ov-bb': 'bb',
      'ov-hud': 'hud',
      'ov-macro': 'macro',
    };

    function syncOverlayMenuCheckboxes() {
      Object.entries(overlayMapping).forEach(([elId, key]) => {
        const chk = document.getElementById(elId);
        if (chk) chk.checked = activeOverlays[key] !== false;
      });
    }

    function applyOverlaySettings(newOverlays) {
      if (newOverlays) {
        Object.assign(activeOverlays, newOverlays);
      }
      syncOverlayMenuCheckboxes();
      updateBadgesVisibility();
      updateMarkers();
      updateForecastLines();
      updateIndicatorLines();
      updatePriceLines();
      drawAllForecastVisuals();
      saveOverlayState();
    }

    // Attach checkbox event listeners
    Object.entries(overlayMapping).forEach(([elId, key]) => {
      const chk = document.getElementById(elId);
      if (chk) {
        chk.onchange = (e) => {
          activeOverlays[key] = !!e.target.checked;
          if (key === 'hud' || key === 'macro') {
            updateBadgesVisibility();
          } else if (key === 'dealMarkers') {
            updateMarkers();
          } else if (key === 'forecastBands') {
            updateForecastLines();
            drawAllForecastVisuals();
          } else if (key === 'sma20' || key === 'sma50' || key === 'sma200' || key === 'bb') {
            updateIndicatorLines();
          } else if (key === 'volumeProfile') {
            updatePriceLines();
            drawAllForecastVisuals();
          } else {
            drawAllForecastVisuals();
          }
          saveOverlayState();
        };
      }
    });

    // Dropdown open/close
    if (btnAiOverlays && aiOverlaysMenu) {
      btnAiOverlays.onclick = (e) => {
        e.stopPropagation();
        aiOverlaysMenu.classList.toggle('menu-hidden');
        btnAiOverlays.classList.toggle('active', !aiOverlaysMenu.classList.contains('menu-hidden'));
      };
      document.addEventListener('click', (e) => {
        if (!aiOverlaysMenu.contains(e.target) && e.target !== btnAiOverlays) {
          aiOverlaysMenu.classList.add('menu-hidden');
          btnAiOverlays.classList.remove('active');
        }
      });
    }

    // Window message listener for external sync from Streamlit without iframe reload
    window.addEventListener('message', function (e) {
      try {
        if (e.data && e.data.type === 'SET_OVERLAYS' && e.data.overlays) {
          applyOverlaySettings(e.data.overlays);
        }
      } catch (err) {
        console.warn('postMessage listener error', err);
      }
    });

    // Cross-iframe storage event listener
    window.addEventListener('storage', function(e) {
      if (e.key === 'lwc_synced_overlays' && e.newValue) {
        try {
          const data = JSON.parse(e.newValue);
          if (data && data.overlays) {
            applyOverlaySettings(data.overlays);
          }
        } catch(err) {}
      }
    });

    // Initial sync of indicators and badges
    syncOverlayMenuCheckboxes();
    updateBadgesVisibility();
    updateMarkers();
    updateForecastLines();
    updateIndicatorLines();
    updatePriceLines();

    // ── Time to X Coordinate Mapper ──
    function timeToX(t) {
      if (t == null) return null;
      const tsVal = typeof t === 'number' ? t : (t.timestamp || t);
      let x = ts.timeToCoordinate(tsVal);
      if (x !== null && x !== undefined && !isNaN(x)) return x;

      if (SPEC.candles && SPEC.candles.length > 0) {
        const lastBar = SPEC.candles[SPEC.candles.length - 1];
        const prevBar = SPEC.candles.length > 1 ? SPEC.candles[SPEC.candles.length - 2] : null;
        const barInterval = prevBar ? Math.max(1, lastBar.time - prevBar.time) : (SPEC.intraday ? 60 : 86400);
        const lastLogical = SPEC.candles.length - 1;
        const diffBars = (tsVal - lastBar.time) / barInterval;
        const lx = ts.logicalToCoordinate(lastLogical + diffBars);
        if (lx !== null && lx !== undefined && !isNaN(lx)) return lx;
      }
      return null;
    }

    // ── PREDICTIVE ANALYTICS CANVAS RENDERING ENGINE ──
    function drawAllForecastVisuals() {
      if (!forecastCanvas || !forecastCtx) return;
      try {
        forecastCtx.clearRect(0, 0, forecastCanvas.width, forecastCanvas.height);

      // A. Multi-Horizon Shaded Confidence Bands (Clouds / Cones)
      if (activeOverlays.forecastBands !== false) {
      (SPEC.forecastBands || []).forEach(band => {
        try {
          if (!band.upper || !band.lower || band.upper.length < 2) return;
          const upperPts = [];
          const lowerPts = [];

          band.upper.forEach(pt => {
            const x = timeToX(pt.time);
            const y = candles.priceToCoordinate(pt.value);
            if (x !== null && y !== null && !isNaN(x) && !isNaN(y)) {
              upperPts.push({ x, y, time: pt.time, val: pt.value });
            }
          });

          band.lower.forEach(pt => {
            const x = timeToX(pt.time);
            const y = candles.priceToCoordinate(pt.value);
            if (x !== null && y !== null && !isNaN(x) && !isNaN(y)) {
              lowerPts.push({ x, y, time: pt.time, val: pt.value });
            }
          });

          if (upperPts.length < 2 || lowerPts.length < 2) return;

          // 1. Shaded Polygon Fill
          forecastCtx.save();
          forecastCtx.beginPath();
          forecastCtx.moveTo(upperPts[0].x, upperPts[0].y);
          for (let i = 1; i < upperPts.length; i++) {
            forecastCtx.lineTo(upperPts[i].x, upperPts[i].y);
          }
          for (let i = lowerPts.length - 1; i >= 0; i--) {
            forecastCtx.lineTo(lowerPts[i].x, lowerPts[i].y);
          }
          forecastCtx.closePath();

          if (band.gradient && upperPts.length > 0) {
            const minY = Math.min(...upperPts.map(p => p.y), ...lowerPts.map(p => p.y));
            const maxY = Math.max(...upperPts.map(p => p.y), ...lowerPts.map(p => p.y));
            const grad = forecastCtx.createLinearGradient(0, minY, 0, maxY);
            grad.addColorStop(0, band.fillColor || 'rgba(56, 189, 248, 0.12)');
            grad.addColorStop(0.5, band.fillColor || 'rgba(56, 189, 248, 0.20)');
            grad.addColorStop(1, band.fillColor || 'rgba(56, 189, 248, 0.12)');
            forecastCtx.fillStyle = grad;
          } else {
            forecastCtx.fillStyle = band.fillColor || 'rgba(56, 189, 248, 0.14)';
          }
          forecastCtx.fill();

          // 2. Boundary Lines
          if (band.lineWidth > 0 && band.lineColor) {
            forecastCtx.strokeStyle = band.lineColor;
            forecastCtx.lineWidth = band.lineWidth;
            if (band.lineStyle === 1) forecastCtx.setLineDash([3, 3]);
            else if (band.lineStyle === 2) forecastCtx.setLineDash([6, 4]);
            else forecastCtx.setLineDash([]);

            // Upper
            forecastCtx.beginPath();
            upperPts.forEach((p, idx) => {
              if (idx === 0) forecastCtx.moveTo(p.x, p.y);
              else forecastCtx.lineTo(p.x, p.y);
            });
            forecastCtx.stroke();

            // Lower
            forecastCtx.beginPath();
            lowerPts.forEach((p, idx) => {
              if (idx === 0) forecastCtx.moveTo(p.x, p.y);
              else forecastCtx.lineTo(p.x, p.y);
            });
            forecastCtx.stroke();
            forecastCtx.setLineDash([]);
          }

          // 3. Band Label at right end
          const lastUpper = upperPts[upperPts.length - 1];
          const lastLower = lowerPts[lowerPts.length - 1];
          if (lastUpper && lastUpper.x < forecastCanvas.width - 60) {
            forecastCtx.font = 'bold 9px sans-serif';
            forecastCtx.fillStyle = band.lineColor || '#38BDF8';
            forecastCtx.fillText(band.name ? band.name.split(' ')[0] : 'CI', lastUpper.x + 4, (lastUpper.y + lastLower.y) / 2);
          }

          forecastCtx.restore();
        } catch (e) {
          console.warn('forecastBand error', e);
        }
      });
      }

      // B. Enhanced Risk-Reward Breakout Corridors
      if (activeOverlays.corridors !== false) {
      (SPEC.breakoutCorridors || []).forEach(corr => {
        try {
          const xStart = timeToX(corr.startTime);
          const xEnd = timeToX(corr.endTime);
          const yEntry = candles.priceToCoordinate(corr.entry);
          const yTP = candles.priceToCoordinate(corr.takeProfit);
          const ySL = candles.priceToCoordinate(corr.stopLoss);

          if (xStart === null || xEnd === null || yEntry === null || yTP === null || ySL === null) return;
          if (isNaN(xStart) || isNaN(xEnd) || isNaN(yEntry) || isNaN(yTP) || isNaN(ySL)) return;

          const left = Math.min(xStart, xEnd);
          const right = Math.max(xStart, xEnd);
          const width = Math.max(20, right - left);

          forecastCtx.save();

          // 1. Take Profit Zone Fill
          const tpTop = Math.min(yEntry, yTP);
          const tpH = Math.abs(yTP - yEntry);
          const tpGrad = forecastCtx.createLinearGradient(0, tpTop, 0, tpTop + tpH);
          tpGrad.addColorStop(0, corr.tpFill || 'rgba(0, 230, 118, 0.16)');
          tpGrad.addColorStop(1, 'rgba(0, 230, 118, 0.04)');
          forecastCtx.fillStyle = tpGrad;
          forecastCtx.fillRect(left, tpTop, width, tpH);

          // 2. Stop Loss Zone Fill
          const slTop = Math.min(yEntry, ySL);
          const slH = Math.abs(ySL - yEntry);
          const slGrad = forecastCtx.createLinearGradient(0, slTop, 0, slTop + slH);
          slGrad.addColorStop(0, 'rgba(255, 82, 82, 0.04)');
          slGrad.addColorStop(1, corr.slFill || 'rgba(255, 82, 82, 0.16)');
          forecastCtx.fillStyle = slGrad;
          forecastCtx.fillRect(left, slTop, width, slH);

          // 3. Boundary Rails
          forecastCtx.setLineDash([5, 4]);
          forecastCtx.strokeStyle = corr.tpBorder || '#00E676';
          forecastCtx.lineWidth = 1.5;
          forecastCtx.beginPath();
          forecastCtx.moveTo(left, yTP);
          forecastCtx.lineTo(right, yTP);
          forecastCtx.stroke();

          forecastCtx.strokeStyle = corr.slBorder || '#FF5252';
          forecastCtx.lineWidth = 1.5;
          forecastCtx.beginPath();
          forecastCtx.moveTo(left, ySL);
          forecastCtx.lineTo(right, ySL);
          forecastCtx.stroke();

          forecastCtx.strokeStyle = corr.entryColor || '#29B6F6';
          forecastCtx.lineWidth = 1.2;
          forecastCtx.beginPath();
          forecastCtx.moveTo(left, yEntry);
          forecastCtx.lineTo(right, yEntry);
          forecastCtx.stroke();
          forecastCtx.setLineDash([]);

          // 4. Central Risk/Reward Badge Pill
          if (corr.showRRPill !== false && width > 60) {
            const midX = (left + right) / 2;
            const pillY = yEntry - 12;
            const rrText = `⚖️ R:R ${Number(corr.rrRatio).toFixed(2)} : 1  (+${Number(corr.tpPct).toFixed(1)}% / -${Number(corr.slPct).toFixed(1)}%)` +
                           (corr.winProb ? ` · Win ${Math.round(corr.winProb * 100)}%` : '');
            forecastCtx.font = 'bold 10px Inter, sans-serif';
            const tw = forecastCtx.measureText(rrText).width;
            const pillW = tw + 16;
            const pillX = midX - pillW / 2;

            forecastCtx.fillStyle = 'rgba(19, 23, 34, 0.90)';
            forecastCtx.strokeStyle = 'rgba(41, 182, 246, 0.6)';
            forecastCtx.lineWidth = 1;
            drawRoundRect(forecastCtx, pillX, pillY - 10, pillW, 20, 5);
            forecastCtx.fill();
            forecastCtx.stroke();

            forecastCtx.fillStyle = '#FFFFFF';
            forecastCtx.fillText(rrText, pillX + 8, pillY + 4);
          }

          // 5. Zone Labels at right edge
          if (right < forecastCanvas.width - 20) {
            forecastCtx.font = 'bold 9.5px sans-serif';
            forecastCtx.fillStyle = corr.tpBorder || '#00E676';
            forecastCtx.fillText(corr.tpLabel || '🎯 TP', right + 4, yTP + 3);
            forecastCtx.fillStyle = corr.slBorder || '#FF5252';
            forecastCtx.fillText(corr.slLabel || '🛑 SL', right + 4, ySL + 3);
          }

          forecastCtx.restore();
        } catch (e) {
          console.warn('breakoutCorridor error', e);
        }
      });
      }

      // C. Directional Projections with Glowing Trajectory & Terminal Arrows
      if (activeOverlays.forecastBands !== false) {
      (SPEC.directionalProjections || []).forEach(proj => {
        try {
          if (!proj.points || proj.points.length < 2) return;
          const pts = [];
          proj.points.forEach(p => {
            const x = timeToX(p.time);
            const y = candles.priceToCoordinate(p.value);
            if (x !== null && y !== null && !isNaN(x) && !isNaN(y)) {
              pts.push({ x, y, time: p.time, val: p.value });
            }
          });
          if (pts.length < 2) return;

          forecastCtx.save();

          // Glowing Path
          if (proj.glow !== false) {
            forecastCtx.shadowColor = proj.glowColor || proj.strokeColor || '#00E676';
            forecastCtx.shadowBlur = 9;
          }
          forecastCtx.strokeStyle = proj.strokeColor || '#00E676';
          forecastCtx.lineWidth = proj.width || 2.4;
          forecastCtx.lineCap = 'round';
          forecastCtx.lineJoin = 'round';

          forecastCtx.beginPath();
          pts.forEach((p, idx) => {
            if (idx === 0) forecastCtx.moveTo(p.x, p.y);
            else forecastCtx.lineTo(p.x, p.y);
          });
          forecastCtx.stroke();
          forecastCtx.restore();

          // Terminal Arrowhead
          const last = pts[pts.length - 1];
          const prev = pts[pts.length - 2];
          const angle = Math.atan2(last.y - prev.y, last.x - prev.x);

          forecastCtx.save();
          forecastCtx.fillStyle = proj.strokeColor || '#00E676';
          forecastCtx.beginPath();
          forecastCtx.moveTo(last.x, last.y);
          forecastCtx.lineTo(last.x - 11 * Math.cos(angle - Math.PI / 6), last.y - 11 * Math.sin(angle - Math.PI / 6));
          forecastCtx.lineTo(last.x - 11 * Math.cos(angle + Math.PI / 6), last.y - 11 * Math.sin(angle + Math.PI / 6));
          forecastCtx.closePath();
          forecastCtx.fill();

          // Target Pill
          if (last.x < forecastCanvas.width - 120) {
            const diffPct = proj.diffPct != null ? proj.diffPct : 0;
            const tgtText = `🎯 ${fmt(last.val)} (${diffPct >= 0 ? '+' : ''}${diffPct.toFixed(2)}%)`;
            forecastCtx.font = 'bold 10px sans-serif';
            const tw = forecastCtx.measureText(tgtText).width;
            const pillW = tw + 12;
            const pillX = last.x + 12;
            const pillY = last.y - 10;

            forecastCtx.fillStyle = 'rgba(19, 23, 34, 0.90)';
            forecastCtx.strokeStyle = proj.strokeColor || '#00E676';
            forecastCtx.lineWidth = 1;
            drawRoundRect(forecastCtx, pillX, pillY, pillW, 20, 4);
            forecastCtx.fill();
            forecastCtx.stroke();

            forecastCtx.fillStyle = '#FFFFFF';
            forecastCtx.fillText(tgtText, pillX + 6, pillY + 14);
          }
          forecastCtx.restore();
        } catch (e) {
          console.warn('directionalProjection error', e);
        }
      });
      }

      // D. Forward Realized Volatility (RV) Envelope (1-sigma & 2-sigma Channels)
      if (activeOverlays.rvTunnel !== false) {
      (SPEC.volatilityEnvelopes || []).forEach(env => {
        try {
          if (!env.upper2s || !env.lower2s || env.upper2s.length < 2) return;
          const u2Pts = [], l2Pts = [], u1Pts = [], l1Pts = [];

          env.upper2s.forEach(p => {
            const x = timeToX(p.time), y = candles.priceToCoordinate(p.value);
            if (x !== null && y !== null && !isNaN(x) && !isNaN(y)) u2Pts.push({ x, y, time: p.time, val: p.value });
          });
          env.lower2s.forEach(p => {
            const x = timeToX(p.time), y = candles.priceToCoordinate(p.value);
            if (x !== null && y !== null && !isNaN(x) && !isNaN(y)) l2Pts.push({ x, y, time: p.time, val: p.value });
          });
          (env.upper1s || []).forEach(p => {
            const x = timeToX(p.time), y = candles.priceToCoordinate(p.value);
            if (x !== null && y !== null && !isNaN(x) && !isNaN(y)) u1Pts.push({ x, y, time: p.time, val: p.value });
          });
          (env.lower1s || []).forEach(p => {
            const x = timeToX(p.time), y = candles.priceToCoordinate(p.value);
            if (x !== null && y !== null && !isNaN(x) && !isNaN(y)) l1Pts.push({ x, y, time: p.time, val: p.value });
          });

          if (u2Pts.length < 2 || l2Pts.length < 2) return;

          forecastCtx.save();

          // 1. Outer 2-sigma shaded polygon
          forecastCtx.beginPath();
          forecastCtx.moveTo(u2Pts[0].x, u2Pts[0].y);
          for (let i = 1; i < u2Pts.length; i++) forecastCtx.lineTo(u2Pts[i].x, u2Pts[i].y);
          for (let i = l2Pts.length - 1; i >= 0; i--) forecastCtx.lineTo(l2Pts[i].x, l2Pts[i].y);
          forecastCtx.closePath();
          forecastCtx.fillStyle = env.color2s || 'rgba(168, 85, 247, 0.08)';
          forecastCtx.fill();

          // 2. Inner 1-sigma shaded polygon (if present)
          if (u1Pts.length >= 2 && l1Pts.length >= 2) {
            forecastCtx.beginPath();
            forecastCtx.moveTo(u1Pts[0].x, u1Pts[0].y);
            for (let i = 1; i < u1Pts.length; i++) forecastCtx.lineTo(u1Pts[i].x, u1Pts[i].y);
            for (let i = l1Pts.length - 1; i >= 0; i--) forecastCtx.lineTo(l1Pts[i].x, l1Pts[i].y);
            forecastCtx.closePath();
            forecastCtx.fillStyle = env.color1s || 'rgba(168, 85, 247, 0.16)';
            forecastCtx.fill();
          }

          // 3. Boundary Rails
          forecastCtx.strokeStyle = env.borderColor || 'rgba(168, 85, 247, 0.55)';
          forecastCtx.lineWidth = 1.0;
          forecastCtx.setLineDash([4, 4]);

          // Upper 2s
          forecastCtx.beginPath();
          u2Pts.forEach((p, idx) => (idx === 0 ? forecastCtx.moveTo(p.x, p.y) : forecastCtx.lineTo(p.x, p.y)));
          forecastCtx.stroke();

          // Lower 2s
          forecastCtx.beginPath();
          l2Pts.forEach((p, idx) => (idx === 0 ? forecastCtx.moveTo(p.x, p.y) : forecastCtx.lineTo(p.x, p.y)));
          forecastCtx.stroke();
          forecastCtx.setLineDash([]);

          // 4. Edge Labels
          const lastU2 = u2Pts[u2Pts.length - 1];
          const lastL2 = l2Pts[l2Pts.length - 1];
          if (lastU2 && lastU2.x < forecastCanvas.width - 55) {
            forecastCtx.font = 'bold 9px sans-serif';
            forecastCtx.fillStyle = env.borderColor || '#A855F7';
            forecastCtx.fillText('+2σ RV', lastU2.x + 4, lastU2.y - 2);
            forecastCtx.fillText('-2σ RV', lastL2.x + 4, lastL2.y + 10);
          }

          forecastCtx.restore();
        } catch (e) {
          console.warn('volatilityEnvelope error', e);
        }
      });
      }

      // E. Multi-Scenario Forecast Trajectories (Bull / Base / Bear)
      if (activeOverlays.scenarios !== false) {
      (SPEC.forecastScenarios || []).forEach(sc => {
        try {
          if (!sc.points || sc.points.length < 2) return;
          const pts = [];
          sc.points.forEach(p => {
            const x = timeToX(p.time), y = candles.priceToCoordinate(p.value);
            if (x !== null && y !== null && !isNaN(x) && !isNaN(y)) pts.push({ x, y, time: p.time, val: p.value });
          });
          if (pts.length < 2) return;

          forecastCtx.save();
          forecastCtx.shadowColor = sc.color || '#38BDF8';
          forecastCtx.shadowBlur = 6;
          forecastCtx.strokeStyle = sc.color || '#38BDF8';
          forecastCtx.lineWidth = sc.width || 2.0;

          if (sc.style === 1) forecastCtx.setLineDash([3, 3]);
          else if (sc.style === 2) forecastCtx.setLineDash([6, 4]);
          else forecastCtx.setLineDash([]);

          forecastCtx.beginPath();
          pts.forEach((p, idx) => (idx === 0 ? forecastCtx.moveTo(p.x, p.y) : forecastCtx.lineTo(p.x, p.y)));
          forecastCtx.stroke();
          forecastCtx.setLineDash([]);
          forecastCtx.restore();

          // Scenario pill tag at end
          const last = pts[pts.length - 1];
          if (last && last.x < forecastCanvas.width - 130) {
            forecastCtx.save();
            const tagText = `${sc.title || sc.name}: ${fmt(last.val)}`;
            forecastCtx.font = 'bold 9.5px sans-serif';
            const tw = forecastCtx.measureText(tagText).width;
            const pw = tw + 10, px = last.x + 10, py = last.y - 9;

            forecastCtx.fillStyle = 'rgba(19, 23, 34, 0.88)';
            forecastCtx.strokeStyle = sc.color || '#38BDF8';
            forecastCtx.lineWidth = 1;
            drawRoundRect(forecastCtx, px, py, pw, 18, 4);
            forecastCtx.fill();
            forecastCtx.stroke();

            forecastCtx.fillStyle = '#FFFFFF';
            forecastCtx.fillText(tagText, px + 5, py + 12);
            forecastCtx.restore();
          }
        } catch (e) {
          console.warn('forecastScenario error', e);
        }
      });
      }

      // F. Turning Point & Reversal Target Markers (Peaks & Troughs)
      if (activeOverlays.turningPoints !== false) {
      (SPEC.turningPoints || []).forEach(tp => {
        try {
          const x = timeToX(tp.time);
          const y = candles.priceToCoordinate(tp.price);
          if (x === null || y === null || isNaN(x) || isNaN(y)) return;

          const isPeak = tp.kind === 'peak';
          const tpCol = tp.color || (isPeak ? '#FFD700' : '#E040FB');

          forecastCtx.save();

          // Connecting guide stem line
          const stemOffset = isPeak ? -22 : 22;
          forecastCtx.strokeStyle = tpCol;
          forecastCtx.lineWidth = 1.2;
          forecastCtx.setLineDash([2, 2]);
          forecastCtx.beginPath();
          forecastCtx.moveTo(x, y);
          forecastCtx.lineTo(x, y + stemOffset);
          forecastCtx.stroke();
          forecastCtx.setLineDash([]);

          // Glowing pulse dot at exact coordinate
          forecastCtx.fillStyle = tpCol;
          forecastCtx.shadowColor = tpCol;
          forecastCtx.shadowBlur = 8;
          forecastCtx.beginPath();
          forecastCtx.arc(x, y, 4, 0, Math.PI * 2);
          forecastCtx.fill();
          forecastCtx.strokeStyle = '#FFFFFF';
          forecastCtx.lineWidth = 1.2;
          forecastCtx.stroke();

          // Banner badge pill
          forecastCtx.shadowBlur = 0;
          forecastCtx.font = 'bold 10px Inter, sans-serif';
          const txt = tp.label || (isPeak ? `▲ Peak ${fmt(tp.price)}` : `▼ Trough ${fmt(tp.price)}`);
          const tw = forecastCtx.measureText(txt).width;
          const pw = tw + 14;
          const px = Math.max(10, Math.min(forecastCanvas.width - pw - 10, x - pw / 2));
          const py = isPeak ? (y - 32) : (y + 16);

          forecastCtx.fillStyle = 'rgba(19, 23, 34, 0.92)';
          forecastCtx.strokeStyle = tpCol;
          forecastCtx.lineWidth = 1.5;
          drawRoundRect(forecastCtx, px, py, pw, 20, 5);
          forecastCtx.fill();
          forecastCtx.stroke();

          forecastCtx.fillStyle = '#FFFFFF';
          forecastCtx.fillText(txt, px + 7, py + 14);

          forecastCtx.restore();
        } catch (e) {
          console.warn('turningPoint error', e);
        }
      });
      }

      // G. Harmonic Swing Waves with Glowing Bezier Splines & Amplitude Crest/Trough Tags
      if (activeOverlays.swingWave !== false) {
      (SPEC.swingWaves || []).forEach(wave => {
        try {
          if (!wave.points || wave.points.length < 2) return;
          const pts = [];
          wave.points.forEach(p => {
            const x = timeToX(p.time);
            const y = candles.priceToCoordinate(p.value);
            if (x !== null && y !== null && !isNaN(x) && !isNaN(y)) {
              pts.push({ x, y, time: p.time, val: p.value });
            }
          });
          if (pts.length < 2) return;

          forecastCtx.save();

          // 1. Dual-phase neon gradient (cyan/lime bullish, coral/magenta bearish)
          const minX = pts[0].x;
          const maxX = pts[pts.length - 1].x;
          const crestCol = wave.crestColor || '#00E5FF';
          const troughCol = wave.troughColor || '#FF3D71';

          const grad = forecastCtx.createLinearGradient(minX, 0, maxX, 0);
          grad.addColorStop(0, crestCol);
          grad.addColorStop(0.5, '#7C4DFF');
          grad.addColorStop(1, troughCol);

          forecastCtx.strokeStyle = grad;
          forecastCtx.lineWidth = wave.width || 2.8;
          forecastCtx.lineCap = 'round';
          forecastCtx.lineJoin = 'round';
          forecastCtx.shadowColor = crestCol;
          forecastCtx.shadowBlur = 10;

          if (wave.smooth !== false) {
            drawSmoothSpline(forecastCtx, pts, 0.45);
          } else {
            forecastCtx.beginPath();
            pts.forEach((p, idx) => (idx === 0 ? forecastCtx.moveTo(p.x, p.y) : forecastCtx.lineTo(p.x, p.y)));
            forecastCtx.stroke();
          }
          forecastCtx.restore();

          // 2. Visual swing amplitude tags (crests and troughs)
          if (wave.showCrestTags !== false) {
            // Crests (Peaks)
            (wave.crests || []).forEach(c => {
              const cx = timeToX(c.time);
              const cy = candles.priceToCoordinate(c.price);
              if (cx === null || cy === null || isNaN(cx) || isNaN(cy)) return;

              forecastCtx.save();
              // Glowing node
              forecastCtx.fillStyle = crestCol;
              forecastCtx.shadowColor = crestCol;
              forecastCtx.shadowBlur = 8;
              forecastCtx.beginPath();
              forecastCtx.arc(cx, cy, 4, 0, Math.PI * 2);
              forecastCtx.fill();
              forecastCtx.strokeStyle = '#FFFFFF';
              forecastCtx.lineWidth = 1.2;
              forecastCtx.stroke();
              forecastCtx.shadowBlur = 0;

              // Vertical indicator stem
              forecastCtx.strokeStyle = crestCol;
              forecastCtx.lineWidth = 1.2;
              forecastCtx.setLineDash([2, 2]);
              forecastCtx.beginPath();
              forecastCtx.moveTo(cx, cy);
              forecastCtx.lineTo(cx, cy - 20);
              forecastCtx.stroke();
              forecastCtx.setLineDash([]);

              // Tag text & pill
              const crestText = c.label || `▲ +${fmt(c.diff)} (${c.pct ? c.pct.toFixed(2) : '0.00'}%) Crest`;
              forecastCtx.font = 'bold 9.5px Inter, sans-serif';
              const tw = forecastCtx.measureText(crestText).width;
              const pw = tw + 14;
              const px = Math.max(10, Math.min(forecastCanvas.width - pw - 10, cx - pw / 2));
              const py = cy - 38;

              forecastCtx.fillStyle = 'rgba(13, 17, 23, 0.92)';
              forecastCtx.strokeStyle = crestCol;
              forecastCtx.lineWidth = 1.2;
              drawRoundRect(forecastCtx, px, py, pw, 18, 4);
              forecastCtx.fill();
              forecastCtx.stroke();

              forecastCtx.fillStyle = '#FFFFFF';
              forecastCtx.fillText(crestText, px + 7, py + 12);
              forecastCtx.restore();
            });

            // Troughs (Dips)
            (wave.troughs || []).forEach(tr => {
              const tx = timeToX(tr.time);
              const ty = candles.priceToCoordinate(tr.price);
              if (tx === null || ty === null || isNaN(tx) || isNaN(ty)) return;

              forecastCtx.save();
              // Glowing node
              forecastCtx.fillStyle = troughCol;
              forecastCtx.shadowColor = troughCol;
              forecastCtx.shadowBlur = 8;
              forecastCtx.beginPath();
              forecastCtx.arc(tx, ty, 4, 0, Math.PI * 2);
              forecastCtx.fill();
              forecastCtx.strokeStyle = '#FFFFFF';
              forecastCtx.lineWidth = 1.2;
              forecastCtx.stroke();
              forecastCtx.shadowBlur = 0;

              // Vertical indicator stem
              forecastCtx.strokeStyle = troughCol;
              forecastCtx.lineWidth = 1.2;
              forecastCtx.setLineDash([2, 2]);
              forecastCtx.beginPath();
              forecastCtx.moveTo(tx, ty);
              forecastCtx.lineTo(tx, ty + 20);
              forecastCtx.stroke();
              forecastCtx.setLineDash([]);

              // Tag text & pill
              const trText = tr.label || `▼ ${fmt(tr.diff)} (${tr.pct ? tr.pct.toFixed(2) : '0.00'}%) Dip`;
              forecastCtx.font = 'bold 9.5px Inter, sans-serif';
              const tw = forecastCtx.measureText(trText).width;
              const pw = tw + 14;
              const px = Math.max(10, Math.min(forecastCanvas.width - pw - 10, tx - pw / 2));
              const py = ty + 22;

              forecastCtx.fillStyle = 'rgba(13, 17, 23, 0.92)';
              forecastCtx.strokeStyle = troughCol;
              forecastCtx.lineWidth = 1.2;
              drawRoundRect(forecastCtx, px, py, pw, 18, 4);
              forecastCtx.fill();
              forecastCtx.stroke();

              forecastCtx.fillStyle = '#FFFFFF';
              forecastCtx.fillText(trText, px + 7, py + 12);
              forecastCtx.restore();
            });
          }
        } catch (e) {
          console.warn('swingWave error', e);
        }
      });
      }

      // H. Translucent Volume Profile (POC / VAH / VAL)
      if (activeOverlays.volumeProfile !== false && SPEC.volumeProfile && SPEC.volumeProfile.bins && SPEC.volumeProfile.bins.length > 0) {
        try {
          const vp = SPEC.volumeProfile;
          const bins = vp.bins;
          const maxVol = Math.max(...bins.map(b => b.volume || 0), 1);
          const alignRight = vp.align !== 'left';
          const maxBarWidth = Math.min(180, forecastCanvas.width * 0.22);
          const baseOpacity = vp.opacity != null ? vp.opacity : 0.28;
          const anchorX = alignRight ? (forecastCanvas.width - 65) : 10;

          forecastCtx.save();
          const halfBinHeight = Math.max(2, (forecastCanvas.height / bins.length) * 0.45);

          bins.forEach(b => {
            const y = candles.priceToCoordinate(b.price);
            if (y === null || isNaN(y)) return;
            const barW = (b.volume / maxVol) * maxBarWidth;
            if (barW < 1) return;

            const x = alignRight ? (anchorX - barW) : anchorX;
            const buyFraction = b.volume > 0 ? ((b.buy_vol || (b.volume * 0.5)) / b.volume) : 0.5;
            const buyW = barW * buyFraction;
            const sellW = barW - buyW;

            // Buy volume bar
            forecastCtx.fillStyle = `rgba(0, 230, 118, ${baseOpacity})`;
            forecastCtx.fillRect(x, y - halfBinHeight, buyW, halfBinHeight * 2);

            // Sell volume bar
            forecastCtx.fillStyle = `rgba(255, 61, 113, ${baseOpacity * 0.9})`;
            forecastCtx.fillRect(x + buyW, y - halfBinHeight, sellW, halfBinHeight * 2);

            // Subtle border
            forecastCtx.strokeStyle = `rgba(255, 255, 255, ${baseOpacity * 0.4})`;
            forecastCtx.lineWidth = 0.5;
            forecastCtx.strokeRect(x, y - halfBinHeight, barW, halfBinHeight * 2);
          });

          // Highlight Point of Control (POC)
          if (vp.poc != null) {
            const pocY = candles.priceToCoordinate(vp.poc);
            if (pocY !== null && !isNaN(pocY)) {
              forecastCtx.strokeStyle = '#FFD700';
              forecastCtx.lineWidth = 1.6;
              forecastCtx.setLineDash([4, 3]);
              forecastCtx.beginPath();
              forecastCtx.moveTo(alignRight ? (anchorX - maxBarWidth - 10) : anchorX, pocY);
              forecastCtx.lineTo(alignRight ? anchorX : (anchorX + maxBarWidth + 10), pocY);
              forecastCtx.stroke();
              forecastCtx.setLineDash([]);

              if (vp.showLabels !== false) {
                forecastCtx.font = 'bold 9.5px Inter, sans-serif';
                forecastCtx.fillStyle = '#FFD700';
                forecastCtx.fillText(`POC ${fmt(vp.poc)}`, alignRight ? (anchorX - maxBarWidth - 62) : (anchorX + maxBarWidth + 14), pocY + 3);
              }
            }
          }

          // Highlight Value Area (VAH and VAL)
          if (vp.vah != null) {
            const vahY = candles.priceToCoordinate(vp.vah);
            if (vahY !== null && !isNaN(vahY)) {
              forecastCtx.strokeStyle = 'rgba(56, 189, 248, 0.7)';
              forecastCtx.lineWidth = 1.0;
              forecastCtx.setLineDash([3, 3]);
              forecastCtx.beginPath();
              forecastCtx.moveTo(alignRight ? (anchorX - maxBarWidth * 0.7) : anchorX, vahY);
              forecastCtx.lineTo(alignRight ? anchorX : (anchorX + maxBarWidth * 0.7), vahY);
              forecastCtx.stroke();
              forecastCtx.setLineDash([]);
              if (vp.showLabels !== false) {
                forecastCtx.font = '8.5px Inter, sans-serif';
                forecastCtx.fillStyle = 'rgba(56, 189, 248, 0.85)';
                forecastCtx.fillText(`VAH ${fmt(vp.vah)}`, alignRight ? (anchorX - maxBarWidth * 0.7 - 48) : (anchorX + maxBarWidth * 0.7 + 6), vahY + 3);
              }
            }
          }

          if (vp.val != null) {
            const valY = candles.priceToCoordinate(vp.val);
            if (valY !== null && !isNaN(valY)) {
              forecastCtx.strokeStyle = 'rgba(56, 189, 248, 0.7)';
              forecastCtx.lineWidth = 1.0;
              forecastCtx.setLineDash([3, 3]);
              forecastCtx.beginPath();
              forecastCtx.moveTo(alignRight ? (anchorX - maxBarWidth * 0.7) : anchorX, valY);
              forecastCtx.lineTo(alignRight ? anchorX : (anchorX + maxBarWidth * 0.7), valY);
              forecastCtx.stroke();
              forecastCtx.setLineDash([]);
              if (vp.showLabels !== false) {
                forecastCtx.font = '8.5px Inter, sans-serif';
                forecastCtx.fillStyle = 'rgba(56, 189, 248, 0.85)';
                forecastCtx.fillText(`VAL ${fmt(vp.val)}`, alignRight ? (anchorX - maxBarWidth * 0.7 - 48) : (anchorX + maxBarWidth * 0.7 + 6), valY + 3);
              }
            }
          }

          forecastCtx.restore();
        } catch (e) {
          console.warn('volumeProfile error', e);
        }
      }
      } catch (err) {
        console.warn('drawAllForecastVisuals top-level error', err);
      }
    }

    // ── Setup Prediction Accuracy HUD Card ──
    if (SPEC.predictionBadge && predBadgeEl) {
      const b = SPEC.predictionBadge;
      const isUp = b.direction === 'BULLISH';
      const isDn = b.direction === 'BEARISH';
      const badgeCls = isUp ? 'up' : (isDn ? 'dn' : 'nt');
      const badgeSym = isUp ? '▲ BULLISH' : (isDn ? '▼ BEARISH' : '◆ NEUTRAL');

      let statsHtml = '';
      if (b.confidence != null) {
        statsHtml += `<div class="pred-hud-stat"><span class="k">Confidence:</span> <span class="v highlight">${(b.confidence * 100).toFixed(1)}%</span></div>`;
      }
      if (b.accuracy != null) {
        statsHtml += `<div class="pred-hud-stat"><span class="k">Accuracy:</span> <span class="v">${(b.accuracy * 100).toFixed(1)}%</span></div>`;
      }
      if (b.horizon_bars != null) {
        statsHtml += `<div class="pred-hud-stat"><span class="k">Horizon:</span> <span class="v">${b.horizon_bars} bars</span></div>`;
      }
      if (b.win_rate != null) {
        statsHtml += `<div class="pred-hud-stat"><span class="k">Win Rate:</span> <span class="v">${(b.win_rate * 100).toFixed(1)}%</span></div>`;
      }
      if (b.brier_score != null) {
        statsHtml += `<div class="pred-hud-stat"><span class="k">Brier:</span> <span class="v">${Number(b.brier_score).toFixed(3)}</span></div>`;
      }

      predBadgeEl.innerHTML = `
        <div class="pred-hud-header">
          <div class="pred-hud-title">
            <span>🔮</span>
            <span>${b.model_name || 'Forecast Engine'}</span>
            ${b.direction ? `<span class="pred-hud-badge ${badgeCls}">${badgeSym}</span>` : ''}
          </div>
          <button id="pred-hud-toggle" class="pred-hud-toggle" title="Collapse / Expand">_</button>
        </div>
        <div class="pred-hud-body">${statsHtml}</div>
      `;
      predBadgeEl.style.display = 'flex';

      const hudToggle = document.getElementById('pred-hud-toggle');
      if (hudToggle) {
        hudToggle.onclick = () => {
          predBadgeEl.classList.toggle('collapsed');
          hudToggle.innerText = predBadgeEl.classList.contains('collapsed') ? '+' : '_';
        };
      }

      if (btnHud) {
        btnHud.classList.toggle('active', activeOverlays.hud !== false);
        btnHud.onclick = () => {
          activeOverlays.hud = !activeOverlays.hud;
          updateBadgesVisibility();
          syncOverlayMenuCheckboxes();
          saveOverlayState();
        };
      }
    } else {
      if (predBadgeEl) predBadgeEl.style.display = 'none';
      if (btnHud) btnHud.style.display = 'none';
    }

    // ── Setup Macro Catalyst & Conviction Badge ──
    if (SPEC.macroCatalyst && macroBadgeEl) {
      const mc = SPEC.macroCatalyst;
      let driverText = mc.primary_driver || 'None';
      let convText = mc.conviction || 'High';
      macroBadgeEl.innerHTML = `
        <span class="macro-badge-icon">🏛️</span>
        <span>Catalyst: <span class="macro-badge-driver">${driverText}</span></span>
        <span class="macro-badge-conv">${convText} Conviction</span>
      `;
      macroBadgeEl.style.display = 'flex';
      if (btnMacro) {
        btnMacro.classList.toggle('active', activeOverlays.macro !== false);
        btnMacro.onclick = () => {
          activeOverlays.macro = !activeOverlays.macro;
          updateBadgesVisibility();
          syncOverlayMenuCheckboxes();
          saveOverlayState();
        };
      }
    } else {
      if (macroBadgeEl) macroBadgeEl.style.display = 'none';
      if (btnMacro) btnMacro.style.display = 'none';
    }

    // ── UNRESTRICTED MARKINGS & DRAWING ENGINE ──
    const storageKey = 'tv_markings_' + (sym || 'SYMBOL');
    const modalSym = document.getElementById('modal-symbol');
    if (modalSym) modalSym.innerText = (sym || 'SYMBOL');

    let activeTool = null; // 'pen' | 'pin' | 'arrow' | 'box' | 'level' | 'eraser' | null
    let currentColor = '#FFD700';
    let currentWidth = 2.5;
    let isDrawing = false;
    let currentStroke = [];
    let startPoint = null;

    const markingsCanvas = document.getElementById('markings-canvas');
    const markingsCtx = markingsCanvas ? markingsCanvas.getContext('2d') : null;
    const drawCanvas = document.getElementById('draw-canvas');
    const drawCtx = drawCanvas ? drawCanvas.getContext('2d') : null;
    const drawBanner = document.getElementById('draw-banner');
    const bannerText = document.getElementById('banner-text');
    const drawPalette = document.getElementById('draw-palette');
    const markingsModal = document.getElementById('markings-modal');
    const markCountSpan = document.getElementById('mark-count');
    const btnUndo = document.getElementById('btn-undo');
    const btnClear = document.getElementById('btn-clear');
    const drawToast = document.getElementById('draw-toast');

    const nativeLevelObjects = {};

    function resizeCanvases() {
      const w = el.clientWidth, h = el.clientHeight;
      if (forecastCanvas) { forecastCanvas.width = w; forecastCanvas.height = h; }
      if (markingsCanvas) { markingsCanvas.width = w; markingsCanvas.height = h; }
      if (drawCanvas) { drawCanvas.width = w; drawCanvas.height = h; }
      drawAllForecastVisuals();
      drawAllMarkings();
    }
    resizeCanvases();
    window.addEventListener('resize', resizeCanvases);

    function getMetrics() {
      const x0 = ts.logicalToCoordinate(0) ?? 0;
      const x1 = ts.logicalToCoordinate(1) ?? (x0 + 10);
      const barSpacing = x1 - x0 || 10;
      return { x0, barSpacing };
    }

    function xToFl(x) {
      const m = getMetrics();
      return (x - m.x0) / m.barSpacing;
    }

    function flToX(fl) {
      const m = getMetrics();
      return m.x0 + fl * m.barSpacing;
    }

    function showToast(msg) {
      if (!drawToast) return;
      drawToast.innerText = msg;
      drawToast.classList.add('show');
      setTimeout(() => drawToast.classList.remove('show'), 2200);
    }

    function getSavedMarkings() {
      try {
        const raw = localStorage.getItem(storageKey);
        return raw ? JSON.parse(raw) : [];
      } catch (e) {
        return [];
      }
    }

    function saveMarkingsList(list) {
      try {
        localStorage.setItem(storageKey, JSON.stringify(list));
      } catch (e) {
        console.warn('Failed to save markings to localStorage', e);
      }
      updateMarkingsUI();
      drawAllMarkings();
    }

    function drawAllMarkings() {
      if (!markingsCanvas || !markingsCtx) return;
      markingsCtx.clearRect(0, 0, markingsCanvas.width, markingsCanvas.height);
      const list = getSavedMarkings();

      list.forEach(m => {
        if (m.type === 'freehand') {
          if (!m.points || m.points.length < 2) return;
          markingsCtx.strokeStyle = m.color || '#FFD700';
          markingsCtx.lineWidth = m.width || 2.5;
          markingsCtx.lineCap = 'round';
          markingsCtx.lineJoin = 'round';
          markingsCtx.beginPath();
          m.points.forEach((pt, idx) => {
            const x = flToX(pt.fl);
            const y = candles.priceToCoordinate(pt.p);
            if (idx === 0) markingsCtx.moveTo(x, y);
            else markingsCtx.lineTo(x, y);
          });
          markingsCtx.stroke();
        } else if (m.type === 'pin' || m.type === 'dot') {
          const x = flToX(m.fl);
          const y = candles.priceToCoordinate(m.p);
          markingsCtx.fillStyle = m.color || '#FFD700';
          markingsCtx.beginPath();
          markingsCtx.arc(x, y, m.radius || 6, 0, Math.PI * 2);
          markingsCtx.fill();
          markingsCtx.strokeStyle = '#FFFFFF';
          markingsCtx.lineWidth = 1.5;
          markingsCtx.stroke();
          if (m.label) {
            markingsCtx.fillStyle = '#FFFFFF';
            markingsCtx.font = 'bold 11px sans-serif';
            markingsCtx.fillText(m.label, x + 8, y + 4);
          }
        } else if (m.type === 'arrow') {
          const x1 = flToX(m.p1.fl), y1 = candles.priceToCoordinate(m.p1.p);
          const x2 = flToX(m.p2.fl), y2 = candles.priceToCoordinate(m.p2.p);
          markingsCtx.strokeStyle = m.color || '#FFD700';
          markingsCtx.fillStyle = m.color || '#FFD700';
          markingsCtx.lineWidth = m.width || 2.5;
          markingsCtx.lineCap = 'round';
          markingsCtx.beginPath();
          markingsCtx.moveTo(x1, y1);
          markingsCtx.lineTo(x2, y2);
          markingsCtx.stroke();
          const angle = Math.atan2(y2 - y1, x2 - x1);
          markingsCtx.beginPath();
          markingsCtx.moveTo(x2, y2);
          markingsCtx.lineTo(x2 - 12 * Math.cos(angle - Math.PI / 6), y2 - 12 * Math.sin(angle - Math.PI / 6));
          markingsCtx.lineTo(x2 - 12 * Math.cos(angle + Math.PI / 6), y2 - 12 * Math.sin(angle + Math.PI / 6));
          markingsCtx.closePath();
          markingsCtx.fill();
        } else if (m.type === 'box') {
          const x1 = flToX(m.p1.fl), y1 = candles.priceToCoordinate(m.p1.p);
          const x2 = flToX(m.p2.fl), y2 = candles.priceToCoordinate(m.p2.p);
          const minX = Math.min(x1, x2), maxX = Math.max(x1, x2);
          const minY = Math.min(y1, y2), maxY = Math.max(y1, y2);
          markingsCtx.strokeStyle = m.color || '#FFD700';
          markingsCtx.lineWidth = m.width || 1.5;
          markingsCtx.fillStyle = (m.color || '#FFD700') + '24';
          markingsCtx.fillRect(minX, minY, maxX - minX, maxY - minY);
          markingsCtx.strokeRect(minX, minY, maxX - minX, maxY - minY);
        } else if (m.type === 'level' || m.type === 'support' || m.type === 'resistance' || m.type === 'target') {
          if (!nativeLevelObjects[m.id]) {
            try {
              const pl = candles.createPriceLine({
                price: m.price,
                color: m.color || '#FFD700',
                lineWidth: m.width || 2,
                lineStyle: m.lineStyle ?? LWC.LineStyle.Dashed,
                axisLabelVisible: true,
                title: m.title || 'Level',
              });
              nativeLevelObjects[m.id] = pl;
            } catch (e) {}
          }
        }
      });
    }

    ts.subscribeVisibleLogicalRangeChange(() => {
      drawAllForecastVisuals();
      drawAllMarkings();
    });
    ts.subscribeVisibleTimeRangeChange(() => {
      drawAllForecastVisuals();
      drawAllMarkings();
    });

    function updateMarkingsUI() {
      const list = getSavedMarkings();
      if (markCountSpan) markCountSpan.innerText = list.length;
      if (btnUndo) btnUndo.disabled = list.length === 0;
      if (btnClear) btnClear.disabled = list.length === 0;
      renderModalList(list);
    }

    function renderModalList(list) {
      list = list || getSavedMarkings();
      const container = document.getElementById('markings-list');
      if (!container) return;
      if (!list.length) {
        container.innerHTML = '<div class="modal-empty">No saved markings.<br>Click <b>✏️ Draw</b> to draw anywhere freely on the chart.</div>';
        return;
      }
      container.innerHTML = list.map(m => {
        let desc = m.title || m.type.toUpperCase();
        if (m.type === 'freehand') desc = `🖌️ Stroke (${m.points ? m.points.length : 0} pts)`;
        else if (m.type === 'pin') desc = `📍 Pin at ${fmt(m.p)}`;
        else if (m.type === 'arrow') desc = `➔ Arrow: ${fmt(m.p1.p)} → ${fmt(m.p2.p)}`;
        else if (m.type === 'box') desc = `🔲 Zone Box: ${fmt(m.p1.p)} – ${fmt(m.p2.p)}`;
        else if (m.price) desc = `➖ Level: ${fmt(m.price)}`;
        return `
          <div class="mark-item">
            <span class="mark-badge" style="color:${m.color || '#FFD700'}">● ${desc}</span>
            <button class="mark-del" data-id="${m.id}" title="Remove this marking">🗑️</button>
          </div>
        `;
      }).join('');
      container.querySelectorAll('.mark-del').forEach(b => {
        b.onclick = () => {
          removeMarking(b.dataset.id);
        };
      });
    }

    function addMarking(m) {
      const list = getSavedMarkings();
      list.push(m);
      saveMarkingsList(list);
      showToast('✓ ' + (m.title || m.type) + ' saved');
    }

    function removeMarking(mId) {
      let list = getSavedMarkings();
      const target = list.find(m => m.id === mId);
      list = list.filter(m => m.id !== mId);
      if (nativeLevelObjects[mId]) {
        try { candles.removePriceLine(nativeLevelObjects[mId]); } catch (e) {}
        delete nativeLevelObjects[mId];
      }
      saveMarkingsList(list);
      showToast('Removed ' + (target ? (target.title || target.type) : 'marking'));
    }

    function undoLastMarking() {
      const list = getSavedMarkings();
      if (!list.length) return;
      const last = list.pop();
      if (nativeLevelObjects[last.id]) {
        try { candles.removePriceLine(nativeLevelObjects[last.id]); } catch (e) {}
        delete nativeLevelObjects[last.id];
      }
      saveMarkingsList(list);
      showToast('⌫ Removed ' + (last.title || last.type));
    }

    function clearAllMarkings() {
      const list = getSavedMarkings();
      if (!list.length) return;
      Object.keys(nativeLevelObjects).forEach(id => {
        try { candles.removePriceLine(nativeLevelObjects[id]); } catch (e) {}
      });
      saveMarkingsList([]);
      showToast('🗑️ Cleared all markings');
    }

    function setTool(tool) {
      activeTool = tool;
      startPoint = null;
      currentStroke = [];
      isDrawing = false;
      if (drawCtx) drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);

      if (drawCanvas) {
        drawCanvas.classList.add('active');
        drawCanvas.classList.toggle('eraser-mode', tool === 'eraser');
      }
      if (drawBanner) drawBanner.classList.remove('banner-hidden');
      document.querySelectorAll('.palette-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.tool === tool);
      });

      if (bannerText) {
        if (tool === 'pen') bannerText.innerText = '🖌️ Freehand: Drag anywhere to draw · Click to place dot';
        else if (tool === 'pin') bannerText.innerText = '📍 Click anywhere to drop a marker pin';
        else if (tool === 'arrow') bannerText.innerText = '➔ Drag from start to end to draw an arrow';
        else if (tool === 'box') bannerText.innerText = '🔲 Drag to draw a shaded box / zone';
        else if (tool === 'level') bannerText.innerText = '➖ Click anywhere to place a horizontal price level';
        else if (tool === 'eraser') bannerText.innerText = '🧹 Click or drag over any marking to erase it';
      }
    }

    function closeDrawingMode() {
      activeTool = null;
      isDrawing = false;
      currentStroke = [];
      startPoint = null;
      if (drawCanvas) {
        drawCanvas.classList.remove('active');
        drawCanvas.classList.remove('eraser-mode');
        if (drawCtx) drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
      }
      if (drawBanner) drawBanner.classList.add('banner-hidden');
      if (drawPalette) drawPalette.classList.add('palette-hidden');
      const btnToggle = document.getElementById('btn-draw-toggle');
      if (btnToggle) btnToggle.classList.remove('active');
    }

    function openDrawingMode() {
      if (drawPalette) drawPalette.classList.remove('palette-hidden');
      const btnToggle = document.getElementById('btn-draw-toggle');
      if (btnToggle) btnToggle.classList.add('active');
      if (markingsModal) markingsModal.classList.add('modal-hidden');
      setTool('pen');
    }

    function isNearMarking(m, x, y) {
      const threshold = 18;
      if (m.type === 'freehand' && m.points) {
        return m.points.some(pt => {
          const px = flToX(pt.fl), py = candles.priceToCoordinate(pt.p);
          return Math.hypot(px - x, py - y) < threshold;
        });
      } else if (m.type === 'pin') {
        const px = flToX(m.fl), py = candles.priceToCoordinate(m.p);
        return Math.hypot(px - x, py - y) < threshold;
      } else if (m.type === 'arrow' || m.type === 'box') {
        const x1 = flToX(m.p1.fl), y1 = candles.priceToCoordinate(m.p1.p);
        const x2 = flToX(m.p2.fl), y2 = candles.priceToCoordinate(m.p2.p);
        const minX = Math.min(x1, x2) - threshold, maxX = Math.max(x1, x2) + threshold;
        const minY = Math.min(y1, y2) - threshold, maxY = Math.max(y1, y2) + threshold;
        return x >= minX && x <= maxX && y >= minY && y <= maxY;
      } else if (m.type === 'level' || m.price) {
        const py = candles.priceToCoordinate(m.price);
        return Math.abs(py - y) < 8;
      }
      return false;
    }

    if (drawCanvas && drawCtx) {
      function getCanvasPos(e) {
        const rect = drawCanvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return { x: clientX - rect.left, y: clientY - rect.top };
      }

      drawCanvas.addEventListener('mousedown', e => {
        if (!activeTool) return;
        const { x, y } = getCanvasPos(e);
        const price = candles.coordinateToPrice(y);
        if (price == null || isNaN(price)) return;

        if (activeTool === 'eraser') {
          const list = getSavedMarkings();
          const hit = list.find(m => isNearMarking(m, x, y));
          if (hit) removeMarking(hit.id);
          return;
        }

        isDrawing = true;
        startPoint = { x, y, fl: xToFl(x), p: price };
        if (activeTool === 'pen') {
          currentStroke = [{ x, y, fl: xToFl(x), p: price }];
          drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
          drawCtx.strokeStyle = currentColor;
          drawCtx.lineWidth = currentWidth;
          drawCtx.lineCap = 'round';
          drawCtx.lineJoin = 'round';
          drawCtx.beginPath();
          drawCtx.moveTo(x, y);
        }
      });

      drawCanvas.addEventListener('mousemove', e => {
        if (!activeTool) return;
        const { x, y } = getCanvasPos(e);
        const price = candles.coordinateToPrice(y);
        if (price == null || isNaN(price)) return;

        if (activeTool === 'eraser' && isDrawing) {
          const list = getSavedMarkings();
          const hit = list.find(m => isNearMarking(m, x, y));
          if (hit) removeMarking(hit.id);
          return;
        }

        if (activeTool === 'pen' && isDrawing) {
          currentStroke.push({ x, y, fl: xToFl(x), p: price });
          drawCtx.lineTo(x, y);
          drawCtx.stroke();
        } else if (activeTool === 'arrow' && isDrawing && startPoint) {
          drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
          drawCtx.strokeStyle = currentColor;
          drawCtx.fillStyle = currentColor;
          drawCtx.lineWidth = currentWidth;
          drawCtx.beginPath();
          drawCtx.moveTo(startPoint.x, startPoint.y);
          drawCtx.lineTo(x, y);
          drawCtx.stroke();
          const angle = Math.atan2(y - startPoint.y, x - startPoint.x);
          drawCtx.beginPath();
          drawCtx.moveTo(x, y);
          drawCtx.lineTo(x - 12 * Math.cos(angle - Math.PI / 6), y - 12 * Math.sin(angle - Math.PI / 6));
          drawCtx.lineTo(x - 12 * Math.cos(angle + Math.PI / 6), y - 12 * Math.sin(angle + Math.PI / 6));
          drawCtx.closePath();
          drawCtx.fill();
        } else if (activeTool === 'box' && isDrawing && startPoint) {
          drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
          const minX = Math.min(startPoint.x, x), maxX = Math.max(startPoint.x, x);
          const minY = Math.min(startPoint.y, y), maxY = Math.max(startPoint.y, y);
          drawCtx.strokeStyle = currentColor;
          drawCtx.lineWidth = currentWidth;
          drawCtx.fillStyle = currentColor + '28';
          drawCtx.fillRect(minX, minY, maxX - minX, maxY - minY);
          drawCtx.strokeRect(minX, minY, maxX - minX, maxY - minY);
        } else if (activeTool === 'level' && !isDrawing) {
          drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
          drawCtx.strokeStyle = currentColor;
          drawCtx.lineWidth = 1.5;
          drawCtx.setLineDash([4, 4]);
          drawCtx.beginPath();
          drawCtx.moveTo(0, y);
          drawCtx.lineTo(drawCanvas.width, y);
          drawCtx.stroke();
          drawCtx.setLineDash([]);
          const tag = 'Level ' + fmt(price);
          drawCtx.font = '11px sans-serif';
          const tw = drawCtx.measureText(tag).width;
          const tagX = drawCanvas.width - tw - 90;
          drawCtx.fillStyle = currentColor;
          drawCtx.beginPath();
          drawCtx.roundRect(tagX - 4, y - 10, tw + 8, 20, 3);
          drawCtx.fill();
          drawCtx.fillStyle = '#000';
          drawCtx.fillText(tag, tagX, y + 4);
        }
      });

      drawCanvas.addEventListener('mouseup', e => {
        if (!activeTool || !isDrawing) return;
        isDrawing = false;
        const { x, y } = getCanvasPos(e);
        const price = candles.coordinateToPrice(y);
        if (price == null || isNaN(price)) return;

        if (activeTool === 'pen') {
          if (currentStroke.length >= 2) {
            addMarking({
              id: 'm_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4),
              type: 'freehand',
              points: currentStroke.map(pt => ({ fl: Number(pt.fl.toFixed(4)), p: Number(pt.p.toFixed(2)) })),
              color: currentColor,
              width: currentWidth,
              title: 'Freehand Stroke',
              createdAt: Date.now()
            });
          } else {
            addMarking({
              id: 'm_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4),
              type: 'pin',
              fl: Number(xToFl(x).toFixed(4)),
              p: Number(price.toFixed(2)),
              color: currentColor,
              radius: currentWidth === 4.5 ? 8 : 6,
              title: 'Marker Pin',
              createdAt: Date.now()
            });
          }
          currentStroke = [];
          drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
        } else if (activeTool === 'arrow' && startPoint) {
          if (Math.hypot(x - startPoint.x, y - startPoint.y) > 8) {
            addMarking({
              id: 'm_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4),
              type: 'arrow',
              p1: { fl: Number(startPoint.fl.toFixed(4)), p: Number(startPoint.p.toFixed(2)) },
              p2: { fl: Number(xToFl(x).toFixed(4)), p: Number(price.toFixed(2)) },
              color: currentColor,
              width: currentWidth,
              title: 'Arrow',
              createdAt: Date.now()
            });
          }
          startPoint = null;
          drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
        } else if (activeTool === 'box' && startPoint) {
          if (Math.hypot(x - startPoint.x, y - startPoint.y) > 8) {
            addMarking({
              id: 'm_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4),
              type: 'box',
              p1: { fl: Number(startPoint.fl.toFixed(4)), p: Number(startPoint.p.toFixed(2)) },
              p2: { fl: Number(xToFl(x).toFixed(4)), p: Number(price.toFixed(2)) },
              color: currentColor,
              width: currentWidth,
              title: 'Zone Box',
              createdAt: Date.now()
            });
          }
          startPoint = null;
          drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
        }
      });

      drawCanvas.addEventListener('click', e => {
        if (!activeTool) return;
        const { x, y } = getCanvasPos(e);
        const price = candles.coordinateToPrice(y);
        if (price == null || isNaN(price)) return;

        if (activeTool === 'pin') {
          addMarking({
            id: 'm_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4),
            type: 'pin',
            fl: Number(xToFl(x).toFixed(4)),
            p: Number(price.toFixed(2)),
            color: currentColor,
            radius: currentWidth === 4.5 ? 8 : 6,
            title: 'Marker Pin',
            createdAt: Date.now()
          });
        } else if (activeTool === 'level') {
          addMarking({
            id: 'm_' + Date.now() + '_' + Math.random().toString(36).substr(2, 4),
            type: 'level',
            price: Number(price.toFixed(price < 1 ? 6 : (price < 100 ? 4 : 2))),
            color: currentColor,
            width: currentWidth,
            lineStyle: 2,
            title: 'Level',
            createdAt: Date.now()
          });
        }
      });
    }

    const btnDrawToggle = document.getElementById('btn-draw-toggle');
    if (btnDrawToggle) {
      btnDrawToggle.onclick = () => {
        if (drawPalette.classList.contains('palette-hidden')) {
          openDrawingMode();
        } else {
          closeDrawingMode();
        }
      };
    }

    const btnClosePal = document.getElementById('btn-close-palette');
    if (btnClosePal) btnClosePal.onclick = () => closeDrawingMode();
    const bannerCancel = document.getElementById('banner-cancel');
    if (bannerCancel) bannerCancel.onclick = () => closeDrawingMode();

    document.querySelectorAll('.palette-btn').forEach(b => {
      b.onclick = () => setTool(b.dataset.tool);
    });

    document.querySelectorAll('.color-dot').forEach(d => {
      d.onclick = () => {
        currentColor = d.dataset.color;
        document.querySelectorAll('.color-dot').forEach(el => el.classList.remove('active'));
        d.classList.add('active');
      };
    });

    document.querySelectorAll('.size-btn').forEach(s => {
      s.onclick = () => {
        currentWidth = parseFloat(s.dataset.size);
        document.querySelectorAll('.size-btn').forEach(el => el.classList.remove('active'));
        s.classList.add('active');
      };
    });

    if (btnUndo) btnUndo.onclick = () => undoLastMarking();
    if (btnClear) btnClear.onclick = () => {
      if (confirm('Clear all drawings and markings for ' + sym + '?')) {
        clearAllMarkings();
      }
    };

    const btnList = document.getElementById('btn-list');
    if (btnList) {
      btnList.onclick = () => {
        if (markingsModal.classList.contains('modal-hidden')) {
          renderModalList();
          markingsModal.classList.remove('modal-hidden');
          closeDrawingMode();
        } else {
          markingsModal.classList.add('modal-hidden');
        }
      };
    }

    const modalClose = document.getElementById('modal-close');
    if (modalClose) modalClose.onclick = () => markingsModal.classList.add('modal-hidden');
    const modalClearAll = document.getElementById('modal-clear-all');
    if (modalClearAll) modalClearAll.onclick = () => {
      clearAllMarkings();
      markingsModal.classList.add('modal-hidden');
    };

    window.addEventListener('keydown', e => {
      if (e.key === 'Escape') closeDrawingMode();
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        undoLastMarking();
      }
    });

    syncOverlayMenuCheckboxes();
    updateBadgesVisibility();
    updateMarkers();
    updateForecastLines();
    drawAllForecastVisuals();
    drawAllMarkings();
    updateMarkingsUI();

    // ── Toolbar ──
    const btnTz = document.getElementById('btn-tz');
    const tzCycle = ['IST', 'Local', 'UTC'];
    function updateTzBtn() {
      if (btnTz) {
        const cfg = getTzConfig();
        btnTz.innerText = '🕒 ' + cfg.label;
        btnTz.title = 'Current Timezone: ' + cfg.label + ' (Click to toggle IST / Local / UTC)';
      }
    }
    updateTzBtn();
    if (btnTz) {
      btnTz.onclick = () => {
        const idx = tzCycle.indexOf(currentTz);
        currentTz = tzCycle[(idx + 1) % tzCycle.length];
        updateTzBtn();
        chart.applyOptions({
          timeScale: { tickMarkFormatter: (time, tickMarkType) => formatTickTime(time, tickMarkType) },
          localization: { timeFormatter: (time) => formatLocalizationTime(time) },
        });
        renderLegend();
      };
    }
    document.getElementById('btn-fit').onclick = () => {
      try { store && store.removeItem(KEY); } catch (e) {}
      chart.priceScale('right').applyOptions({ autoScale: true });
      ts.fitContent();
    };
    document.getElementById('btn-live').onclick = () => ts.scrollToRealTime();
    document.getElementById('btn-shot').onclick = () => {
      const c = chart.takeScreenshot();
      const a = document.createElement('a');
      a.href = c.toDataURL('image/png'); a.download = SPEC.filename + '.png'; a.click();
    };

    // ── Resizable Graph Drag Handler ──
    const resizeHandle = document.getElementById('resize-handle');
    let isResizing = false;
    let startY = 0;
    let startHeight = parseInt(wrap.clientHeight || '__HEIGHT__', 10);

    if (resizeHandle) {
      resizeHandle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startY = e.clientY;
        startHeight = wrap.clientHeight;
        resizeHandle.classList.add('resizing');
        document.body.style.cursor = 'ns-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
      });

      window.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        const delta = e.clientY - startY;
        const newH = Math.max(380, Math.min(2200, startHeight + delta));
        wrap.style.height = newH + 'px';
        resizeCanvases();
        try {
          window.parent.postMessage({ isStreamlitMessage: true, type: 'streamlit:setFrameHeight', height: newH + 8 }, '*');
        } catch(err) {}
      });

      window.addEventListener('mouseup', () => {
        if (isResizing) {
          isResizing = false;
          resizeHandle.classList.remove('resizing');
          document.body.style.cursor = '';
          document.body.style.userSelect = '';
          resizeCanvases();
        }
      });
    }

    // ── Fullscreen Toggle Handler ──
    const btnFs = document.getElementById('btn-fs');
    function toggleFullscreen() {
      const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement || document.body.classList.contains('pseudo-fullscreen'));
      if (!isFs) {
        const el = document.documentElement;
        const req = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen || el.msRequestFullscreen;
        if (req) {
          req.call(el).catch(err => {
            console.warn('Native requestFullscreen denied:', err);
            document.body.classList.toggle('pseudo-fullscreen');
            updateFsUI();
          });
        } else {
          document.body.classList.toggle('pseudo-fullscreen');
          updateFsUI();
        }
      } else {
        const exit = document.exitFullscreen || document.webkitExitFullscreen || document.mozCancelFullScreen || document.msExitFullscreen;
        if (exit && (document.fullscreenElement || document.webkitFullscreenElement)) {
          exit.call(document).catch(() => {
            document.body.classList.remove('pseudo-fullscreen');
            updateFsUI();
          });
        } else {
          document.body.classList.remove('pseudo-fullscreen');
          updateFsUI();
        }
      }
    }

    function updateFsUI() {
      const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement || document.body.classList.contains('pseudo-fullscreen'));
      if (btnFs) {
        btnFs.innerHTML = isFs ? '🗗 Exit FS' : '⛶ Fullscreen';
        btnFs.classList.toggle('active', isFs);
      }
      if (isFs) {
        wrap.style.height = '100vh';
        wrap.style.borderRadius = '0';
        wrap.style.border = 'none';
        if (resizeHandle) resizeHandle.style.display = 'none';
      } else {
        wrap.style.height = '__HEIGHT__px';
        wrap.style.borderRadius = '10px';
        wrap.style.border = '1px solid var(--border)';
        if (resizeHandle) resizeHandle.style.display = 'flex';
      }
      setTimeout(resizeCanvases, 50);
      setTimeout(() => { if (ts) ts.fitContent(); }, 120);
    }

    if (btnFs) btnFs.onclick = toggleFullscreen;
    document.addEventListener('fullscreenchange', updateFsUI);
    document.addEventListener('webkitfullscreenchange', updateFsUI);
    window.addEventListener('keydown', (e) => {
      if (e.key === 'F11') {
        e.preventDefault();
        toggleFullscreen();
      }
    });
  } catch (e) {
    wrap.innerHTML = '<div id="err">Chart error: ' + (e && e.stack || e) + '</div>';
  }
})();
</script>
</body></html>
"""


def build_tv_chart_html(spec: ChartSpec, *, title: str, height: int, intraday: bool,
                        state_key: str, zoom_mode: str = "both", filename: str = "chart",
                        tz_mode: str = "IST") -> str:
    """Return a self-contained HTML page (library inlined) for the given ChartSpec."""
    payload = spec.to_json()
    payload.update(title=title, intraday=intraday, stateKey=state_key, zoomMode=zoom_mode,
                   filename=filename, tzMode=tz_mode, up=UP_COLOR, down=DOWN_COLOR)
    # `</` must not appear raw inside the inline <script> JSON.
    spec_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return (_HTML_TEMPLATE
            .replace("__HEIGHT__", str(int(height)))
            .replace("__UP__", UP_COLOR)
            .replace("__DOWN__", DOWN_COLOR)
            .replace("__SPEC__", spec_json)
            .replace("__LWC_BUNDLE__", load_lwc_bundle()))


def render_tv_chart(spec: ChartSpec, *, title: str, height: int, intraday: bool,
                    state_key: str, zoom_mode: str = "both", filename: str = "chart",
                    tz_mode: str = "IST") -> None:
    """Render a ChartSpec with the local Lightweight Charts bundle inside Streamlit."""
    html = build_tv_chart_html(spec, title=title, height=height, intraday=intraday,
                               state_key=state_key, zoom_mode=zoom_mode, filename=filename,
                               tz_mode=tz_mode)
    components.html(html, height=int(height) + 4, scrolling=False)

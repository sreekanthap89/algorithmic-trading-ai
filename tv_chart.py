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
             last_value: bool = False, price_lines: Optional[List[Dict]] = None) -> None:
        if len(points) < 1:
            return
        self.lines.append(dict(data=points, color=color, width=width, style=style,
                               title=title, pane=pane, legend=legend,
                               lastValue=last_value, priceLines=price_lines or []))

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
                   width: int = 1) -> None:
        if _ok(price):
            self.price_lines.append(dict(price=float(price), color=color, title=title,
                                         style=style, width=width))

    def marker(self, ts: Any, price: Optional[float], text: str, color: str,
               shape: str = "circle", position: str = "atPriceMiddle") -> None:
        t = to_time(ts)
        if t not in self._candle_times:
            return
        m = dict(time=t, position=position, shape=shape, color=color, text=text)
        if position.startswith("atPrice"):
            if not _ok(price):
                return
            m["price"] = float(price)
        self.markers.append(m)

    def to_json(self) -> Dict[str, Any]:
        return dict(candles=self.candles, lines=self.lines, baselines=self.baselines,
                    histograms=self.histograms, priceLines=self.price_lines,
                    markers=sorted(self.markers, key=lambda m: m["time"]))


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
        if t in spec._candle_times:
            existing = any(
                em.get("time") == t and em.get("position") == m.get("position")
                for em in spec.markers
            )
            if not existing:
                spec.markers.append(m)


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
    color:var(--txt); font-size:12px; line-height:1.55; }
  #legend .sym { font-weight:600; font-size:13px; color:#fff; margin-right:8px; }
  #legend .k { color:var(--muted); margin-left:6px; }
  #legend .up { color:__UP__; } #legend .dn { color:__DOWN__; }
  #legend .items { display:flex; flex-wrap:wrap; gap:4px 12px; margin-top:2px; }
  #legend .dot { display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:5px; }
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
  #markings-canvas {
    position:absolute; inset:0; width:100%; height:100%; z-index:5; pointer-events:none;
  }
  #draw-canvas {
    position:absolute; inset:0; width:100%; height:100%; z-index:6; pointer-events:none;
  }
  #draw-canvas.active { pointer-events:auto; cursor:crosshair; }
  #draw-canvas.eraser-mode { cursor:cell; }

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

  <div id="tools">
    <div class="tool-group">
      <button id="btn-draw-toggle" title="Freehand & Shape Drawing Tools">✏️ Draw ▾</button>
      <button id="btn-undo" title="Remove last marking (Ctrl+Z)" disabled>⌫ Undo</button>
      <button id="btn-clear" title="Clear all markings" disabled>🗑️ Clear</button>
      <button id="btn-list" title="View & manage all saved markings">📋 (<span id="mark-count">0</span>)</button>
    </div>
    <div class="tool-sep"></div>
    <button id="btn-tz" title="Switch between Local timezone and UTC">🕒 Local</button>
    <button id="btn-fit" title="Fit all data">⤢ Fit</button>
    <button id="btn-live" title="Scroll to latest bar">⇥ Live</button>
    <button id="btn-shot" title="Download PNG">📷</button>
  </div>
</div>
<script>__LWC_BUNDLE__</script>
<script>
(function () {
  const SPEC = __SPEC__;
  const LWC = window.LightweightCharts;
  const wrap = document.getElementById('wrap');
  const el = document.getElementById('chart');
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

    const chart = LWC.createChart(el, {
      autoSize: true,
      layout: { background: { type: LWC.ColorType.Solid, color: '#0D1117' }, textColor: '#B2B5BE',
                fontSize: 11, panes: { separatorColor: 'rgba(255,255,255,0.08)', enableResize: true } },
      grid: { vertLines: { color: 'rgba(255,255,255,0.04)' }, horzLines: { color: 'rgba(255,255,255,0.05)' } },
      crosshair: { mode: LWC.CrosshairMode.Normal,
        vertLine: { color: 'rgba(255,255,255,0.35)', style: 1, labelBackgroundColor: '#2962FF' },
        horzLine: { color: 'rgba(255,255,255,0.35)', style: 1, labelBackgroundColor: '#2962FF' } },
      rightPriceScale: { borderColor: 'rgba(255,255,255,0.1)', scaleMargins: { top: 0.12, bottom: 0.08 } },
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

    // ── Adaptive price precision for Forex (1.0845), Natural Gas (2.645), Oil/Silver (32.45) & Crypto/Gold ──
    const lastCandle = (SPEC.candles && SPEC.candles.length) ? SPEC.candles[SPEC.candles.length - 1] : null;
    const refPrice = lastCandle ? Math.abs(Number(lastCandle.close) || 100) : 100;
    const mainDecimals = refPrice < 1 ? 5 : (refPrice < 20 ? 4 : (refPrice < 200 ? 3 : 2));
    const mainMinMove = Number(Math.pow(10, -mainDecimals).toFixed(mainDecimals));
    const mainPriceFmt = { type: 'price', precision: mainDecimals, minMove: mainMinMove };

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

    // ── Trade price lines (Entry / TP / SL) ──
    (SPEC.priceLines || []).forEach(p => candles.createPriceLine({
      price: p.price, color: p.color, lineWidth: p.width, lineStyle: p.style,
      axisLabelVisible: true, title: p.title,
    }));

    // ── Pattern & AI Trade Deal Markers ──
    if ((SPEC.markers || []).length) {
      try {
        if (typeof candles.setMarkers === 'function') {
          candles.setMarkers(SPEC.markers);
        } else if (typeof LWC !== 'undefined' && typeof LWC.createSeriesMarkers === 'function') {
          LWC.createSeriesMarkers(candles, SPEC.markers);
        }
      } catch (e) { console.warn('markers error', e); }
    }

    // ── Pane sizing ──
    try { const panes = chart.panes(); if (panes.length > 1) { panes[0].setStretchFactor(3); panes[1].setStretchFactor(1); } } catch (e) {}

    // ── Legend (OHLC on crosshair) ──
    const legend = document.getElementById('legend');
    const fmt = v => (v == null || isNaN(v)) ? '—' : Number(v).toLocaleString(undefined, { maximumFractionDigits: mainDecimals, minimumFractionDigits: mainDecimals });
    const last = SPEC.candles[SPEC.candles.length - 1];
    function renderLegend(bar, vals) {
      bar = bar || last;
      let html = `<span class="sym">${SPEC.title}</span>`;
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
      if (legendItems.length) {
        html += '<div class="items">' + legendItems.map((l, i) => {
          const v = vals ? vals[i] : null;
          return `<span><span class="dot" style="background:${l.color}"></span>${l.title}${v != null ? ' <b>' + fmt(v) + '</b>' : ''}</span>`;
        }).join('') + '</div>';
      }
      legend.innerHTML = html;
    }
    renderLegend();
    chart.subscribeCrosshairMove(param => {
      if (!param || !param.time) { renderLegend(); return; }
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

    // ── UNRESTRICTED MARKINGS & DRAWING ENGINE ──
    const sym = (SPEC.filename || '').split('_')[0] || (SPEC.title ? SPEC.title.split(' ')[0] : 'SYMBOL');
    const storageKey = 'tv_markings_' + sym;
    const modalSym = document.getElementById('modal-symbol');
    if (modalSym) modalSym.innerText = sym;

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
      if (markingsCanvas) { markingsCanvas.width = w; markingsCanvas.height = h; }
      if (drawCanvas) { drawCanvas.width = w; drawCanvas.height = h; }
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

    ts.subscribeVisibleLogicalRangeChange(() => drawAllMarkings());
    ts.subscribeVisibleTimeRangeChange(() => drawAllMarkings());

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
    if hasattr(st, "iframe"):
        st.iframe(html, height=int(height) + 4, width="stretch")
    else:
        components.html(html, height=int(height) + 4, scrolling=False)

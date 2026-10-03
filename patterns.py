"""
patterns.py — Chart Pattern Detection & Drawing
Patterns: Bullish Wolfe Wave, Bearish Inverted Cup & Handle,
          Bearish Bat (Harmonic XABCD), Bullish Falling Wedge
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _pivots(arr: np.ndarray, mode: str, order: int = 5) -> List[int]:
    """
    Return indices of local maxima (mode='h') or minima (mode='l').
    Consecutive pivots must be at least *order* bars apart.
    """
    out: List[int] = []
    n = len(arr)
    for i in range(order, n - order):
        window = arr[i - order: i + order + 1]
        extreme = np.max(window) if mode == "h" else np.min(window)
        if arr[i] == extreme:
            if not out or (i - out[-1]) >= order:
                out.append(i)
    return out


def _ly(x1: float, y1: float, x2: float, y2: float, x: float) -> float:
    """Interpolate / extrapolate y on the line through (x1,y1)–(x2,y2)."""
    return y1 if x2 == x1 else y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def _tr(fig: go.Figure, trace, has_sub: bool) -> None:
    """Add a trace to subplot row-1 (if has_sub) or the plain figure."""
    if has_sub:
        fig.add_trace(trace, row=1, col=1)
    else:
        fig.add_trace(trace)


# ─────────────────────────────────────────────────────────────
# 1.  Bullish Wolfe Wave
# ─────────────────────────────────────────────────────────────

def detect_wolfe_wave(df: pd.DataFrame, lookback: int = 150) -> Optional[Dict]:
    """
    Detect a Bullish Wolfe Wave (expanding wedge pointing downward).

    Structure: 3 descending lows  P1 > P3 > P5
               2 descending highs P2 > P4
    • P5 must lie near the line P1→P3 extended (entry zone)
    • P6 = target on the ray P1→P4 projected past P5
    """
    sub = df.tail(lookback)
    hi, lo, dt = sub["High"].values, sub["Low"].values, sub.index
    n = len(lo)
    order = max(3, min(8, n // 18))

    ph = _pivots(hi, "h", order)
    pl = _pivots(lo, "l", order)
    if len(ph) < 2 or len(pl) < 3:
        return None

    best_err, res = np.inf, None

    for ia in range(len(pl) - 2):
        i1 = pl[ia];  y1 = lo[i1]
        for ib in range(ia + 1, len(pl) - 1):
            i3 = pl[ib];  y3 = lo[i3]
            if y3 >= y1:
                continue                          # lows must descend

            p2s = [h for h in ph if i1 < h < i3]
            if not p2s:
                continue
            i2 = max(p2s, key=lambda h: hi[h]);  y2 = hi[i2]

            for ic in range(ib + 1, len(pl)):
                i5 = pl[ic];  y5 = lo[i5]
                if y5 >= y3:
                    continue

                p4s = [h for h in ph if i3 < h < i5]
                if not p4s:
                    continue
                i4 = max(p4s, key=lambda h: hi[h]);  y4 = hi[i4]
                if y4 >= y2:
                    continue                      # highs must descend too

                # P5 should sit near the line P1→P3 extended
                y5e  = _ly(i1, y1, i3, y3, i5)
                err  = abs(y5 - y5e) / (abs(y3 - y1) + 1e-9)
                if err > 0.35 or err >= best_err:
                    continue

                # P6 on ray P1→P4 extended past P5
                step = max(1, i5 - i4)
                i6   = min(i5 + step, n - 1)
                y6   = _ly(i1, y1, i4, y4, i6)

                best_err = err
                res = dict(
                    p1=(dt[i1], y1), p2=(dt[i2], y2),
                    p3=(dt[i3], y3), p4=(dt[i4], y4),
                    p5=(dt[i5], y5), p6=(dt[i6], y6),
                )
    return res


def draw_wolfe_wave(fig: go.Figure, p: Dict, has_sub: bool) -> None:
    p1, p2, p3, p4, p5, p6 = (p[k] for k in ("p1", "p2", "p3", "p4", "p5", "p6"))

    # Filled wedge body
    _tr(fig, go.Scatter(
        x=[p1[0], p2[0], p4[0], p6[0], p5[0], p3[0], p1[0]],
        y=[p1[1], p2[1], p4[1], p6[1], p5[1], p3[1], p1[1]],
        fill="toself", fillcolor="rgba(38,166,154,0.13)",
        line=dict(color="rgba(0,0,0,0)"),
        name="🌊 Wolfe Wave", showlegend=True,
    ), has_sub)

    # Support line 1-3-5 (entry ray)
    _tr(fig, go.Scatter(
        x=[p1[0], p3[0], p5[0]], y=[p1[1], p3[1], p5[1]],
        mode="lines", line=dict(color="#26A69A", width=2, dash="dot"),
        name="WW Support 1-3-5", showlegend=False,
    ), has_sub)

    # Resistance line 2-4 extended to P6
    _tr(fig, go.Scatter(
        x=[p2[0], p4[0], p6[0]], y=[p2[1], p4[1], p6[1]],
        mode="lines", line=dict(color="#EF5350", width=2, dash="dot"),
        name="WW Resistance 2-4", showlegend=False,
    ), has_sub)

    # Target ray P1→P4→P6 (gold)
    _tr(fig, go.Scatter(
        x=[p1[0], p4[0], p6[0]], y=[p1[1], p4[1], p6[1]],
        mode="lines", line=dict(color="#FFD700", width=1.8, dash="dash"),
        name="WW Target Ray 1→4→6", showlegend=False,
    ), has_sub)

    # Entry zone highlight at P5
    _tr(fig, go.Scatter(
        x=[p5[0], p5[0]], y=[p5[1] * 0.997, p5[1] * 1.003],
        mode="lines", line=dict(color="rgba(0,255,127,0.7)", width=8),
        name="WW Entry Zone", showlegend=False,
    ), has_sub)

    # Numbered labels
    for lbl, pt, pos, col in [
        ("1",       p1, "bottom center", "#87CEEB"),
        ("2",       p2, "top center",    "#87CEEB"),
        ("3",       p3, "bottom center", "#87CEEB"),
        ("4",       p4, "top center",    "#87CEEB"),
        ("5\nEntry",p5, "bottom center", "#00FF7F"),
        ("6\nTarget",p6,"top center",    "#FFD700"),
    ]:
        _tr(fig, go.Scatter(
            x=[pt[0]], y=[pt[1]],
            mode="markers+text",
            marker=dict(size=9, color=col, symbol="circle",
                        line=dict(width=1.5, color="white")),
            text=[lbl], textposition=pos,
            textfont=dict(color=col, size=10),
            showlegend=False, name=f"WW {lbl}",
        ), has_sub)


# ─────────────────────────────────────────────────────────────
# 2.  Bearish Inverted Cup & Handle
# ─────────────────────────────────────────────────────────────

def detect_cup_handle(df: pd.DataFrame, lookback: int = 150) -> Optional[Dict]:
    """
    Bearish Inverted Cup & Handle.
    Cup  = dome/hump shape (price rises, arcs, comes back down).
    Handle = minor pullback on the right side of the cup.
    Target = neckline − cup height (100 % projection downward).
    """
    sub = df.tail(lookback)
    hi  = sub["High"].values
    lo  = sub["Low"].values
    cl  = sub["Close"].values
    dt  = sub.index
    n   = len(cl)

    if n < 40:
        return None

    # Cup top = highest high in the window
    cup_i = int(np.argmax(hi))
    if cup_i < 8 or cup_i > n - 10:
        return None
    cup_p = hi[cup_i]

    # Left rim: lowest close in the 40 bars before cup top
    left_start = max(0, cup_i - 40)
    lr_offset   = int(np.argmin(cl[left_start: cup_i]))
    lr_i        = left_start + lr_offset
    lr_p        = cl[lr_i]

    # Right minimum: deepest low on the right of cup top
    rm_offset   = int(np.argmin(lo[cup_i:]))
    rm_i        = cup_i + rm_offset
    rm_p        = lo[rm_i]
    if rm_i >= n - 4:
        return None

    # Neckline / support (average of both rims)
    support_p = (lr_p + rm_p) / 2
    cup_h     = cup_p - support_p
    if cup_h < cup_p * 0.005:
        return None

    # Handle top: highest bar in the 20 bars after right min
    ht_slice  = hi[rm_i: min(rm_i + 20, n)]
    ht_offset = int(np.argmax(ht_slice))
    ht_i      = rm_i + ht_offset
    ht_p      = hi[ht_i]
    if ht_p >= cup_p * 0.96:          # handle must be below cup top
        return None

    target_p = support_p - cup_h
    cur_i    = n - 1

    return dict(
        left_rim  = (dt[lr_i],  lr_p),
        cup_top   = (dt[cup_i], cup_p),
        right_min = (dt[rm_i],  rm_p),
        handle_top= (dt[ht_i],  ht_p),
        current   = (dt[cur_i], cl[cur_i]),
        support   = support_p,
        target    = target_p,
        cup_h     = cup_h,
    )


def draw_cup_handle(fig: go.Figure, p: Dict, has_sub: bool) -> None:
    lr  = p["left_rim"]
    ct  = p["cup_top"]
    rm  = p["right_min"]
    ht  = p["handle_top"]
    cur = p["current"]
    sup, tgt = p["support"], p["target"]

    # Cup arc: left rim → top → right min
    _tr(fig, go.Scatter(
        x=[lr[0], ct[0], rm[0]], y=[lr[1], ct[1], rm[1]],
        mode="lines", line=dict(color="#64B5F6", width=2.5),
        name="☕ Inv. Cup & Handle", showlegend=True,
    ), has_sub)

    # Handle: right min → handle top → current
    _tr(fig, go.Scatter(
        x=[rm[0], ht[0], cur[0]], y=[rm[1], ht[1], cur[1]],
        mode="lines", line=dict(color="#FFA726", width=2.5),
        name="C&H Handle", showlegend=False,
    ), has_sub)

    # Neckline (horizontal support / resistance)
    _tr(fig, go.Scatter(
        x=[lr[0], cur[0]], y=[sup, sup],
        mode="lines", line=dict(color="#EF5350", width=1.5, dash="dash"),
        name="C&H Support Line", showlegend=False,
    ), has_sub)

    # Target level
    _tr(fig, go.Scatter(
        x=[rm[0], cur[0]], y=[tgt, tgt],
        mode="lines", line=dict(color="#26A69A", width=1.5, dash="dash"),
        name="C&H Target", showlegend=False,
    ), has_sub)

    # Key point markers
    for lbl, pt, pos, col, sym in [
        ("Cup Top",    ct,               "top center",    "#64B5F6", "circle"),
        ("Breakout",   (cur[0], sup),    "top right",     "#FF5722", "triangle-down"),
        ("Target ↓",   (cur[0], tgt),    "bottom right",  "#26A69A", "triangle-down"),
    ]:
        _tr(fig, go.Scatter(
            x=[pt[0]], y=[pt[1]],
            mode="markers+text",
            marker=dict(size=9, color=col, symbol=sym,
                        line=dict(width=1.5, color="white")),
            text=[lbl], textposition=pos,
            textfont=dict(color=col, size=10),
            showlegend=False, name=lbl,
        ), has_sub)


# ─────────────────────────────────────────────────────────────
# 3.  Bearish Bat (Harmonic XABCD)
# ─────────────────────────────────────────────────────────────

def detect_bearish_bat(df: pd.DataFrame, lookback: int = 250) -> Optional[Dict]:
    """
    Detect Bearish Bat harmonic pattern (X→A→B→C→D).
    X: swing high   A: swing low   B: swing high (~0.25–0.65 of XA)
    C: swing low    D: swing high  (~0.70–1.05 of XA — relaxed Bat ratio)
    CD extension of BC: 1.0–3.5
    Entry = SELL at D.  Target = A level.
    """
    sub = df.tail(lookback)
    hi, lo, dt = sub["High"].values, sub["Low"].values, sub.index
    n = len(hi)
    order = max(2, min(4, n // 40))

    ph = _pivots(hi, "h", order)
    pl = _pivots(lo, "l", order)
    if len(ph) < 3 or len(pl) < 2:
        return None

    best_err, res = np.inf, None

    # Scan ALL recent X candidates (not just last 10)
    for xi in reversed(ph):
        yx = hi[xi]

        for ai in [l for l in pl if l > xi][:10]:
            ya = lo[ai];  xa = yx - ya
            if xa <= 0 or xa < yx * 0.001:  # Allow very small ranges
                continue

            for bi in [h for h in ph if h > ai][:10]:
                yb    = hi[bi]
                ab_r  = (yb - ya) / xa
                if not 0.20 <= ab_r <= 0.70:  # Further relaxed
                    continue

                for ci in [l for l in pl if l > bi][:10]:
                    yc   = lo[ci];  ab = yb - ya
                    bc_r = (yb - yc) / ab if ab > 0 else 0
                    if not 0.20 <= bc_r <= 1.0:  # Further relaxed
                        continue

                    for di in [h for h in ph if h > ci][:10]:
                        yd   = hi[di]
                        xd_r = (yd - ya) / xa        # 0.886 is golden ratio
                        if not 0.60 <= xd_r <= 1.15:  # Very relaxed
                            continue

                        bc   = yb - yc
                        cd_r = (yd - yc) / bc if bc > 0 else 0
                        if not 0.8 <= cd_r <= 4.0:  # Very relaxed
                            continue

                        err = (abs(xd_r - 0.886) * 2
                               + abs(ab_r  - 0.382)
                               + abs(cd_r  - 1.618) * 0.2)
                        if err < best_err:
                            best_err = err
                            res = dict(
                                X=(dt[xi], yx), A=(dt[ai], ya),
                                B=(dt[bi], yb), C=(dt[ci], yc),
                                D=(dt[di], yd),
                                target = ya,
                                stop   = yx * 1.003,
                                ratios = dict(
                                    AB_XA = round(ab_r,  3),
                                    BC_AB = round(bc_r,  3),
                                    CD_BC = round(cd_r,  3),
                                    XD_XA = round(xd_r,  3),
                                ),
                            )
    return res


def draw_bearish_bat(fig: go.Figure, p: Dict, has_sub: bool) -> None:
    X, A, B, C, D = p["X"], p["A"], p["B"], p["C"], p["D"]
    tgt, stop = p["target"], p["stop"]
    r = p["ratios"]

    # Filled bat body (X→A→B→C→D→X path)
    _tr(fig, go.Scatter(
        x=[X[0], A[0], B[0], C[0], D[0], X[0]],
        y=[X[1], A[1], B[1], C[1], D[1], X[1]],
        fill="toself", fillcolor="rgba(244,143,177,0.11)",
        line=dict(color="rgba(0,0,0,0)"),
        name="🦇 Bearish Bat", showlegend=True,
    ), has_sub)

    # XABCD legs
    for i, (pt1, pt2) in enumerate(zip([X, A, B, C], [A, B, C, D])):
        _tr(fig, go.Scatter(
            x=[pt1[0], pt2[0]], y=[pt1[1], pt2[1]],
            mode="lines", line=dict(color="#EF9A9A", width=1.8),
            showlegend=False, name="Bat leg",
        ), has_sub)

    # XD completion line (0.886)
    _tr(fig, go.Scatter(
        x=[X[0], D[0]], y=[X[1], D[1]],
        mode="lines", line=dict(color="#FFA726", width=1.4, dash="dot"),
        name=f"XD = {r['XD_XA']:.3f}  (0.886)", showlegend=False,
    ), has_sub)

    # Stop-loss and target horizontal levels (extended to latest candle)
    end_x = D[0]
    try:
        if len(fig.data) > 0 and hasattr(fig.data[0], 'x') and len(fig.data[0].x) > 0:
            end_x = fig.data[0].x[-1]
    except Exception:
        pass

    for lbl, y_val, col in [
        ("Bat Stop-Loss", stop, "#EF5350"),
        ("Bat Target",    tgt,  "#26A69A"),
    ]:
        _tr(fig, go.Scatter(
            x=[X[0], end_x], y=[y_val, y_val],
            mode="lines+text", line=dict(color=col, width=1.5, dash="dash"),
            text=["", f"  {lbl}: ${y_val:,.2f}"],
            textposition="middle right",
            textfont=dict(color=col, size=11),
            name=lbl, showlegend=True,
        ), has_sub)

    # XABCD point labels
    for lbl, pt, pos, col in [
        ("X",         X, "top center",    "#F48FB1"),
        ("A",         A, "bottom center", "#CE93D8"),
        ("B",         B, "top center",    "#90CAF9"),
        ("C",         C, "bottom center", "#A5D6A7"),
        ("D\nEntry",  D, "top center",    "#FFCC02"),
    ]:
        _tr(fig, go.Scatter(
            x=[pt[0]], y=[pt[1]],
            mode="markers+text",
            marker=dict(size=9, color=col, symbol="circle",
                        line=dict(width=1.5, color="white")),
            text=[lbl], textposition=pos,
            textfont=dict(color=col, size=10),
            showlegend=False, name=f"Bat {lbl}",
        ), has_sub)

    # Fibonacci ratio label (0.886) on the XD line midpoint
    _tr(fig, go.Scatter(
        x=[B[0]], y=[(X[1] + A[1]) / 2],
        mode="text",
        text=[f"0.886\n(actual {r['XD_XA']:.3f})"],
        textfont=dict(color="#FFA726", size=9),
        showlegend=False, name="",
    ), has_sub)


# ─────────────────────────────────────────────────────────────
# 4.  Bullish Falling Wedge
# ─────────────────────────────────────────────────────────────

def detect_falling_wedge(df: pd.DataFrame, lookback: int = 200) -> Optional[Dict]:
    """
    Detect a Bullish Falling Wedge.
    Both trendlines slope downward and CONVERGE (upper line steeper than lower).
    Entry = breakout above upper line at the apex.
    Target = height of wedge at start, projected from breakout.
    """
    sub = df.tail(lookback)
    hi, lo, dt = sub["High"].values, sub["Low"].values, sub.index
    n = len(hi)
    order = max(2, min(4, n // 35))

    ph = _pivots(hi, "h", order)
    pl = _pivots(lo, "l", order)
    if len(ph) < 2 or len(pl) < 2:
        return None

    best_score, res = np.inf, None

    for i in range(len(ph) - 1):
        h1i, h2i = ph[i], ph[i + 1]
        yh1, yh2 = hi[h1i], hi[h2i]
        if yh2 >= yh1:
            continue                              # upper line must descend
        sl_up = (yh2 - yh1) / (h2i - h1i)

        # Allow lows in wider window
        lows_rng = [l for l in pl if h1i - order*3 <= l <= h2i + order*3]
        if len(lows_rng) < 2:
            continue

        for m in range(len(lows_rng) - 1):
            l1i, l2i = lows_rng[m], lows_rng[m + 1]
            yl1, yl2 = lo[l1i], lo[l2i]
            if yl2 >= yl1:
                continue
            sl_lo = (yl2 - yl1) / (l2i - l1i)
            if sl_up >= sl_lo:                   # upper line must be steeper (more negative) for convergence
                continue

            y_up_l1 = _ly(h1i, yh1, h2i, yh2, l1i)
            if yl1 >= y_up_l1:
                continue
            width = y_up_l1 - yl1
            if width < (yh1 - yl1) * 0.04:
                continue

            # Apex (convergence point)
            b_up  = yh1 - sl_up * h1i
            b_lo  = yl1 - sl_lo * l1i
            denom = sl_up - sl_lo
            if abs(denom) < 1e-10:
                continue
            apex_x = (b_lo - b_up) / denom
            # Require rightmost point < 40% future
            if apex_x <= h2i or apex_x > n * 0.40 or apex_x < 0:
                continue
            apex_y = sl_up * apex_x + b_up
            
            # Sanity check: apex price within reasonable range
            min_price = min(lo[l1i], lo[l2i])
            max_price = max(hi[h1i], hi[h2i])
            if apex_y < min_price * 0.9 or apex_y > max_price * 1.1:
                continue

            # Breakout: at upper line at h2
            bk_p    = yh2
            breakout_height = bk_p - yl2
            target  = bk_p + breakout_height * 0.8
            stop    = yl2 * 0.998

            score = abs(sl_up / sl_lo - 1.8)    # prefer slope ratio ~1.8
            if score < best_score:
                best_score = score
                res = dict(
                    h1=(dt[h1i], yh1), h2=(dt[h2i], yh2),
                    l1=(dt[l1i], yl1), l2=(dt[l2i], yl2),
                    apex=(dt[min(int(apex_x), n - 1)], apex_y),
                    target=target,
                    stop=stop,
                )
    return res


def draw_falling_wedge(fig: go.Figure, p: Dict, has_sub: bool) -> None:
    h1, h2 = p["h1"], p["h2"]
    l1, l2 = p["l1"], p["l2"]
    apex   = p["apex"]
    tgt    = p["target"]
    stop   = p["stop"]

    # Filled wedge body
    _tr(fig, go.Scatter(
        x=[h1[0], h2[0], apex[0], l2[0], l1[0], h1[0]],
        y=[h1[1], h2[1], apex[1], l2[1], l1[1], h1[1]],
        fill="toself", fillcolor="rgba(186,104,200,0.13)",
        line=dict(color="rgba(0,0,0,0)"),
        name="📐 Falling Wedge", showlegend=True,
    ), has_sub)

    # Resistance (upper trendline)
    _tr(fig, go.Scatter(
        x=[h1[0], h2[0], apex[0]], y=[h1[1], h2[1], apex[1]],
        mode="lines", line=dict(color="#EF5350", width=2),
        name="FW Resistance", showlegend=False,
    ), has_sub)

    # Support (lower trendline, steeper)
    _tr(fig, go.Scatter(
        x=[l1[0], l2[0], apex[0]], y=[l1[1], l2[1], apex[1]],
        mode="lines", line=dict(color="#26A69A", width=2),
        name="FW Support", showlegend=False,
    ), has_sub)

    # Target level (gold dashed)
    _tr(fig, go.Scatter(
        x=[h1[0], apex[0]], y=[tgt, tgt],
        mode="lines", line=dict(color="#FFD700", width=1.5, dash="dash"),
        name="FW Target", showlegend=False,
    ), has_sub)

    # Stop-loss level (red dotted)
    _tr(fig, go.Scatter(
        x=[l1[0], apex[0]], y=[stop, stop],
        mode="lines", line=dict(color="#EF5350", width=1.2, dash="dot"),
        name="FW Stop-Loss", showlegend=False,
    ), has_sub)

    # Position Opening marker at apex
    _tr(fig, go.Scatter(
        x=[apex[0]], y=[apex[1]],
        mode="markers+text",
        marker=dict(size=11, color="#00FF7F", symbol="triangle-up",
                    line=dict(width=2, color="white")),
        text=["Position\nOpening"], textposition="top right",
        textfont=dict(color="#00FF7F", size=10),
        showlegend=False, name="FW Entry",
    ), has_sub)

    # TARGET / STOP labels at left edge
    for lbl, y_val, col in [
        ("TARGET ↑",    tgt,  "#FFD700"),
        ("STOP-LOSS ↓", stop, "#EF5350"),
    ]:
        _tr(fig, go.Scatter(
            x=[h1[0]], y=[y_val],
            mode="text", text=[lbl],
            textfont=dict(color=col, size=9),
            showlegend=False, name="",
        ), has_sub)


# ─────────────────────────────────────────────────────────────
# Master entry point
# ─────────────────────────────────────────────────────────────

def detect_and_draw_patterns(
    fig: go.Figure,
    df: pd.DataFrame,
    has_sub: bool,
    show_wolfe:  bool = False,
    show_cup:    bool = False,
    show_bat:    bool = False,
    show_wedge:  bool = False,
) -> dict:
    """
    Detect all enabled chart patterns in *df* and draw them on *fig*.
    Each detector is wrapped in try/except so a single failure never
    crashes the chart.
    Returns a dictionary of detected patterns for trade execution planning.
    """
    # 🛡️ Robustness: Sanitization pass to handle NaNs in price data before detection
    df_clean = df.copy()
    df_clean[['High', 'Low', 'Close']] = df_clean[['High', 'Low', 'Close']].ffill().bfill()

    detected = {}

    if show_wolfe:
        try:
            pat = detect_wolfe_wave(df_clean)
            if pat:
                draw_wolfe_wave(fig, pat, has_sub)
                detected["wolfe_wave"] = pat
        except Exception:
            pass

    if show_cup:
        try:
            pat = detect_cup_handle(df_clean)
            if pat:
                draw_cup_handle(fig, pat, has_sub)
                detected["cup_handle"] = pat
        except Exception:
            pass

    if show_bat:
        try:
            pat = detect_bearish_bat(df_clean)
            if pat:
                draw_bearish_bat(fig, pat, has_sub)
                detected["bearish_bat"] = pat
        except Exception:
            pass

    if show_wedge:
        try:
            pat = detect_falling_wedge(df_clean)
            if pat:
                draw_falling_wedge(fig, pat, has_sub)
                detected["falling_wedge"] = pat
        except Exception:
            pass

    return detected

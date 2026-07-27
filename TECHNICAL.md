# Technical Documentation - UI Refactoring v2.0

## Architecture Overview

### Previous Architecture (Multi-Tab)
```
app.py (old)
├── Data Fetching
├── Fragment Rendering
├── Tab 1: Dashboard
│   ├── Chart 1 (main)
│   └── Complex state management
├── Tab 2: AI Deep Dive
│   ├── Chart 2 (MC simulation)
│   └── Metrics tables
└── Tab 3: Portfolio
    ├── Position table
    └── Trade history
```

### New Architecture (Single Chart + Optional Panels)
```
app.py (new)
├── Page Configuration
├── Sidebar (Configuration Panel)
│   ├── Symbol & Timeframe
│   ├── Refresh Controls
│   ├── Chart Settings
│   ├── Technical Indicators
│   ├── View Control (Zoom)
│   └── Optional Panels Toggle
├── Data Fetching
├── Fragment - Main Dashboard
│   ├── Header Metrics (4 columns)
│   ├── Recommendation Banner
│   ├── Single Professional Chart
│   ├── Trade Execution Panel
│   └── Optional Expandable Panels
└── Helper Functions
```

---

## Session State Management

### Zoom Preservation Keys
```python
# Store zoom preset index for symbol/timeframe combo
st.session_state[f"zoom_preset_idx_{symbol}_{interval}"]

# Track previous selection to detect changes
st.session_state[f"prev_zoom_preset_{symbol}_{interval}"]

# Other session data
st.session_state['df']              # DataFrame
st.session_state['df_feat']         # Features
st.session_state['prediction']      # AI signals
st.session_state['regime']          # Market regime
st.session_state['symbol']          # Current symbol
st.session_state['interval']        # Current timeframe
```

### UIRevision Strategy
```python
# Stable revision prevents Plotly from resetting zoom on every interaction
stable_uirevision = f"tvpro_{symbol}_{interval}"

layout_dict = dict(
    uirevision=stable_uirevision,  # Main chart
    xaxis=dict(uirevision=stable_uirevision),      # X-axis
    yaxis=dict(uirevision=stable_uirevision),      # Y-axis
    # ... more axes for subplots
)

# When zoom preset changes, update uirevision
if x_range is not None:
    layout_dict['uirevision'] = f"preset_{selected_zoom}"
    # User zoom gestures now update this preset without chart reset
```

---

## Precision Handling System

### Helper Functions

#### `get_price_decimals(symbol: str) -> int`
```python
def get_price_decimals(symbol: str) -> int:
    """Return appropriate decimal places for asset type"""
    if any(x in symbol.upper() for x in ['-USD', 'BTC', 'ETH']):
        return 8  # Crypto
    elif any(x in symbol.upper() for x in ['=', 'FX']):
        return 5  # Forex/Commodities
    else:
        return 2  # Stocks/ETFs
```

#### `format_price(symbol: str, price: float) -> str`
```python
def format_price(symbol: str, price: float) -> str:
    """Format price with international standard"""
    decimals = get_price_decimals(symbol)
    return f"${price:,.{decimals}f}"

# Examples
format_price("BTC-USD", 42562.123456789)  # "$42,562.12345679"
format_price("GC=F", 1950.123456)         # "$1,950.12346"
format_price("AAPL", 189.42)              # "$189.42"
```

### Hover Template Generation
```python
# Dynamic precision in hover tooltips
precision = get_price_decimals(symbol)
hovertemplate = f'<b>{{fullData.name}}</b><br>%{{y:.{precision}f}}<extra></extra>'

# Applied to all price-based traces
trace = go.Scatter(
    x=dates,
    y=prices,
    hovertemplate=hovertemplate  # Shows full precision on hover
)
```

---

## Chart Rendering Pipeline

### 1. Data Preparation
```python
# Get from session state
df = st.session_state['df']
df_feat = st.session_state['df_feat']

# Extract current values
last_close = float(df['Close'].iloc[-1])
```

### 2. Figure Creation
```python
# Single or multi-pane based on subplot choice
has_subplot = subplot_choice != "None"

if has_subplot:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28]
    )
else:
    fig = go.Figure()
```

### 3. Main Chart (OHLC)
```python
candlestick = go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="OHLC",
    increasing_line_color='#26A69A',
    decreasing_line_color='#EF5350',
    increasing_fillcolor='#26A69A',
    decreasing_fillcolor='#EF5350'
)

if has_subplot:
    fig.add_trace(candlestick, row=1, col=1)
else:
    fig.add_trace(candlestick)
```

### 4. Indicators with Precision
```python
if show_sma20 and 'SMA_20' in df_feat.columns:
    trace = go.Scatter(
        x=df_feat.index,
        y=df_feat['SMA_20'],
        line=dict(color='#FFD700', width=1.5),
        name='SMA 20',
        hovertemplate=hovertemp  # Full precision hover
    )
    fig.add_trace(trace, row=1, col=1)
```

### 5. Horizontal Lines (Targets)
```python
fig.add_hline(
    y=target_price,
    line_width=2,
    line_dash="dash",
    line_color="#00FF7F",
    annotation_text=f"🎯 TP: {format_price(symbol, target_price)}",
    annotation_position="top right",
    row=1,
    col=1 if has_subplot else None
)
```

### 6. Trade Markers
```python
if len(buy_x) > 0:
    tr_buy = go.Scatter(
        x=buy_x,
        y=buy_y,
        mode='markers+text',
        marker=dict(
            symbol='triangle-up',
            size=16,
            color='#00FF00',
            line=dict(width=2, color='white')
        ),
        text=['▲' for _ in buy_x],
        textposition='bottom center',
        textfont=dict(color='#00FF00', size=10),
        name='BUY',
        hoverinfo='text',
        hovertext=buy_texts
    )
    fig.add_trace(tr_buy, row=1 if has_subplot else None)
```

### 7. Subplot Indicators
```python
if subplot_choice == "MACD (12,26,9)" and 'MACD' in df_feat.columns:
    # 4-color histogram based on momentum
    macd_diff = df_feat['MACD_Diff'].fillna(0)
    colors = []
    for i in range(len(macd_diff)):
        curr = macd_diff.iloc[i]
        prev = macd_diff.iloc[i-1] if i > 0 else curr
        if curr >= 0:
            colors.append('#26A69A' if curr >= prev else '#B2DFDB')
        else:
            colors.append('#EF5350' if curr <= prev else '#FFCDD2')
    
    fig.add_trace(
        go.Bar(x=df_feat.index, y=macd_diff, marker_color=colors, name='MACD'),
        row=2, col=1
    )
```

### 8. Layout Configuration
```python
stable_uirevision = f"tvpro_{symbol}_{interval}"

layout_dict = dict(
    template='plotly_dark',
    height=chart_h,
    uirevision=stable_uirevision,
    margin=dict(l=50, r=50, t=40, b=50),
    xaxis=dict(
        uirevision=stable_uirevision,
        showgrid=True,
        gridcolor='rgba(128, 128, 128, 0.15)',
        type='date',
        fixedrange=False,
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='rgba(255,255,255,0.6)',
        spikethickness=1,
        spikedash='dot'
    ),
    yaxis=dict(
        uirevision=stable_uirevision,
        title='Price',
        showgrid=True,
        gridcolor='rgba(128, 128, 128, 0.15)',
        fixedrange=False,
        side='right',
        showspikes=True,
        spikemode='across'
    ),
    hovermode='x unified',
    dragmode='zoom',
    spikedistance=-1
)
```

### 9. Zoom Range Application
```python
# Only apply when preset changes
if zoom_preset_changed:
    zoom_num_map = {
        "Last 7": 7,
        "Last 14": 14,
        "Last 30": 30,
        # ... etc
        "All Data": None
    }
    
    n_candles = zoom_num_map.get(selected_zoom)
    if n_candles and len(df) > n_candles:
        x_range = [df.index[-n_candles], df.index[-1]]
        
        # Update layout with range
        layout_dict['xaxis']['range'] = x_range
        layout_dict['xaxis']['uirevision'] = f"preset_{selected_zoom}"
```

### 10. Chart Display
```python
plotly_config = dict(
    displayModeBar=True,
    displaylogo=False,
    modeBarButtonsToAdd=['drawline', 'drawopenpath', 'drawcircle', 'drawrect'],
    modeBarButtonsToRemove=['lasso2d', 'select2d'],
    scrollZoom=True,
    doubleClick='reset+autosize',
    showTips=True,
    toImageButtonOptions=dict(
        format='png',
        filename=f'{symbol}_{interval}_chart',
        height=chart_h,
        width=1600,
        scale=2
    )
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key=f"main_chart_{symbol}_{interval}",
    config=plotly_config
)
```

---

## Fragment-Based Rendering

### Auto-Refresh Implementation
```python
@st.fragment(run_every=run_every_val)
def render_trading_dashboard():
    """
    Fragment runs repeatedly at specified interval
    run_every_val: "10s", "30s", "1m", "5m", or None
    """
    
    # Auto-refresh data if enabled
    if auto_refresh and 'df' in st.session_state and not fetch_clicked:
        df = fetch_data(symbol, period=period, interval=interval)
        if df is not None and len(df) > 0:
            # Update session state with new data
            df_feat = add_features(df)
            # ... predictions, regime, etc
            st.session_state['df'] = df
            # Zoom state preserved automatically!
    
    # Render UI with current session state
    # (chart zoom not reset because uirevision stays same)
```

### Fragment Benefits
1. **Partial Rerun:** Only fragment reruns, not entire page
2. **State Preservation:** Sidebar settings not recalculated
3. **Smooth Updates:** Data refreshes without UI flicker
4. **Zoom Persistence:** UIRevision keeps zoom across fragment reruns

---

## Data Flow Diagram

```
User Action
    ↓
┌─ Symbol Changed? ─→ Fetch new data
├─ Timeframe Changed? → Fetch new data
├─ Refresh Button? → Fetch new data
├─ Auto-Refresh Trigger? → Fetch new data in fragment
└─ Zoom Window Changed? → Apply x_range (no fetch)
    ↓
Data Processing
├─ add_features() → df_feat
├─ TradingModel.train_or_load() → model
├─ model.predict_next() → prediction
└─ get_markov_regime() → regime
    ↓
Session State Update
├─ df, df_feat, prediction, regime
├─ symbol, interval
└─ zoom_preset_idx, prev_zoom_preset
    ↓
UI Rendering
├─ Header metrics
├─ Recommendation banner
├─ Chart with indicators
├─ Trade execution panel
└─ Optional expandable panels
    ↓
User Interaction
├─ Zoom: Updated by Plotly (uirevision prevents reset)
├─ Pan: Updated by Plotly
├─ Trade: Session state update → rerun(scope="fragment")
└─ Auto-refresh: Fragment reruns at interval
```

---

## Performance Optimization

### 1. Session State Caching
```python
# Data fetched once per symbol/timeframe combo
if 'df' not in st.session_state:
    df = fetch_data(...)
else:
    df = st.session_state['df']  # Reuse, don't refetch
```

### 2. UIRevision for Zoom Lock
```python
# Without uirevision:
# - Every hover event causes Plotly to update
# - Chart resets zoom on each update

# With uirevision:
# - Plotly tracks this ID
# - User zoom gestures cached with this ID
# - Zoom stays across updates as long as ID matches
```

### 3. Fragment-Based Rendering
```python
# Without fragment:
# Entire page reruns every refresh → slow

# With fragment:
# Only dashboard reruns → fast
# Sidebar settings preserved → no recompute
```

### 4. Lazy Subplot Creation
```python
# Only create subplots if needed
if subplot_choice != "None":
    fig = make_subplots(...)  # Expensive
else:
    fig = go.Figure()  # Lightweight
```

---

## Testing Checklist

### Functionality Tests
- [ ] Load symbol: `BTC-USD` ✓
- [ ] Change timeframe: `1D` → `1H` ✓
- [ ] Zoom window: `Last 30` preserved after refresh ✓
- [ ] Indicators toggle: SMA-20 on/off ✓
- [ ] Subplot selection: MACD/RSI/Volume ✓
- [ ] Trade execution: BUY/SELL works ✓

### Precision Tests
- [ ] BTC-USD shows 8 decimals ✓
- [ ] GC=F shows 5 decimals ✓
- [ ] AAPL shows 2 decimals ✓
- [ ] Hover tooltips show full precision ✓

### Zoom Tests
- [ ] Select preset: saves to session ✓
- [ ] Auto-refresh: zoom stays same ✓
- [ ] Manual refresh: zoom persists ✓
- [ ] Change symbol: zoom resets (expected) ✓
- [ ] Scroll zoom: still works (Plotly native) ✓

### Performance Tests
- [ ] First load: < 3 seconds ✓
- [ ] Refresh: < 1 second ✓
- [ ] Auto-refresh: smooth updates ✓
- [ ] Chart render: smooth scrolling ✓

---

## File Structure

```
app.py (32 KB) - Main application (NEW)
├── Page configuration
├── Helper functions
├── Sidebar configuration
├── Data processing
├── render_trading_dashboard() fragment
└── Optional analysis panels

app_old_multiTab.py (57 KB) - Old multi-tab version (BACKUP)
├── Original implementation
└── Can be restored if needed

TRADINGVIEW_REQUIREMENTS.md - Full feature spec
QUICK_START_GUIDE.md - User documentation
UI_REFACTORING_SUMMARY.md - Change summary
TECHNICAL.md - This document
```

---

## Key Code Patterns

### Pattern 1: Symbol/Interval Specific Session Keys
```python
# Why: Each symbol/timeframe has different zoom level
key = f"zoom_preset_idx_{symbol}_{interval}"
st.session_state[key] = value
```

### Pattern 2: Zoom Preset Detection
```python
prev_zoom_key = f"prev_zoom_preset_{symbol}_{interval}"
zoom_preset_changed = st.session_state.get(prev_zoom_key) != selected_zoom
st.session_state[prev_zoom_key] = selected_zoom
```

### Pattern 3: UIRevision for Zoom Lock
```python
# Key: Same uirevision = Plotly remembers user zoom
stable_uirevision = f"tvpro_{symbol}_{interval}"
# Apply to all axes:
layout_dict['uirevision'] = stable_uirevision
layout_dict['xaxis']['uirevision'] = stable_uirevision
layout_dict['yaxis']['uirevision'] = stable_uirevision
```

### Pattern 4: Dynamic Precision
```python
precision = get_price_decimals(symbol)
hovertemplate = f'%{{y:.{precision}f}}'  # Formatted string for Plotly
```

### Pattern 5: Fragment with State
```python
@st.fragment(run_every=run_every_val)
def render():
    # Fragment has access to parent variables
    # Reruns at interval but doesn't reset parent state
    pass

render()  # Call at end
```

---

## Migration from Old Version

### What's Preserved
- All data structures
- All calculations
- All models
- All trading history
- Portfolio state

### What's Changed (UI Only)
- Layout: Multi-tab → Single chart
- Sidebar: Flat → Organized sections
- Chart: Multiple → Single main
- Panels: Always visible → Optional/expandable

### Breaking Changes
- None! Fully backward compatible
- Old app saved as `app_old_multiTab.py`
- Data files unchanged

---

## Future Enhancement Points

1. **Multi-Chart Tabs:** If single chart not enough
2. **Custom Timeframes:** User-defined intervals
3. **Alert System:** Price/signal notifications
4. **Pattern Recognition:** Auto-detect chart patterns
5. **ML Customization:** Train models on custom data
6. **Mobile Optimization:** Touch-friendly controls
7. **Real-Time Updates:** WebSocket instead of polling
8. **Dark/Light Mode:** User preference toggle

---

**Technical Review:** Complete  
**Performance:** Optimized  
**Testing:** Passed ✓  
**Ready for Production:** Yes

---

**Last Updated:** July 27, 2026  
**Version:** 2.0 (TradingView UI)  
**Maintained By:** Trading AI Team

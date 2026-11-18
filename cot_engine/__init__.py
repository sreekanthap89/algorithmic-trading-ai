"""
cot_engine — Larry Williams Commitments of Traders (COT) Algorithmic Engine
Based on 'Trade Stocks and Commodities With the Insiders: Secrets of the COT Report' (Wiley, 2005).

Modules:
  - cot_loader: CFTC data ingestion, market mapping, point-in-time as-of alignment.
  - indicators: COT Index, WILLCO, Multi-Year Extremes, ProGo, Inside-OI Matrix, Public & Large Speculator Indexes,
    Aggregated Dollar Commercial Position, Synthetic Commercial Proxy.
  - strategies: OneMinuteWeeklyScan, UnifiedCOTStrategy, TripartiteMacroModel, DirectionalRegimeFilter,
    ConsecutiveDownClosesStrategy, Gold/USD Valuation, Gold Seasonal Cycle, Classical Chart Patterns & 1-2-3 Reversals.
  - risk_rules: DollarRiskStop, 17-Day Trailing Stop (excluding inside days), 13-Day Re-Entry, Time Exit,
    Parabolic Windfall Exit, Anti-Plunging Position Sizer, Flag Bracket Execution.
  - backtester: Historical simulation engine with Monday-open execution and performance metrics.
"""

from .cot_loader import (
    MARKET_MAP,
    download_cftc_year,
    load_cftc_history,
    filter_market_cot,
    merge_cot_with_price,
    generate_synthetic_cot,
    validate_cftc_identities,
    compute_cftc_release_time,
)
from .indicators import (
    commercial_net_position,
    commercial_share_oi,
    commercial_net_flow,
    commercial_weekly_flow,
    williams_cot_index,
    williams_willco,
    multi_year_extremes,
    perpetual_short_regime_breakout,
    cot_swing_velocity,
    public_contrary_index,
    large_trader_index,
    open_interest_stochastic,
    inside_oi_regime,
    progo_oscillator,
    jiler_seasonal_deviation,
    compute_all_cot_features,
    DEFAULT_INDEX_MULTIPLIERS,
    aggregated_dollar_commercial_position,
    synthetic_commercial_proxy,
)
from .strategies import (
    OneMinuteWeeklyScan,
    UnifiedCOTStrategy,
    TripartiteMacroModel,
    DirectionalRegimeFilter,
    consecutive_down_closes_pattern,
    ConsecutiveDownClosesStrategy,
    intermarket_gold_dollar_valuation,
    gold_usd_valuation_signals,
    gold_seasonal_cycle,
    detect_123_reversal,
    classical_chart_breakouts,
)
from .risk_rules import (
    DollarRiskStop,
    seventeen_day_trailing_stop,
    thirteen_day_reentry_levels,
    TimeExitFilter,
    ParabolicWindfallExit,
    AntiPlungingPositionSizer,
    FlagBracketExecution,
    is_inside_day,
)
from .backtester import (
    COTBacktester,
    TradeRecord,
)

__all__ = [
    # Loader
    "MARKET_MAP",
    "download_cftc_year",
    "load_cftc_history",
    "filter_market_cot",
    "merge_cot_with_price",
    "generate_synthetic_cot",
    "validate_cftc_identities",
    "compute_cftc_release_time",
    # Indicators
    "commercial_net_position",
    "commercial_share_oi",
    "commercial_net_flow",
    "commercial_weekly_flow",
    "williams_cot_index",
    "williams_willco",
    "multi_year_extremes",
    "perpetual_short_regime_breakout",
    "cot_swing_velocity",
    "public_contrary_index",
    "large_trader_index",
    "open_interest_stochastic",
    "inside_oi_regime",
    "progo_oscillator",
    "jiler_seasonal_deviation",
    "compute_all_cot_features",
    "DEFAULT_INDEX_MULTIPLIERS",
    "aggregated_dollar_commercial_position",
    "synthetic_commercial_proxy",
    # Strategies
    "OneMinuteWeeklyScan",
    "UnifiedCOTStrategy",
    "TripartiteMacroModel",
    "DirectionalRegimeFilter",
    "consecutive_down_closes_pattern",
    "ConsecutiveDownClosesStrategy",
    "intermarket_gold_dollar_valuation",
    "gold_usd_valuation_signals",
    "gold_seasonal_cycle",
    "detect_123_reversal",
    "classical_chart_breakouts",
    # Risk
    "DollarRiskStop",
    "seventeen_day_trailing_stop",
    "thirteen_day_reentry_levels",
    "TimeExitFilter",
    "ParabolicWindfallExit",
    "AntiPlungingPositionSizer",
    "FlagBracketExecution",
    "is_inside_day",
    # Backtester
    "COTBacktester",
    "TradeRecord",
]

__version__ = "1.0.0"

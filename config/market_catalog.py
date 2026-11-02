"""
=============================================================================
 UNIVERSAL MULTI-ASSET CATALOG, SMART SYMBOL RESOLVER & PIP/LOT CALCULATOR
 Supports Commodities (Natural Gas, Gold, Silver, Crude Oil, Copper),
 Forex (Major & Cross Pairs), Crypto (Bitcoin, Ethereum, Solana, etc.),
 and Global Equities / Indices.
=============================================================================
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any


# Organized by Asset Class for 1-click selection in the UI
ASSET_CLASS_CATALOG: Dict[str, Dict[str, Dict[str, Any]]] = {
    "🔥 Commodities (Natural Gas, Gold, Oil)": {
        "🔥 Natural Gas (NG=F)": {
            "symbol": "NG=F",
            "name": "Henry Hub Natural Gas Futures",
            "asset_class": "Commodity (Energy)",
            "unit_label": "MMBtu",
            "contract_size": 10000,
            "currency_prefix": "$",
            "macro_drivers": [
                ("CL=F", "WTI Crude Oil"),
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
                ("XLE", "Energy Select Sector ETF"),
            ],
        },
        "🥇 Gold (GC=F)": {
            "symbol": "GC=F",
            "name": "COMEX Gold Futures (XAU/USD)",
            "asset_class": "Commodity (Precious Metal)",
            "unit_label": "Troy Oz",
            "contract_size": 100,
            "currency_prefix": "$",
            "macro_drivers": [
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
                ("SI=F", "Silver Futures"),
                ("^TNX", "US 10Y Treasury Yield"),
            ],
        },
        "🥈 Silver (SI=F)": {
            "symbol": "SI=F",
            "name": "COMEX Silver Futures (XAG/USD)",
            "asset_class": "Commodity (Precious Metal)",
            "unit_label": "Troy Oz",
            "contract_size": 5000,
            "currency_prefix": "$",
            "macro_drivers": [
                ("GC=F", "Gold Futures"),
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
                ("HG=F", "Copper Futures"),
            ],
        },
        "🛢️ WTI Crude Oil (CL=F)": {
            "symbol": "CL=F",
            "name": "NYMEX WTI Crude Oil Futures",
            "asset_class": "Commodity (Energy)",
            "unit_label": "Barrels",
            "contract_size": 1000,
            "currency_prefix": "$",
            "macro_drivers": [
                ("BZ=F", "Brent Crude Oil"),
                ("NG=F", "Natural Gas"),
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
            ],
        },
        "🛢️ Brent Crude Oil (BZ=F)": {
            "symbol": "BZ=F",
            "name": "ICE Brent Crude Oil Futures",
            "asset_class": "Commodity (Energy)",
            "unit_label": "Barrels",
            "contract_size": 1000,
            "currency_prefix": "$",
            "macro_drivers": [
                ("CL=F", "WTI Crude Oil"),
                ("NG=F", "Natural Gas"),
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
            ],
        },
        "🥉 Copper (HG=F)": {
            "symbol": "HG=F",
            "name": "COMEX High-Grade Copper Futures",
            "asset_class": "Commodity (Industrial Metal)",
            "unit_label": "Lbs",
            "contract_size": 25000,
            "currency_prefix": "$",
            "macro_drivers": [
                ("GC=F", "Gold Futures"),
                ("SI=F", "Silver Futures"),
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
            ],
        },
    },
    "💱 Forex (EUR/USD, GBP/USD, USD/JPY, INR)": {
        "🇪🇺/🇺🇸 EUR/USD (EURUSD=X)": {
            "symbol": "EURUSD=X",
            "name": "Euro / US Dollar Spot",
            "asset_class": "Forex (Major)",
            "unit_label": "Units (EUR)",
            "pip_size": 0.0001,
            "currency_prefix": "",
            "macro_drivers": [
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
                ("GBPUSD=X", "GBP/USD Spot"),
                ("GC=F", "Gold (XAU/USD)"),
            ],
        },
        "🇬🇧/🇺🇸 GBP/USD (GBPUSD=X)": {
            "symbol": "GBPUSD=X",
            "name": "British Pound / US Dollar Spot",
            "asset_class": "Forex (Major)",
            "unit_label": "Units (GBP)",
            "pip_size": 0.0001,
            "currency_prefix": "",
            "macro_drivers": [
                ("EURUSD=X", "EUR/USD Spot"),
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
                ("GC=F", "Gold (XAU/USD)"),
            ],
        },
        "🇺🇸/🇯🇵 USD/JPY (USDJPY=X)": {
            "symbol": "USDJPY=X",
            "name": "US Dollar / Japanese Yen Spot",
            "asset_class": "Forex (Major)",
            "unit_label": "Units (USD)",
            "pip_size": 0.01,
            "currency_prefix": "¥",
            "macro_drivers": [
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
                ("^TNX", "US 10Y Treasury Yield"),
                ("GC=F", "Gold (XAU/USD)"),
            ],
        },
        "🇺🇸/🇮🇳 USD/INR (USDINR=X)": {
            "symbol": "USDINR=X",
            "name": "US Dollar / Indian Rupee Spot",
            "asset_class": "Forex (Emerging)",
            "unit_label": "Units (USD)",
            "pip_size": 0.01,
            "currency_prefix": "₹",
            "macro_drivers": [
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
                ("CL=F", "Crude Oil (Import Driver)"),
                ("^NSEI", "Nifty 50 Index"),
            ],
        },
        "🇦🇺/🇺🇸 AUD/USD (AUDUSD=X)": {
            "symbol": "AUDUSD=X",
            "name": "Australian Dollar / US Dollar Spot",
            "asset_class": "Forex (Commodity Major)",
            "unit_label": "Units (AUD)",
            "pip_size": 0.0001,
            "currency_prefix": "",
            "macro_drivers": [
                ("GC=F", "Gold Futures"),
                ("HG=F", "Copper Futures"),
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
            ],
        },
        "🇺🇸/🇨🇦 USD/CAD (USDCAD=X)": {
            "symbol": "USDCAD=X",
            "name": "US Dollar / Canadian Dollar Spot",
            "asset_class": "Forex (Petro-Currency)",
            "unit_label": "Units (USD)",
            "pip_size": 0.0001,
            "currency_prefix": "",
            "macro_drivers": [
                ("CL=F", "WTI Crude Oil"),
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
                ("NG=F", "Natural Gas"),
            ],
        },
        "🇺🇸/🇨🇭 USD/CHF (USDCHF=X)": {
            "symbol": "USDCHF=X",
            "name": "US Dollar / Swiss Franc Spot",
            "asset_class": "Forex (Safe Haven)",
            "unit_label": "Units (USD)",
            "pip_size": 0.0001,
            "currency_prefix": "",
            "macro_drivers": [
                ("EURUSD=X", "EUR/USD Spot"),
                ("GC=F", "Gold Futures"),
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
            ],
        },
    },
    "₿ Crypto (Bitcoin, Ethereum, Solana, XRP)": {
        "₿ Bitcoin (BTC-USD)": {
            "symbol": "BTC-USD",
            "name": "Bitcoin / US Dollar",
            "asset_class": "Cryptocurrency",
            "unit_label": "BTC",
            "currency_prefix": "$",
            "macro_drivers": [
                ("ETH-USD", "Ethereum (ETH)"),
                ("QQQ", "Nasdaq 100 Tech ETF"),
                ("GC=F", "Gold (Digital Gold Proxy)"),
            ],
        },
        "Ξ Ethereum (ETH-USD)": {
            "symbol": "ETH-USD",
            "name": "Ethereum / US Dollar",
            "asset_class": "Cryptocurrency",
            "unit_label": "ETH",
            "currency_prefix": "$",
            "macro_drivers": [
                ("BTC-USD", "Bitcoin (BTC)"),
                ("SOL-USD", "Solana (SOL)"),
                ("QQQ", "Nasdaq 100 Tech ETF"),
            ],
        },
        "◎ Solana (SOL-USD)": {
            "symbol": "SOL-USD",
            "name": "Solana / US Dollar",
            "asset_class": "Cryptocurrency",
            "unit_label": "SOL",
            "currency_prefix": "$",
            "macro_drivers": [
                ("BTC-USD", "Bitcoin (BTC)"),
                ("ETH-USD", "Ethereum (ETH)"),
                ("QQQ", "Nasdaq 100 Tech ETF"),
            ],
        },
        "✕ XRP (XRP-USD)": {
            "symbol": "XRP-USD",
            "name": "Ripple XRP / US Dollar",
            "asset_class": "Cryptocurrency",
            "unit_label": "XRP",
            "currency_prefix": "$",
            "macro_drivers": [
                ("BTC-USD", "Bitcoin (BTC)"),
                ("ETH-USD", "Ethereum (ETH)"),
                ("SOL-USD", "Solana (SOL)"),
            ],
        },
        "🐕 Dogecoin (DOGE-USD)": {
            "symbol": "DOGE-USD",
            "name": "Dogecoin / US Dollar",
            "asset_class": "Cryptocurrency",
            "unit_label": "DOGE",
            "currency_prefix": "$",
            "macro_drivers": [
                ("BTC-USD", "Bitcoin (BTC)"),
                ("ETH-USD", "Ethereum (ETH)"),
                ("SOL-USD", "Solana (SOL)"),
            ],
        },
    },
    "📊 Equities & Global Indices (S&P 500, Nifty, Tech)": {
        "📊 S&P 500 ETF (SPY)": {
            "symbol": "SPY",
            "name": "SPDR S&P 500 ETF Trust",
            "asset_class": "Equity Index",
            "unit_label": "Shares",
            "currency_prefix": "$",
            "macro_drivers": [
                ("QQQ", "Nasdaq 100 ETF"),
                ("^TNX", "US 10Y Treasury Yield"),
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
            ],
        },
        "💻 Nasdaq 100 ETF (QQQ)": {
            "symbol": "QQQ",
            "name": "Invesco QQQ Nasdaq 100",
            "asset_class": "Equity Index",
            "unit_label": "Shares",
            "currency_prefix": "$",
            "macro_drivers": [
                ("SPY", "S&P 500 ETF"),
                ("NVDA", "NVIDIA Corp"),
                ("^TNX", "US 10Y Treasury Yield"),
            ],
        },
        "🇮🇳 Nifty 50 Index (^NSEI)": {
            "symbol": "^NSEI",
            "name": "NSE NIFTY 50 Index (India)",
            "asset_class": "Equity Index",
            "unit_label": "Index Units",
            "currency_prefix": "₹",
            "macro_drivers": [
                ("SPY", "US S&P 500 ETF"),
                ("USDINR=X", "USD/INR Spot"),
                ("CL=F", "WTI Crude Oil"),
            ],
        },
        "🟢 NVIDIA (NVDA)": {
            "symbol": "NVDA",
            "name": "NVIDIA Corporation",
            "asset_class": "US Equity",
            "unit_label": "Shares",
            "currency_prefix": "$",
            "macro_drivers": [
                ("QQQ", "Nasdaq 100 ETF"),
                ("SPY", "S&P 500 ETF"),
                ("BTC-USD", "Bitcoin Risk Proxy"),
            ],
        },
        "🍎 Apple (AAPL)": {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "asset_class": "US Equity",
            "unit_label": "Shares",
            "currency_prefix": "$",
            "macro_drivers": [
                ("QQQ", "Nasdaq 100 ETF"),
                ("SPY", "S&P 500 ETF"),
                ("^TNX", "US 10Y Treasury Yield"),
            ],
        },
        "⚡ Tesla (TSLA)": {
            "symbol": "TSLA",
            "name": "Tesla, Inc.",
            "asset_class": "US Equity",
            "unit_label": "Shares",
            "currency_prefix": "$",
            "macro_drivers": [
                ("QQQ", "Nasdaq 100 ETF"),
                ("SPY", "S&P 500 ETF"),
                ("BTC-USD", "Bitcoin Risk Proxy"),
            ],
        },
    },
}

# Flat lookup by symbol
SYMBOL_METADATA: Dict[str, Dict[str, Any]] = {}
for _cat, _items in ASSET_CLASS_CATALOG.items():
    for _label, _meta in _items.items():
        SYMBOL_METADATA[_meta["symbol"]] = _meta

# Natural language aliases so users can type "natural gas", "gold", "eur/usd", "bitcoin", etc.
NATURAL_LANGUAGE_ALIASES: Dict[str, str] = {
    "NATURAL GAS": "NG=F",
    "NATGAS": "NG=F",
    "GAS": "NG=F",
    "NG": "NG=F",
    "NG=F": "NG=F",
    "GOLD": "GC=F",
    "XAUUSD": "GC=F",
    "XAU/USD": "GC=F",
    "GC": "GC=F",
    "GC=F": "GC=F",
    "SILVER": "SI=F",
    "XAGUSD": "SI=F",
    "XAG/USD": "SI=F",
    "SI": "SI=F",
    "SI=F": "SI=F",
    "OIL": "CL=F",
    "CRUDE": "CL=F",
    "CRUDE OIL": "CL=F",
    "WTI": "CL=F",
    "USOIL": "CL=F",
    "CL": "CL=F",
    "CL=F": "CL=F",
    "BRENT": "BZ=F",
    "UKOIL": "BZ=F",
    "BZ=F": "BZ=F",
    "COPPER": "HG=F",
    "HG=F": "HG=F",
    "EURUSD": "EURUSD=X",
    "EUR/USD": "EURUSD=X",
    "EURO": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "GBP/USD": "GBPUSD=X",
    "CABLE": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USD/JPY": "USDJPY=X",
    "JPY": "USDJPY=X",
    "USDINR": "USDINR=X",
    "USD/INR": "USDINR=X",
    "INR": "USDINR=X",
    "AUDUSD": "AUDUSD=X",
    "AUD/USD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USD/CAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "USD/CHF": "USDCHF=X",
    "BITCOIN": "BTC-USD",
    "BTC": "BTC-USD",
    "BTCUSD": "BTC-USD",
    "BTC/USD": "BTC-USD",
    "ETHEREUM": "ETH-USD",
    "ETH": "ETH-USD",
    "ETHUSD": "ETH-USD",
    "ETH/USD": "ETH-USD",
    "SOLANA": "SOL-USD",
    "SOL": "SOL-USD",
    "SOLUSD": "SOL-USD",
    "XRP": "XRP-USD",
    "RIPPLE": "XRP-USD",
    "DOGE": "DOGE-USD",
    "DOGECOIN": "DOGE-USD",
    "NIFTY": "^NSEI",
    "NIFTY 50": "^NSEI",
    "NIFTY50": "^NSEI",
    "SP500": "SPY",
    "S&P 500": "SPY",
    "NASDAQ": "QQQ",
    "DXY": "DX-Y.NYB",
}


def resolve_symbol(user_query: str) -> str:
    """
    Resolves natural-language names ('natural gas', 'gold', 'eur/usd', 'bitcoin')
    or standard tickers to their canonical Yahoo Finance symbol.
    """
    if not user_query:
        return "BTC-USD"
    clean = user_query.strip().upper()
    if clean in NATURAL_LANGUAGE_ALIASES:
        return NATURAL_LANGUAGE_ALIASES[clean]
    # Check 6-letter forex pair format like EURGBP or AUDJPY
    if len(clean) == 6 and clean.isalpha() and clean[:3] in ("EUR", "GBP", "USD", "AUD", "NZD", "CAD", "CHF"):
        return f"{clean}=X"
    return clean


def get_asset_metadata(symbol: str) -> Dict[str, Any]:
    """
    Returns full metadata (name, asset_class, unit_label, macro_drivers, currency_prefix)
    for any symbol, inferring sensible defaults for custom tickers.
    """
    sym = symbol.upper().strip()
    if sym in SYMBOL_METADATA:
        return SYMBOL_METADATA[sym]

    if sym.endswith("=X"):
        return {
            "symbol": sym,
            "name": f"{sym[:-2]} Forex Spot Pair",
            "asset_class": "Forex",
            "unit_label": "Units",
            "pip_size": 0.01 if "JPY" in sym or "INR" in sym else 0.0001,
            "currency_prefix": "",
            "macro_drivers": [
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
                ("GC=F", "Gold Futures"),
                ("^TNX", "US 10Y Treasury Yield"),
            ],
        }
    if sym.endswith("=F"):
        return {
            "symbol": sym,
            "name": f"{sym} Commodity Futures",
            "asset_class": "Commodity Futures",
            "unit_label": "Contracts",
            "currency_prefix": "$",
            "macro_drivers": [
                ("DX-Y.NYB", "US Dollar Index (DXY)"),
                ("CL=F", "WTI Crude Oil"),
                ("GC=F", "Gold Futures"),
            ],
        }
    if sym.endswith("-USD"):
        return {
            "symbol": sym,
            "name": f"{sym} Crypto Pair",
            "asset_class": "Cryptocurrency",
            "unit_label": sym.split("-")[0],
            "currency_prefix": "$",
            "macro_drivers": [
                ("BTC-USD", "Bitcoin (BTC)"),
                ("ETH-USD", "Ethereum (ETH)"),
                ("QQQ", "Nasdaq 100 ETF"),
            ],
        }
    return {
        "symbol": sym,
        "name": f"{sym} Market Asset",
        "asset_class": "Equity / Market",
        "unit_label": "Shares",
        "currency_prefix": "₹" if sym.endswith(".NS") or sym.endswith(".BO") else "$",
        "macro_drivers": [
            ("SPY", "S&P 500 ETF"),
            ("QQQ", "Nasdaq 100 ETF"),
            ("DX-Y.NYB", "US Dollar Index (DXY)"),
        ],
    }


def get_price_decimals(price: float, symbol: str = "") -> int:
    """
    Determines the right number of decimal places so Forex (1.0845),
    Natural Gas ($2.6450), Silver ($32.450), and Bitcoin ($77,466.09)
    never lose precision.
    """
    sym = (symbol or "").upper()
    if sym.endswith("=X"):
        if "JPY" in sym or "INR" in sym:
            return 3
        return 5
    if sym == "NG=F":
        return 4
    abs_p = abs(float(price)) if price is not None else 100.0
    if abs_p == 0:
        return 2
    if abs_p < 1.0:
        return 5
    if abs_p < 20.0:
        return 4
    if abs_p < 200.0:
        return 3
    return 2


def smart_round(price: float, symbol: str = "") -> float:
    """Rounds a price preserving asset-appropriate tick/pip precision."""
    if price is None:
        return 0.0
    dec = get_price_decimals(price, symbol)
    return round(float(price), dec)


def format_price(price: float, symbol: str = "") -> str:
    """Formats a price string with appropriate currency prefix and decimal precision."""
    if price is None:
        return "—"
    meta = get_asset_metadata(symbol)
    prefix = meta.get("currency_prefix", "$")
    dec = get_price_decimals(price, symbol)
    return f"{prefix}{float(price):,.{dec}f}"


def compute_asset_position_Breakdown(
    symbol: str,
    entry: float,
    stop_loss: float,
    pos_size_usd: float,
) -> Dict[str, Any]:
    """
    Translates a USD position allocation into asset-specific trading units:
      - Forex: Standard Lots (100k), Mini Lots (10k), and Pip distance to Stop-Loss
      - Commodities (Natural Gas, Gold, Oil): Units (MMBtu, Troy Oz, Barrels) + Futures Contracts
      - Crypto & Equities: Exact fractional coins or shares
    """
    meta = get_asset_metadata(symbol)
    unit_label = meta.get("unit_label", "Units")
    raw_units = (pos_size_usd / entry) if entry > 0 else 0.0
    price_diff = abs(entry - stop_loss)

    if meta["asset_class"].startswith("Forex"):
        pip_size = float(meta.get("pip_size", 0.0001))
        pips_risk = price_diff / pip_size if pip_size > 0 else 0.0
        std_lots = pos_size_usd / 100000.0
        mini_lots = pos_size_usd / 10000.0
        return {
            "primary_size_str": f"{std_lots:.2f} Std Lots ({mini_lots:.1f} Mini)",
            "secondary_info_str": f"Stop Distance: {pips_risk:.1f} Pips · {raw_units:,.0f} {unit_label}",
            "recommended_qty": round(max(raw_units, 1.0), 2),
        }

    if "Commodity" in meta["asset_class"]:
        contract_sz = float(meta.get("contract_size", 100.0))
        contracts = raw_units / contract_sz if contract_sz > 0 else 0.0
        return {
            "primary_size_str": f"{raw_units:,.1f} {unit_label}",
            "secondary_info_str": f"≈ {contracts:.2f} Futures Contracts (or CFD equivalent)",
            "recommended_qty": round(max(raw_units, 0.01), 3),
        }

    return {
        "primary_size_str": f"{raw_units:,.4f} {unit_label}",
        "secondary_info_str": f"Notional Value: ${pos_size_usd:,.2f} (2% Max Risk)",
        "recommended_qty": round(max(raw_units, 0.0001), 4),
    }


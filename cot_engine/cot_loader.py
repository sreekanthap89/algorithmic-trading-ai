"""
cot_engine/cot_loader.py — CFTC Commitments of Traders (COT) Data Ingestion & ETL Engine
Supports Legacy format (Futures-Only & Combined), caching, market resolution,
point-in-time (PIT) merger to eliminate lookahead bias, accounting identity validation,
and offline synthetic test generation.
"""

import os
import io
import re
import urllib.request
import zipfile
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join("data", "cot")

# Mapping from canonical futures/stock symbols to CFTC Contract Market Codes and matching market names.
# Includes both modern CFTC contract names (2020+) and legacy historical names (1986-2019).
MARKET_MAP = {
    "GC=F": [
        "088691",
        "GOLD - COMMODITY EXCHANGE INC.",
        "GOLD",
    ],
    "SI=F": [
        "084691",
        "SILVER - COMMODITY EXCHANGE INC.",
        "SILVER",
    ],
    "HG=F": [
        "085692",
        "COPPER- #1 - COMMODITY EXCHANGE INC.",
        "COPPER-GRADE #1 - COMMODITY EXCHANGE INC.",
        "COPPER",
    ],
    "CL=F": [
        "067651",
        "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE",
        "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
        "LIGHT SWEET CRUDE OIL",
    ],
    "NG=F": [
        "023651",
        "NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE",
        "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE",
        "NATURAL GAS",
    ],
    "ZC=F": [
        "002602",
        "CORN - CHICAGO BOARD OF TRADE",
        "CORN",
    ],
    "ZW=F": [
        "001602",
        "WHEAT-SRW - CHICAGO BOARD OF TRADE",
        "WHEAT - CHICAGO BOARD OF TRADE",
        "WHEAT",
    ],
    "ZS=F": [
        "005602",
        "SOYBEANS - CHICAGO BOARD OF TRADE",
        "SOYBEANS",
    ],
    "KC=F": [
        "083731",
        "COFFEE C - ICE FUTURES U.S.",
        "COFFEE C",
    ],
    "ZB=F": [
        "020601",
        "UST BOND - CHICAGO BOARD OF TRADE",
        "U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE",
        "30-YEAR U.S. TREASURY BONDS",
    ],
    "ZN=F": [
        "043602",
        "UST 10Y NOTE - CHICAGO BOARD OF TRADE",
        "10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
        "10-YEAR U.S. TREASURY",
    ],
    "ZF=F": [
        "044601",
        "UST 5Y NOTE - CHICAGO BOARD OF TRADE",
        "5-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
    ],
    "ZT=F": [
        "042601",
        "UST 2Y NOTE - CHICAGO BOARD OF TRADE",
        "2-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",
    ],
    "ES=F": [
        "13874+",
        "13874A",
        "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE",
        "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
        "E-MINI S&P 500 STOCK INDEX",
        "S&P 500",
    ],
    "NQ=F": [
        "20974+",
        "209742",
        "NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE",
        "E-MINI NASDAQ-100",
        "NASDAQ-100",
    ],
    "YM=F": [
        "12460+",
        "124603",
        "DJIA Consolidated - CHICAGO BOARD OF TRADE",
        "MINI-SIZED DOW",
    ],
    "6J=F": [
        "097741",
        "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
        "JAPANESE YEN",
    ],
    "6B=F": [
        "096742",
        "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE",
        "BRITISH POUND STERLING",
    ],
    "6E=F": [
        "099741",
        "EURO FX - CHICAGO MERCANTILE EXCHANGE",
        "EURO FX",
    ],
    "6A=F": [
        "232741",
        "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    ],
    "6C=F": [
        "090741",
        "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    ],
    "6S=F": [
        "092741",
        "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE",
    ],
}

CFTC_COLUMN_MAP = {
    "Market and Exchange Names": "market_name",
    "As of Date in Form YYYY-MM-DD": "report_date",
    "As of Date in Form YYMMDD": "report_date_raw",
    "CFTC Contract Market Code": "cftc_code",
    "Open Interest (All)": "open_interest",
    "Commercial Positions-Long (All)": "comm_long",
    "Commercial Positions-Short (All)": "comm_short",
    "Noncommercial Positions-Long (All)": "noncomm_long",
    "Noncommercial Positions-Short (All)": "noncomm_short",
    "Noncommercial Positions-Spreading (All)": "noncomm_spread",
    "Total Reportable Positions-Long (All)": "rep_long",
    "Total Reportable Positions-Short (All)": "rep_short",
    "Nonreportable Positions-Long (All)": "nonrep_long",
    "Nonreportable Positions-Short (All)": "nonrep_short",
}


def _ensure_cache_dir(cache_dir: str = CACHE_DIR) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_easter_date(year: int) -> date:
    """
    Computes Easter Sunday using the Anonymous Gregorian algorithm.
    Used for determining Good Friday (CFTC holiday).
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def is_us_federal_or_cftc_holiday(d: date) -> bool:
    """
    Determines if a date is a US Federal holiday or CFTC holiday schedule day
    that delays the CFTC weekly report release.
    """
    year = d.year

    # New Year's Day (Jan 1, or observed Fri Dec 31 if Jan 1 is Sat, or Mon Jan 2 if Jan 1 is Sun)
    if d.month == 1 and d.day == 1:
        return True
    if d.month == 12 and d.day == 31 and d.weekday() == 4:
        return True
    if d.month == 1 and d.day == 2 and d.weekday() == 0:
        return True

    # MLK Day: 3rd Monday of Jan
    if d.month == 1 and d.weekday() == 0 and 15 <= d.day <= 21:
        return True

    # Washington's Birthday (Presidents' Day): 3rd Monday of Feb
    if d.month == 2 and d.weekday() == 0 and 15 <= d.day <= 21:
        return True

    # Good Friday (Friday before Easter)
    easter = get_easter_date(year)
    good_friday = easter - timedelta(days=2)
    if d == good_friday:
        return True

    # Memorial Day: last Monday of May
    if d.month == 5 and d.weekday() == 0 and d.day >= 25:
        return True

    # Juneteenth: June 19 (since 2021)
    if year >= 2021:
        if d.month == 6 and d.day == 19:
            return True
        if d.month == 6 and d.day == 18 and d.weekday() == 4:
            return True
        if d.month == 6 and d.day == 20 and d.weekday() == 0:
            return True

    # Independence Day: July 4
    if d.month == 7 and d.day == 4:
        return True
    if d.month == 7 and d.day == 3 and d.weekday() == 4:
        return True
    if d.month == 7 and d.day == 5 and d.weekday() == 0:
        return True

    # Labor Day: 1st Monday of Sept
    if d.month == 9 and d.weekday() == 0 and d.day <= 7:
        return True

    # Columbus Day: 2nd Monday of Oct
    if d.month == 10 and d.weekday() == 0 and 8 <= d.day <= 14:
        return True

    # Veterans Day: Nov 11
    if d.month == 11 and d.day == 11:
        return True
    if d.month == 11 and d.day == 10 and d.weekday() == 4:
        return True
    if d.month == 11 and d.day == 12 and d.weekday() == 0:
        return True

    # Thanksgiving Day: 4th Thursday of Nov
    if d.month == 11 and d.weekday() == 3 and 22 <= d.day <= 28:
        return True

    # Day after Thanksgiving (Friday): CFTC staff holiday, release slides to Monday
    if d.month == 11 and d.weekday() == 4 and 23 <= d.day <= 29:
        return True

    # Christmas Day: Dec 25
    if d.month == 12 and d.day == 25:
        return True
    if d.month == 12 and d.day == 24 and d.weekday() == 4:
        return True
    if d.month == 12 and d.day == 26 and d.weekday() == 0:
        return True

    return False


def compute_cftc_release_time(report_date: Union[date, datetime, str]) -> datetime:
    """
    Computes the Point-In-Time (PIT) publication timestamp (15:30 US/Eastern) for a given COT snapshot.
    Accounts for:
      - Tuesday snapshot -> Friday 15:30 ET release
      - Non-Tuesday snapshot (e.g. Monday snapshot before holiday) -> Friday 15:30 ET release
      - Friday holidays & Thanksgiving Friday -> release slides to next business day (Monday 15:30 ET).
    """
    if isinstance(report_date, str):
        report_dt = pd.to_datetime(report_date).date()
    elif isinstance(report_date, datetime):
        report_dt = report_date.date()
    else:
        report_dt = report_date

    # Scheduled Friday of the report week (weekday 4)
    days_to_friday = (4 - report_dt.weekday()) % 7
    target_date = report_dt + timedelta(days=days_to_friday)

    # If target Friday is a holiday or day after Thanksgiving, slide forward to next business day
    while is_us_federal_or_cftc_holiday(target_date) or target_date.weekday() >= 5:
        target_date += timedelta(days=1)

    return datetime(target_date.year, target_date.month, target_date.day, 15, 30)


def validate_cftc_identities(df: pd.DataFrame, max_tolerance: float = 1.0) -> pd.DataFrame:
    """
    Validates CFTC accounting balance identities across records according to DOCS/COT_Master_Specification.md §1.3:
      1. rep_long == comm_long + noncomm_long + noncomm_spread (within tolerance)
      2. rep_short == comm_short + noncomm_short + noncomm_spread (within tolerance)
      3. nonrep_long == open_interest - rep_long (within tolerance)
      4. nonrep_short == open_interest - rep_short (within tolerance)
      5. total longs == total shorts == open_interest (within tolerance)
      6. Net_comm + Net_noncomm + Net_nonrep == 0 (within 2 * tolerance)
    Rejects rows with open_interest <= 0 or identity violations > max_tolerance.
    """
    if df is None or len(df) == 0:
        return df

    req_cols = [
        "open_interest", "comm_long", "comm_short",
        "noncomm_long", "noncomm_short", "noncomm_spread",
        "rep_long", "rep_short", "nonrep_long", "nonrep_short"
    ]
    for c in req_cols:
        if c not in df.columns:
            return df  # Skip if required numeric columns not present

    valid = df["open_interest"] > 0

    # 1. Reportable positions identity
    expected_rep_long = df["comm_long"] + df["noncomm_long"] + df["noncomm_spread"]
    valid = valid & ((df["rep_long"] - expected_rep_long).abs() <= max_tolerance)

    expected_rep_short = df["comm_short"] + df["noncomm_short"] + df["noncomm_spread"]
    valid = valid & ((df["rep_short"] - expected_rep_short).abs() <= max_tolerance)

    # 2. Non-reportable positions identity
    expected_nonrep_long = df["open_interest"] - df["rep_long"]
    valid = valid & ((df["nonrep_long"] - expected_nonrep_long).abs() <= max_tolerance)

    expected_nonrep_short = df["open_interest"] - df["rep_short"]
    valid = valid & ((df["nonrep_short"] - expected_nonrep_short).abs() <= max_tolerance)

    # 3. Total balance: Total longs == Total shorts == Open Interest
    total_long = df["rep_long"] + df["nonrep_long"]
    total_short = df["rep_short"] + df["nonrep_short"]
    valid = valid & ((total_long - df["open_interest"]).abs() <= max_tolerance)
    valid = valid & ((total_short - df["open_interest"]).abs() <= max_tolerance)

    # 4. Net sum across groups == 0 (spreading is symmetric)
    net_comm = df["comm_long"] - df["comm_short"]
    net_noncomm = df["noncomm_long"] - df["noncomm_short"]
    net_nonrep = df["nonrep_long"] - df["nonrep_short"]
    valid = valid & ((net_comm + net_noncomm + net_nonrep).abs() <= (2 * max_tolerance))

    return df[valid].copy()


def download_cftc_year(
    year: int,
    report_type: str = "futures_only",
    cache_dir: str = CACHE_DIR,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Downloads and caches the official CFTC Legacy annual zip file for a given year.
    report_type: 'futures_only' (deacot{year}.zip) or 'combined' (deacom{year}.zip)
    Validates CFTC accounting identities and computes holiday-aware PIT release timestamps.
    """
    _ensure_cache_dir(cache_dir)
    file_prefix = "deacot" if report_type == "futures_only" else "deacom"
    cache_path = os.path.join(cache_dir, f"{file_prefix}_{year}.parquet")
    csv_fallback_path = os.path.join(cache_dir, f"{file_prefix}_{year}.csv.gz")

    if not force_download:
        if os.path.exists(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass
        if os.path.exists(csv_fallback_path):
            try:
                return pd.read_csv(csv_fallback_path, low_memory=False)
            except Exception:
                pass

    url = f"https://www.cftc.gov/files/dea/history/{file_prefix}{year}.zip"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read()
    except Exception as e:
        raise RuntimeError(f"Failed to download CFTC data for year {year} from {url}: {e}")

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        txt_files = [f for f in z.namelist() if f.endswith(".txt") or f.endswith(".csv")]
        if not txt_files:
            raise ValueError(f"No text file found in CFTC zip for {year}")
        with z.open(txt_files[0]) as f:
            df = pd.read_csv(f, low_memory=False)

    df.columns = [c.strip() for c in df.columns]

    # Standardize column names
    rename_dict = {}
    for src, dst in CFTC_COLUMN_MAP.items():
        if src in df.columns:
            rename_dict[src] = dst
        else:
            matches = [c for c in df.columns if src.lower() == c.lower().strip()]
            if matches:
                rename_dict[matches[0]] = dst
    df = df.rename(columns=rename_dict)

    # Format report_date
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    elif "report_date_raw" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date_raw"].astype(str), format="%y%m%d", errors="coerce")

    # Clean numeric fields
    numeric_cols = [
        "open_interest", "comm_long", "comm_short",
        "noncomm_long", "noncomm_short", "noncomm_spread",
        "rep_long", "rep_short", "nonrep_long", "nonrep_short"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # De-fragment before assigning new columns
    df = df.copy()

    # Calculate publication timestamp: Tuesday snapshot -> Friday 15:30 ET
    # Holiday-aware calculation handles holiday weekend slides (e.g. Thanksgiving, Good Friday)
    df["available_at"] = df["report_date"].apply(
        lambda d: compute_cftc_release_time(d) if pd.notnull(d) else pd.NaT
    )

    # Filter invalid records and validate accounting balance identities
    df = df.dropna(subset=["report_date", "market_name"])
    df = validate_cftc_identities(df, max_tolerance=1.0)

    # Save cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        df.to_csv(csv_fallback_path, index=False, compression="gzip")

    return df


def load_cftc_history(
    start_year: int = 2018,
    end_year: Optional[int] = None,
    report_type: str = "futures_only",
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame:
    """
    Loads multi-year CFTC history into a unified DataFrame.
    """
    current_year = datetime.now().year
    end_year = end_year or current_year
    frames = []

    for year in range(start_year, end_year + 1):
        try:
            df_year = download_cftc_year(year, report_type=report_type, cache_dir=cache_dir)
            frames.append(df_year)
        except Exception as e:
            # Continue on missing future year or network failure
            print(f"[COT Loader] Notice: Could not load year {year}: {e}")

    if not frames:
        return pd.DataFrame()

    full_df = pd.concat(frames, ignore_index=True)
    full_df = full_df.sort_values(by=["report_date", "market_name"]).reset_index(drop=True)
    return full_df


def filter_market_cot(
    cot_df: pd.DataFrame,
    symbol_or_keyword: str,
) -> pd.DataFrame:
    """
    Filters a master COT dataframe for a specific market ticker or keyword.
    Uses MARKET_MAP when available.
    Supports CFTC Contract Market Codes (exact matching) and market names (with precision matching).
    """
    if cot_df is None or len(cot_df) == 0:
        return pd.DataFrame()

    target_keywords = MARKET_MAP.get(symbol_or_keyword, [symbol_or_keyword])

    # 1. Match CFTC Contract Code first if available in dataframe (exact, immune to text renaming/collisions)
    if "cftc_code" in cot_df.columns:
        codes = [str(kw).strip() for kw in target_keywords if re.match(r"^\d{5,6}[\w\+]?$", str(kw).strip())]
        if codes:
            code_mask = cot_df["cftc_code"].astype(str).str.strip().isin(codes)
            matched = cot_df[code_mask].copy()
            if len(matched) > 0:
                matched = matched.sort_values(by=["report_date", "open_interest"], ascending=[True, False])
                matched = matched.drop_duplicates(subset=["report_date"], keep="first").reset_index(drop=True)
                return matched

    # 2. Text matching on market_name
    mask = pd.Series(False, index=cot_df.index)
    for kw in target_keywords:
        # Skip pure numeric codes in text search
        if re.match(r"^\d{5,6}[\w\+]?$", str(kw).strip()):
            continue
        mask = mask | cot_df["market_name"].str.contains(kw, case=False, na=False, regex=False)

    market_df = cot_df[mask].copy()
    if len(market_df) == 0:
        return pd.DataFrame()

    # Deduplicate in case multiple exchange sub-lines match; pick primary largest OI
    market_df = market_df.sort_values(by=["report_date", "open_interest"], ascending=[True, False])
    market_df = market_df.drop_duplicates(subset=["report_date"], keep="first").reset_index(drop=True)
    return market_df


def merge_cot_with_price(
    price_df: pd.DataFrame,
    cot_df: pd.DataFrame,
    price_datetime_col: Optional[str] = None,
    price_tz: Optional[str] = None,
) -> pd.DataFrame:
    """
    Zero-lookahead point-in-time (PIT) merger of weekly COT data with price bars.
    price_df: DataFrame with OHLCV data. Can have DatetimeIndex or a datetime column.
    cot_df: Output from filter_market_cot containing 'available_at' timestamp.
    price_datetime_col: Optional name of the datetime column in price_df.
    price_tz: Optional timezone name of naive price bars (e.g. 'America/New_York', 'UTC').

    Timezone Handling:
      - If price timestamps are timezone-aware (e.g. UTC), available_at is localized to
        'America/New_York' (CFTC release timezone) and converted to UTC, preventing false
        forward lookahead leakage.
      - If price timestamps are timezone-naive, both are aligned to America/New_York naive time.
    """
    if price_df is None or len(price_df) == 0:
        return price_df
    if cot_df is None or len(cot_df) == 0:
        return price_df.copy()

    price = price_df.copy()
    has_datetime_index = isinstance(price.index, pd.DatetimeIndex)

    if has_datetime_index:
        price["_merge_time"] = price.index
    elif price_datetime_col and price_datetime_col in price.columns:
        price["_merge_time"] = pd.to_datetime(price[price_datetime_col])
    else:
        # Check first column for datetime
        first_col = price.columns[0]
        price["_merge_time"] = pd.to_datetime(price[first_col])

    cot = cot_df.copy()
    cot["available_at"] = pd.to_datetime(cot["available_at"])

    # Timezone alignment
    # If price is naive but price_tz was explicitly provided, localize it
    if price["_merge_time"].dt.tz is None and price_tz is not None:
        price["_merge_time"] = price["_merge_time"].dt.tz_localize(price_tz)

    price_has_tz = price["_merge_time"].dt.tz is not None

    if price_has_tz:
        # Price is timezone-aware: convert COT available_at to UTC
        if cot["available_at"].dt.tz is None:
            # COT publication is officially 15:30 US/Eastern (America/New_York)
            cot["_pit_time"] = cot["available_at"].dt.tz_localize("America/New_York").dt.tz_convert("UTC")
        else:
            cot["_pit_time"] = cot["available_at"].dt.tz_convert("UTC")
        price["_pit_time"] = price["_merge_time"].dt.tz_convert("UTC")
    else:
        # Price is timezone-naive: align COT to America/New_York naive representation
        if cot["available_at"].dt.tz is not None:
            cot["_pit_time"] = cot["available_at"].dt.tz_convert("America/New_York").dt.tz_localize(None)
        else:
            cot["_pit_time"] = cot["available_at"]
        price["_pit_time"] = price["_merge_time"]

    cot = cot.sort_values("_pit_time").reset_index(drop=True)
    price = price.sort_values("_pit_time")

    # Point-in-time merge_asof: only matches COT rows released at or before bar timestamp
    merged = pd.merge_asof(
        price,
        cot,
        on="_pit_time",
        direction="backward",
    )

    merged = merged.drop(columns=["_merge_time", "_pit_time"])
    if has_datetime_index:
        merged.index = price.index

    return merged


def generate_synthetic_cot(
    start_date: Union[str, datetime] = "2020-01-01",
    end_date: Union[str, datetime] = "2024-12-31",
    market_name: str = "GOLD - COMMODITY EXCHANGE INC.",
    base_oi: int = 500000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates realistic synthetic weekly COT data strictly adhering to CFTC accounting identities:
      - rep_long = comm_long + noncomm_long + noncomm_spread
      - rep_short = comm_short + noncomm_short + noncomm_spread
      - nonrep_long = open_interest - rep_long
      - nonrep_short = open_interest - rep_short
      - total longs == total shorts == open_interest
      - Net_comm + Net_noncomm + Net_nonrep == 0.
    Used for offline testing, CI/CD, and backtest validation.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="W-TUE")
    n = len(dates)

    t = np.linspace(0, 4 * np.pi, n)
    cycle = np.sin(t) + 0.3 * np.sin(2.5 * t)

    target_oi = base_oi + (rng.normal(0, 20000, n) + cycle * 50000).astype(int)
    target_oi = np.maximum(target_oi, 100000)

    # Commercial net position oscillates cyclically
    comm_net = -180000 + (cycle * 100000 + rng.normal(0, 15000, n)).astype(int)
    comm_short = (target_oi * 0.55 - comm_net / 2).astype(int)
    comm_long = (comm_short + comm_net).astype(int)
    comm_long = np.maximum(comm_long, 10000)
    comm_short = (comm_long - comm_net).astype(int)

    # Large speculators take the other side
    noncomm_net = -comm_net - rng.integers(10000, 30000, n)
    noncomm_long = (target_oi * 0.40 + noncomm_net / 2).astype(int)
    noncomm_short = (noncomm_long - noncomm_net).astype(int)
    noncomm_spread = rng.integers(20000, 60000, n)

    rep_long = comm_long + noncomm_long + noncomm_spread
    rep_short = comm_short + noncomm_short + noncomm_spread

    # Open interest must strictly exceed max(rep_long, rep_short) to ensure positive non-reportables
    min_nonrep = 15000
    oi = np.maximum(target_oi, np.maximum(rep_long, rep_short) + min_nonrep)

    # Non-reportables derived strictly from Open Interest identity
    nonrep_long = oi - rep_long
    nonrep_short = oi - rep_short

    # Holiday-aware release dates
    available_at = [compute_cftc_release_time(d) for d in dates]

    df = pd.DataFrame({
        "market_name": market_name,
        "report_date": dates,
        "available_at": available_at,
        "open_interest": oi,
        "comm_long": comm_long,
        "comm_short": comm_short,
        "noncomm_long": noncomm_long,
        "noncomm_short": noncomm_short,
        "noncomm_spread": noncomm_spread,
        "rep_long": rep_long,
        "rep_short": rep_short,
        "nonrep_long": nonrep_long,
        "nonrep_short": nonrep_short,
    })
    return df

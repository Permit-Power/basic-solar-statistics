"""
average_electricity_prices.py

Computes inflation-adjusted average retail electricity prices ($/kWh) by
utility and by state/sector, using two data sources:

Utility-level (EIA Form 861 annual ZIP files):
    • Downloads f861YYYY.zip from EIA for each requested year.
    • Reads the "States" sheet of Sales_Ult_Cust_YYYY.xlsx.
    • Pulls Residential, Commercial, Industrial revenue / sales / customers.
    • Inflation-adjusts revenues to real dollars via FRED CPI.
    • Aggregates to utility × year × sector.

State-level (EIA API v2 — retail-sales endpoint):
    • Fetches annual customers, price, revenue, and sales for all sectors
      and all states over a configurable year range (default 2015–2026).
    • Converts nominal cents/kWh to real $/kWh using FRED CPI.
    • Consistent methodology across all years (no file-format changes).

Inflation adjustment:
    • Annual-average CPI (CPIAUCSL) from FRED API.
    • Reference year defaults to 2026; partial-year average used if the
      reference year is not yet complete.

The output is two long-format DataFrames:
    (util_prices, state_prices)

Dependencies:
    pandas, requests, openpyxl

Required environment variables:
    FRED_API_KEY   — St. Louis Fed FRED API key (for CPI)
    EIA_API_KEY    — EIA API key (for state-level retail sales)
"""

from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Sectors included in the analysis
_SECTORS = ("Residential", "Commercial", "Industrial")


# ---------------------------------------------------------------------------
# URL construction and file download
# ---------------------------------------------------------------------------

def _build_eia861_url(year: int) -> str:
    """
    Construct the download URL for EIA Form 861 annual ZIP archives.

    Parameters
    ----------
    year : int
        Data year.

    Returns
    -------
    str
        URL to the corresponding f861YYYY.zip file.
    """
    if year >= 2024:
        base = "https://www.eia.gov/electricity/data/eia861/zip"
    else:
        base = "https://www.eia.gov/electricity/data/eia861/archive/zip"

    return f"{base}/f861{year}.zip"


def _download_zip_to_bytes(url: str) -> bytes:
    """
    Download a ZIP file from EIA and return raw bytes.

    Parameters
    ----------
    url : str
        URL of the f861YYYY.zip file.

    Returns
    -------
    bytes
        The ZIP archive content.
    """
    logger.info("Downloading EIA-861 archive: %s", url)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# FRED inflation adjustment (annual CPI)
# ---------------------------------------------------------------------------

def _load_cpi_annual_from_fred() -> pd.DataFrame:
    """
    Load annual-average CPI (CPIAUCSL) from the FRED API.

    Users must insert their FRED API key in the placeholder below.

    Returns
    -------
    pandas.DataFrame
        Columns:
            year : int
            cpi  : float
    """
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        raise RuntimeError("FRED_API_KEY environment variable is not set.")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "CPIAUCSL",
        "api_key": api_key,
        "file_type": "json",
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    observations = r.json()["observations"]

    df = pd.DataFrame(
        [
            {"year": int(obs["date"][0:4]), "cpi": float(obs["value"])}
            for obs in observations
            if obs["value"] not in ("", ".", None)
        ]
    )

    df = df.groupby("year", as_index=False)["cpi"].mean()
    return df


def _build_inflation_multipliers(
    reference_year: int = 2026,
) -> pd.DataFrame:
    """
    Construct inflation multipliers that convert nominal revenues into
    real dollars using:
        multiplier(year) = CPI(reference_year) / CPI(year).

    For years where only partial CPI data exists (e.g. current year),
    the available monthly observations are averaged.

    Parameters
    ----------
    reference_year : int
        The dollar-year to adjust to. Default: 2026.

    Returns
    -------
    pandas.DataFrame
        Columns:
            year                : int
            inflation_multiplier : float
    """
    cpi = _load_cpi_annual_from_fred()

    ref_cpi_rows = cpi.loc[cpi["year"] == reference_year, "cpi"]
    if ref_cpi_rows.empty:
        raise ValueError(
            f"No CPI data available for reference year {reference_year}. "
            "Check that FRED has data for this year."
        )
    ref_cpi = ref_cpi_rows.mean()

    cpi["inflation_multiplier"] = ref_cpi / cpi["cpi"]

    return cpi[["year", "inflation_multiplier"]]


# ---------------------------------------------------------------------------
# Read States sheet from Excel
# ---------------------------------------------------------------------------

def _extract_states_sheet_from_zip(zip_bytes: bytes, year: int) -> pd.DataFrame:
    """
    Extract and load the Sales_Ult_Cust_YYYY.xlsx 'States' sheet.

    A temporary directory is used for extraction and removed afterwards.

    Parameters
    ----------
    zip_bytes : bytes
        Downloaded ZIP archive.
    year : int
        Year of data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the multi-row header preserved.
    """
    sales_name = f"Sales_Ult_Cust_{year}.xlsx"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        zip_path = tmp_path / f"f861{year}.zip"
        zip_path.write_bytes(zip_bytes)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)

        matches = list(tmp_path.rglob(sales_name))
        if not matches:
            raise FileNotFoundError(
                f"Could not find {sales_name} within the EIA-861 archive for {year}."
            )

        file_path = matches[0]

        logger.info("Reading States sheet: %s", file_path)
        df = pd.read_excel(
            file_path,
            sheet_name="States",
            header=[0, 1, 2],
            engine="openpyxl",
        )

    return df


# ---------------------------------------------------------------------------
# Header parsing helpers
# ---------------------------------------------------------------------------

def _normalize_token(val) -> str:
    """Normalize a header field for case-insensitive matching."""
    return str(val).strip().lower().replace("\n", " ")


def _column_matches(col, required_tokens) -> bool:
    """
    Determine if a multi-level column contains all required tokens.

    Parameters
    ----------
    col : tuple or str
        Column name (possibly multi-level).
    required_tokens : Sequence[str]
        Tokens that must appear somewhere across column levels.

    Returns
    -------
    bool
        True if all tokens are matched.
    """
    levels = col if isinstance(col, tuple) else (col,)
    levels = [_normalize_token(lv) for lv in levels]

    for token in required_tokens:
        target = _normalize_token(token)
        if not any(target in lv for lv in levels):
            return False
    return True


def _find_column(df: pd.DataFrame, tokens) -> str:
    """
    Locate the single column in `df` whose header matches the required tokens.

    Raises an error if none or multiple matches exist.
    """
    matches = [c for c in df.columns if _column_matches(c, tokens)]
    if not matches:
        raise KeyError(f"No column found matching tokens {tokens}")
    if len(matches) > 1:
        raise KeyError(f"Multiple columns found matching tokens {tokens}: {matches}")
    return matches[0]


# ---------------------------------------------------------------------------
# Raw → long-format utility records
# ---------------------------------------------------------------------------

def _tidy_states_sheet_to_long(df_raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Convert the raw 'States' sheet for a single year into long-format
    utility-level records by sector.

    Parameters
    ----------
    df_raw : pandas.DataFrame
        Raw States sheet.
    year : int
        Data year.

    Returns
    -------
    pandas.DataFrame
        Columns:
            year
            utility_number
            utility_name
            state
            ownership
            sector
            revenue_thousand_dollars
            sales_mwh
            customers
    """
    utility_number_col = _find_column(df_raw, ["utility number"])
    utility_name_col = _find_column(df_raw, ["utility name"])
    state_col = _find_column(df_raw, ["state"])
    ownership_col = _find_column(df_raw, ["ownership"])

    meta = pd.DataFrame({
        "year": year,
        "utility_number": df_raw[utility_number_col],
        "utility_name": df_raw[utility_name_col],
        "state": df_raw[state_col],
        "ownership": df_raw[ownership_col],
    })

    # Keep only valid utility/state rows
    meta = meta[meta["state"].notna()].reset_index(drop=True)
    df_raw = df_raw.loc[meta.index].reset_index(drop=True)

    frames = []

    for sector in _SECTORS:
        revenue_col = _find_column(df_raw, [sector, "revenues", "thousand dollars"])
        sales_col = _find_column(df_raw, [sector, "sales", "megawatthours"])
        customers_col = _find_column(df_raw, [sector, "customers", "count"])

        tmp = meta.copy()
        tmp["sector"] = sector
        tmp["revenue_thousand_dollars"] = pd.to_numeric(df_raw[revenue_col], errors="coerce")
        tmp["sales_mwh"] = pd.to_numeric(df_raw[sales_col], errors="coerce")
        tmp["customers"] = pd.to_numeric(df_raw[customers_col], errors="coerce")

        frames.append(tmp)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Price computation and aggregation
# ---------------------------------------------------------------------------

def _compute_price_per_kwh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price_per_kwh = revenue_thousand_dollars / sales_mwh.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'revenue_thousand_dollars' and 'sales_mwh'.

    Returns
    -------
    pandas.DataFrame
        Same DataFrame with 'price_per_kwh'.
    """
    df = df.copy()
    sales = df["sales_mwh"].astype(float)
    revenue = df["revenue_thousand_dollars"].astype(float)

    df["price_per_kwh"] = pd.NA
    valid = sales > 0
    df.loc[valid, "price_per_kwh"] = revenue[valid] / sales[valid]
    return df


def _aggregate_utility_level(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate long-format utility-sector-year data into
    one record per (year, utility_number, utility_name, state, ownership, sector).

    Computes:
        • Real revenue (already inflation-adjusted upstream)
        • Price per kWh (real)
        • Average load per customer
        • Average annual bill
        • Year-over-year price change
        • Excludes adjustment utilities (utility_number = 99999)

    Parameters
    ----------
    df_all : pandas.DataFrame
        Combined long-format data across all years.

    Returns
    -------
    pandas.DataFrame
        Utility-level records sorted by year, customer count (desc).
    """
    group_cols = [
        "year",
        "utility_number",
        "utility_name",
        "state",
        "ownership",
        "sector",
    ]

    agg = (
        df_all.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            revenue_thousand_dollars=("revenue_thousand_dollars", "sum"),
            sales_mwh=("sales_mwh", "sum"),
            customers=("customers", "sum"),
        )
    )

    agg = _compute_price_per_kwh(agg)
    agg = agg.sort_values(["year", "customers"], ascending=[True, False])

    # Remove EIA adjustment rows
    agg = agg[agg["utility_number"] != 99999]

    # Replace invalid/nonpositive prices with NA
    agg.loc[agg["price_per_kwh"] <= 0, "price_per_kwh"] = pd.NA

    # Average consumption
    agg["avg_mwh_per_customer"] = agg["sales_mwh"] / agg["customers"]
    agg["avg_kwh_per_customer"] = agg["avg_mwh_per_customer"] * 1000

    # Annual bill in real dollars
    agg["avg_annual_bill"] = agg["avg_kwh_per_customer"] * agg["price_per_kwh"]

    # Year-over-year % change in real price
    agg["pct_change_yoy"] = (
        agg.groupby(["utility_number", "sector"])["price_per_kwh"].pct_change(fill_method=None)
    )

    return agg.drop(columns=['avg_mwh_per_customer', 'avg_mwh_per_customer', 'avg_kwh_per_customer'])



# ---------------------------------------------------------------------------
# EIA API (state-level prices, all years)
# ---------------------------------------------------------------------------

EIA_API_URL = "https://api.eia.gov/v2/electricity/retail-sales/data/"

_API_SECTOR_MAP = {
    "residential": "Residential",
    "commercial": "Commercial",
    "industrial": "Industrial",
}


def _fetch_state_prices_from_api(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch state-level electricity retail sales from EIA API v2 for all sectors
    over a range of years.

    Returns raw API response as a DataFrame with columns:
        period, stateid, sectorid, sectorName, customers, price, revenue, sales
    """
    api_key = os.environ.get("EIA_API_KEY", "")
    if not api_key:
        raise RuntimeError("EIA_API_KEY environment variable is not set.")

    params = {
        "api_key": api_key,
        "frequency": "annual",
        "data[0]": "customers",
        "data[1]": "price",
        "data[2]": "revenue",
        "data[3]": "sales",
        "start": str(start_year),
        "end": str(end_year),
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }

    logger.info("Fetching EIA API state prices %s–%s", start_year, end_year)
    resp = requests.get(EIA_API_URL, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json().get("response", {}).get("data", [])
    if not data:
        raise RuntimeError(f"No EIA API data returned for {start_year}–{end_year}.")

    df = pd.DataFrame(data)
    for col in ["customers", "price", "revenue", "sales"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _harmonize_api_to_state_prices(
    df_api: pd.DataFrame,
    infl: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert EIA API output to state_prices schema with inflation adjustment.

    API units:
        price    : cents/kWh
        revenue  : million dollars
        sales    : MWh
        customers: count

    Output columns: year, state, sector, customers, price_per_kwh, avg_annual_bill, pct_change_yoy
    """
    _VALID_STATES = {
        "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL",
        "IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE",
        "NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD",
        "TN","TX","UT","VT","VA","WA","WV","WI","WY",
    }

    df = df_api.copy()
    df["year"] = pd.to_numeric(df["period"], errors="coerce").astype(int)
    df["state"] = df["stateid"]
    df = df[df["state"].isin(_VALID_STATES)].copy()
    df["sector"] = df["sectorName"].str.lower().map(_API_SECTOR_MAP)
    df = df[df["sector"].notna()].copy()

    df = df.merge(infl[["year", "inflation_multiplier"]], on="year", how="left")

    df["price_per_kwh"] = (df["price"] / 100) * df["inflation_multiplier"]
    df.loc[df["price_per_kwh"] <= 0, "price_per_kwh"] = pd.NA

    df["avg_kwh_per_customer"] = (df["sales"] * 1_000_000) / df["customers"]
    df["avg_annual_bill"] = df["price_per_kwh"] * df["avg_kwh_per_customer"]

    df = df[["year", "state", "sector", "customers", "price_per_kwh", "avg_annual_bill"]].sort_values(
        ["state", "sector", "year"]
    )

    df["pct_change_yoy"] = df.groupby(["state", "sector"])["price_per_kwh"].pct_change(fill_method=None)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_eia861_prices(
    utility_years: Sequence[int],
    state_start_year: int = 2015,
    state_end_year: int = 2026,
    reference_year: int = 2026,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build long-format real-dollar utility and state electricity price tables.

    Data sources:
        - EIA Form 861 ZIP files: utility-level only, for `utility_years`
        - EIA API v2: state-level for all years from `state_start_year` to
          `state_end_year` (consistent methodology across all years)

    All prices are inflation-adjusted to `reference_year` dollars using
    annual-average CPI from FRED (partial-year average used if the reference
    year is not yet complete).

    Parameters
    ----------
    utility_years : Sequence[int]
        Years to pull from EIA 861 files for utility-level data (e.g., range(2015, 2025)).
    state_start_year : int
        First year for EIA API state-level pull. Default: 2015.
    state_end_year : int
        Last year for EIA API state-level pull. Default: 2026.
    reference_year : int
        CPI reference year for inflation adjustment. Default: 2026.

    Returns
    -------
    (util_prices, state_prices) : Tuple[pd.DataFrame, pd.DataFrame]

        util_prices :
            Utility-level real-dollar prices from EIA 861 files.

        state_prices :
            State-level real-dollar prices from EIA API, consistent
            methodology across all years.
    """
    # --- Utility-level from 861 files ---
    frames = []
    for year in utility_years:
        url = _build_eia861_url(year)
        logger.info("Processing EIA-861 year %s", year)
        zip_bytes = _download_zip_to_bytes(url)
        df_raw = _extract_states_sheet_from_zip(zip_bytes, year)
        df_long = _tidy_states_sheet_to_long(df_raw, year)
        frames.append(df_long)

    if not frames:
        raise ValueError("No utility_years supplied.")

    df_all = pd.concat(frames, ignore_index=True)

    infl = _build_inflation_multipliers(reference_year=reference_year)
    df_all = df_all.merge(infl, on="year", how="left")
    df_all["revenue_thousand_dollars"] = (
        df_all["revenue_thousand_dollars"] * df_all["inflation_multiplier"]
    )
    util_prices = _aggregate_utility_level(df_all)

    # --- State-level from EIA API ---
    df_api = _fetch_state_prices_from_api(state_start_year, state_end_year)
    state_prices = _harmonize_api_to_state_prices(df_api, infl)

    return util_prices, state_prices

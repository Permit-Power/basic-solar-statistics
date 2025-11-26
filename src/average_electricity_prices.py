"""
average_electricity_prices.py

Utilities to compute inflation-adjusted average retail electricity prices 
($/kWh) by utility and by state/sector using EIA Form 861 "Sales to 
Ultimate Customers" (Sales_Ult_Cust_YYYY.xlsx, States sheet).

This module:
    • Downloads each year's f861YYYY.zip from EIA.
    • Extracts to a temporary directory (auto-deleted afterwards).
    • Reads the "States" sheet of Sales_Ult_Cust_YYYY.xlsx.
    • Pulls Residential, Commercial, Industrial revenue / sales / customers.
    • Inflation-adjusts all revenues to real dollars using annual-average CPI 
      from the Federal Reserve Economic Data (FRED) API.
    • Computes price_per_kwh = real_revenue_thousand_dollars / sales_mwh.
    • Aggregates:
          - Utility-level: by year, utility_number, utility_name, state, ownership.
          - State-level: by year, state.
    • Computes:
          - Average MWh per customer
          - Average annual bill ($ per customer)
          - Year-over-year % change in real price_per_kwh

The output is two long-format DataFrames:
    (util_prices, state_prices)

Dependencies:
    pandas
    requests
    openpyxl

Users must supply a FRED API key for CPI retrieval.
"""

from __future__ import annotations

import logging
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
    api_key = "2764715428a4687d2a8ce57af948081d"
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


def _build_inflation_multipliers(years: Sequence[int]) -> pd.DataFrame:
    """
    Construct inflation multipliers that convert nominal revenues into
    real dollars using:
        multiplier(year) = CPI(latest_year) / CPI(year).

    Parameters
    ----------
    years : Sequence[int]
        List of years for which multipliers are needed.

    Returns
    -------
    pandas.DataFrame
        Columns:
            year                : int
            inflation_multiplier : float
    """
    cpi = _load_cpi_annual_from_fred()
    max_year = max(years)

    latest_cpi = cpi.loc[cpi["year"] == max_year, "cpi"].iloc[0]
    cpi["inflation_multiplier"] = latest_cpi / cpi["cpi"]

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
        agg.groupby(["utility_number", "sector"])["price_per_kwh"].pct_change()
    )

    return agg.drop(columns=['avg_mwh_per_customer', 'avg_mwh_per_customer', 'avg_kwh_per_customer'])


def _aggregate_state_level(df_utility: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate utility-level values into state-level weighted averages.

    Parameters
    ----------
    df_utility : pandas.DataFrame
        Utility-level aggregated data.

    Returns
    -------
    pandas.DataFrame
        State-level (year, state, sector) records with real prices.
    """
    group_cols = ["year", "state", "sector"]

    agg = (
        df_utility.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            revenue_thousand_dollars=("revenue_thousand_dollars", "sum"),
            sales_mwh=("sales_mwh", "sum"),
            customers=("customers", "sum"),
        )
    )

    agg = _compute_price_per_kwh(agg)
    agg = agg.sort_values(["year", "state", "sector"])

    agg.loc[agg["price_per_kwh"] <= 0, "price_per_kwh"] = pd.NA

    agg["avg_mwh_per_customer"] = agg["sales_mwh"] / agg["customers"]
    agg["avg_kwh_per_customer"] = agg["avg_mwh_per_customer"] * 1000
    agg["avg_annual_bill"] = agg["avg_kwh_per_customer"] * agg["price_per_kwh"]

    agg["pct_change_yoy"] = (
        agg.groupby(["state", "sector"])["price_per_kwh"].pct_change()
    )

    return agg.drop(columns=['avg_mwh_per_customer', 'avg_kwh_per_customer', 'revenue_thousand_dollars', 'sales_mwh'])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_eia861_prices(
    years: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build long-format real-dollar utility and state electricity price tables
    from EIA Form 861 Sales to Ultimate Customers.

    Parameters
    ----------
    years : Sequence[int]
        Years to download and process (e.g., range(2010, 2025)).

    Returns
    -------
    (util_prices, state_prices) : Tuple[pandas.DataFrame, pandas.DataFrame]

        util_prices :
            Long-format utility-level dataset including:
                • Real revenue
                • Sales (MWh)
                • Customers
                • Real price per kWh
                • Avg consumption per customer
                • Avg annual bill
                • YoY price changes

        state_prices :
            State-level equivalent aggregations.
    """
    frames = []

    for year in years:
        url = _build_eia861_url(year)
        logger.info("Processing EIA-861 year %s", year)

        zip_bytes = _download_zip_to_bytes(url)
        df_raw = _extract_states_sheet_from_zip(zip_bytes, year)
        df_long = _tidy_states_sheet_to_long(df_raw, year)

        frames.append(df_long)

    if not frames:
        raise ValueError("No years supplied.")

    df_all = pd.concat(frames, ignore_index=True)

    # Apply real-dollar inflation adjustment
    infl = _build_inflation_multipliers(years)
    df_all = df_all.merge(infl, on="year", how="left")
    df_all["revenue_thousand_dollars"] = (
        df_all["revenue_thousand_dollars"] * df_all["inflation_multiplier"]
    )

    # Aggregate
    util_prices = _aggregate_utility_level(df_all)
    state_prices = _aggregate_state_level(util_prices)

    return util_prices, state_prices

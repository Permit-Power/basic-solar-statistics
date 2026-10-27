"""
average_electricity_prices.py

Computes average retail electricity prices ($/kWh) by utility and by
state/sector, inflation-adjusted to dollars of the latest year with EIA state
data (so that year's prices are what customers paid, as in EIA's tables, and
earlier years are comparable to it), using two data sources:

Utility-level (EIA Form 861 annual ZIP files):
    • Downloads f861YYYY.zip from EIA for each requested year.
    • Reads the "States" sheet of Sales_Ult_Cust_YYYY.xlsx.
    • Pulls Residential, Commercial, Industrial revenue / sales / customers.
    • Inflation-adjusts revenues to reference-year dollars via FRED CPI.
    • Aggregates to utility × year × sector.

State-level (EIA API v2 — retail-sales endpoint):
    • Fetches annual customers, price, revenue, and sales for all sectors
      and all states plus a US total over a configurable year range (default 2018–2026).
    • Converts cents/kWh to reference-year $/kWh using FRED CPI.
    • Consistent methodology across all years (no file-format changes).

Inflation adjustment:
    • Annual-average CPI (CPIAUCSL) from FRED API.
    • Reference year = the latest year in the EIA API state data (2025 as of
      October 2026); it moves forward when EIA publishes a new year.

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
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence, Tuple

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


def _build_inflation_multipliers(reference_year: int) -> pd.DataFrame:
    """
    Construct inflation multipliers that convert nominal revenues into
    real dollars using:
        multiplier(year) = CPI(reference_year) / CPI(year).

    For years where only partial CPI data exists (e.g. current year),
    the available monthly observations are averaged.

    Parameters
    ----------
    reference_year : int
        The dollar-year to adjust to (the latest year with EIA state data).

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

def _extract_states_sheet_from_zip(
    zip_bytes: bytes,
    year: int,
    file_stem: str = "Sales_Ult_Cust",
    sheet_name: str = "States",
) -> pd.DataFrame:
    """
    Extract and load the Sales_Ult_Cust_YYYY.xlsx 'States' sheet (or another
    sheet with the same three-row header, e.g. Delivery_Companies_YYYY.xlsx).

    A temporary directory is used for extraction and removed afterwards.

    Parameters
    ----------
    zip_bytes : bytes
        Downloaded ZIP archive.
    year : int
        Year of data.
    file_stem : str
        File name before "_YYYY.xlsx". Default: "Sales_Ult_Cust".
    sheet_name : str
        Sheet to read. Default: "States".

    Returns
    -------
    pandas.DataFrame
        DataFrame with the multi-row header preserved.
    """
    sales_name = f"{file_stem}_{year}.xlsx"

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

        logger.info("Reading %s sheet: %s", sheet_name, file_path)
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
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
    # The API returns at most `length` rows per request, so page through.
    data = []
    while True:
        resp = requests.get(EIA_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        response = resp.json().get("response", {})
        page = response.get("data", [])
        data.extend(page)
        params["offset"] += len(page)
        if not page or params["offset"] >= int(response.get("total", 0)):
            break

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
    # The 50 states, DC, and "US" (EIA's national total: national revenue /
    # national sales). Census-region rows (ENC, PACC, ...) are dropped.
    _VALID_STATES = {"US",
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
    state_start_year: int = 2018,
    state_end_year: int = 2026,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build long-format real-dollar utility and state electricity price tables.

    Data sources:
        - EIA Form 861 ZIP files: utility-level only, for `utility_years`
        - EIA API v2: state-level for all years from `state_start_year` to
          `state_end_year` (consistent methodology across all years)

    All prices are inflation-adjusted to dollars of the latest year in the EIA
    API state data (annual-average CPI from FRED), so that year's prices are
    nominal.

    Parameters
    ----------
    utility_years : Sequence[int]
        Years to pull from EIA 861 files for utility-level data (e.g., range(2018, 2025)).
    state_start_year : int
        First year for EIA API state-level pull. Default: 2018.
    state_end_year : int
        Last year for EIA API state-level pull. Default: 2026.

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

    # --- State-level from EIA API; its latest year sets the dollar-year ---
    df_api = _fetch_state_prices_from_api(state_start_year, state_end_year)
    reference_year = int(pd.to_numeric(df_api["period"]).max())
    infl = _build_inflation_multipliers(reference_year=reference_year)
    state_prices = _harmonize_api_to_state_prices(df_api, infl)

    df_all = df_all.merge(infl, on="year", how="left")
    df_all["revenue_thousand_dollars"] = (
        df_all["revenue_thousand_dollars"] * df_all["inflation_multiplier"]
    )
    util_prices = _aggregate_utility_level(df_all)

    return util_prices, state_prices


# ---------------------------------------------------------------------------
# Residential fixed charges and the energy price without them
# ---------------------------------------------------------------------------
#
# EIA revenue includes fixed monthly charges (customer charge, basic service
# charge), so revenue / sales is more than what one more kWh costs. EIA does not
# report fixed charges, so we assume one per state and year (2020 onward):
#
#   1. Main utilities. From that year's EIA-861, the utilities that together serve
#      95% of the state's residential customers, largest first. Only IOUs, co-ops,
#      munis, public power districts, and state and federal utilities count; retail
#      power marketers and behind-the-meter sellers (e.g. Sunrun) are left out.
#      Customers come from Sales_Ult_Cust "States" plus Delivery_Companies, which
#      holds the Texas TDUs (Oncor, CenterPoint, ...). EIA counts competitive Texas
#      customers under their retail providers, so without it those households would
#      have no utility in the list.
#   2. Each main utility's monthly charge that year, in this order:
#      a. data/residential_fixed_charges_manual.csv: values checked by hand for large
#         utilities where URDB was wrong or missing (October 2026: CA, ME, MS, NE and
#         the Texas TDUs). Each row has the source URL, effective date, a quote or
#         page, and `evidence`: "tariff" (read in the tariff PDF), "tariff (inferred)"
#         (a neighbouring year's tariff shows the same structure), "secondary" (URDB
#         record, PUCT summary or a rate-history site; tariff not read) or "not found"
#         (blank charge: URDB's value was wrong and no replacement was found, so the
#         utility counts as uncovered). Minimum bills, including Maine's minimums that
#         cover the first 50-100 kWh, are not fixed charges. Tiered charges (LADWP's
#         Power Access Charge, LES's facilities charge) use one tier, named in `note`.
#      b. data/urdb_residential_schedules.csv: a snapshot of OpenEI's U.S. Utility
#         Rate Database (see refresh_urdb_snapshot). Residential schedules in effect
#         on July 1 that started at most URDB_MAX_AGE_YEARS earlier. URDB's default
#         schedule if one is flagged, else the median over standard schedules (names
#         matching _NONSTANDARD_SCHEDULE, such as three-phase or master-metered, are
#         skipped). $/day is converted at 365/12 days per month. Charges above
#         MAX_FIXED_CHARGE are dropped; they are unit errors or not household rates.
#   3. State charge: the customer-weighted mean over main utilities that have one.
#      Coverage is their share of main-utility customers. Below MIN_COVERAGE, the
#      state's nearest year at or above it is used instead (the earlier one on ties),
#      and fixed_charge_year says which.
#   4. energy_price_per_kwh = price_per_kwh - 12 x charge / kWh per customer, with
#      the charge converted from fixed_charge_year dollars to the prices' dollar-year
#      (the latest year in state_prices), so the latest year's charge is as tariffed.
#
# Not removed: retail suppliers' own fixed charges (e.g. Texas REP base charges,
# not in URDB) and minimum bills (paid only by low users).

URDB_DOWNLOAD_URL = "https://openei.org/apps/USURDB/download/usurdb.csv.gz"
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
URDB_SNAPSHOT = _DATA_DIR / "urdb_residential_schedules.csv"
MANUAL_FIXED_CHARGES = _DATA_DIR / "residential_fixed_charges_manual.csv"

MAIN_UTILITY_SHARE = 0.95
MAX_FIXED_CHARGE = 50.0  # $/month
URDB_MAX_AGE_YEARS = 6
MIN_COVERAGE = 0.5

_UTILITY_OWNERSHIP = {
    "Investor Owned", "Cooperative", "Municipal", "Political Subdivision", "State", "Federal",
}
_NONSTANDARD_SCHEDULE = re.compile(
    r"three.?phase|3.?phase|master.?meter|primary|general service|farm|demand|\bev\b"
    r"|electric vehicle|optional|seasonal|lighting|net meter|multi",
    re.IGNORECASE,
)
_URDB_COLUMNS = [
    "label", "eiaid", "utility", "name", "is_default", "startdate", "enddate",
    "fixedchargefirstmeter", "fixedchargeunits", "source",
]


def refresh_urdb_snapshot(path: Path = URDB_SNAPSHOT) -> pd.DataFrame:
    """
    Download URDB's full export and save its residential schedules to `path`.

    Run by hand when updating (not by the pipeline), then commit the file and
    note the download date in the README. Keeps schedules with a $/month or $/day
    fixed charge and the columns in _URDB_COLUMNS. A schedule is looked up at
    https://apps.openei.org/USURDB/rate/view/<label>.
    """
    logger.info("Downloading URDB export: %s", URDB_DOWNLOAD_URL)
    urdb = pd.read_csv(URDB_DOWNLOAD_URL, usecols=_URDB_COLUMNS + ["sector"], low_memory=False)
    urdb = urdb[
        (urdb["sector"] == "Residential")
        & urdb["fixedchargefirstmeter"].notna()
        & urdb["fixedchargeunits"].isin(["$/month", "$/day"])
        & urdb["eiaid"].notna()
    ]
    urdb = urdb[_URDB_COLUMNS].astype({"eiaid": int})
    urdb.to_csv(path, index=False)
    return urdb


def _main_utilities(zip_bytes: bytes, year: int) -> pd.DataFrame:
    """
    Main utilities per state for one EIA-861 year (step 1 above).

    Returns
    -------
    pandas.DataFrame
        Columns: year, state, utility_number, utility_name, ownership, customers
        (residential), largest first within each state.
    """
    frames = []
    for file_stem, sheet_name in [("Sales_Ult_Cust", "States"), ("Delivery_Companies", "Delivery Companies")]:
        raw = _extract_states_sheet_from_zip(zip_bytes, year, file_stem, sheet_name)
        frames.append(_tidy_states_sheet_to_long(raw, year))
    df = pd.concat(frames, ignore_index=True)

    df = df[(df["sector"] == "Residential") & df["ownership"].isin(_UTILITY_OWNERSHIP)]
    df = df[df["customers"] > 0]
    df = (
        df.groupby(["year", "state", "utility_number", "utility_name", "ownership"], as_index=False)
        ["customers"].sum()
        .sort_values(["state", "customers"], ascending=[True, False])
    )

    # Keep utilities until 95% of the state's customers are covered (the one
    # crossing 95% is kept).
    share = df["customers"] / df.groupby("state")["customers"].transform("sum")
    customers_before = share.groupby(df["state"]).cumsum() - share
    df = df[customers_before < MAIN_UTILITY_SHARE].copy()
    df["utility_number"] = df["utility_number"].astype(int)
    return df.reset_index(drop=True)


def _urdb_fixed_charges(urdb: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    One monthly fixed charge per EIA utility ID from URDB for `year` (step 2b).

    Returns
    -------
    pandas.DataFrame
        Indexed by utility_number. Columns: fixed_charge_monthly, fixed_charge_source,
        schedule, reference (URDB label), effective_date.
    """
    on = pd.Timestamp(f"{year}-07-01")
    start = pd.to_datetime(urdb["startdate"], errors="coerce")
    end = pd.to_datetime(urdb["enddate"], errors="coerce")
    in_effect = (start <= on) & (end.isna() | (end > on)) & (start >= on - pd.DateOffset(years=URDB_MAX_AGE_YEARS))

    df = urdb[in_effect].assign(effective_date=start[in_effect].dt.date)
    df["fixed_charge_monthly"] = df["fixedchargefirstmeter"] * df["fixedchargeunits"].map(
        {"$/month": 1.0, "$/day": 365 / 12}
    )
    df = df[df["fixed_charge_monthly"].between(0, MAX_FIXED_CHARGE)]
    df["standard"] = ~df["name"].str.contains(_NONSTANDARD_SCHEDULE, na=False)

    rows = {}
    for uid, g in df.groupby("eiaid"):
        default, standard = g[g["is_default"] == True], g[g["standard"]]  # noqa: E712 (NaN = not flagged)
        if len(default):
            chosen, source = default, "URDB default schedule"
        elif len(standard):
            chosen, source = standard, f"URDB median of {len(standard)} standard schedules"
        else:
            continue
        # The schedule reported is the one at the median (lower middle for even counts).
        chosen = chosen.sort_values("fixed_charge_monthly")
        mid = chosen.iloc[(len(chosen) - 1) // 2]
        rows[uid] = {
            "fixed_charge_monthly": round(chosen["fixed_charge_monthly"].median(), 2),
            "fixed_charge_source": source,
            "schedule": mid["name"],
            "reference": mid["label"],
            "effective_date": mid["effective_date"],
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "utility_number"
    return out


def build_residential_fixed_charges(
    years: Sequence[int] = range(2020, 2026),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assumed residential fixed charge per main utility and per state (nominal $/month).

    See the comment at the top of this section for the method.

    Returns
    -------
    (utility_fixed, state_fixed) : Tuple[pd.DataFrame, pd.DataFrame]

        utility_fixed :
            One row per year, state and main utility: customers,
            fixed_charge_monthly (blank if none found), fixed_charge_source,
            schedule, reference (URDB label or tariff URL), effective_date.

        state_fixed :
            One row per year and state: main_utilities, main_utility_customers,
            fixed_charge_coverage, fixed_charge_monthly, fixed_charge_year (the year
            whose value is used; differs from year when coverage is below MIN_COVERAGE).
    """
    urdb = pd.read_csv(URDB_SNAPSHOT)
    manual = pd.read_csv(MANUAL_FIXED_CHARGES)
    assert not manual.duplicated(["year", "state", "utility_number"]).any(), "duplicate manual rows"

    frames = []
    for year in years:
        main = _main_utilities(_download_zip_to_bytes(_build_eia861_url(year)), year)
        urdb_year = _urdb_fixed_charges(urdb, year)
        df = main.merge(urdb_year, left_on="utility_number", right_index=True, how="left")

        # A hand-checked row replaces URDB for that utility and year, including a
        # blank charge ("not found"), which means URDB's value was checked and is wrong.
        hand = manual[manual["year"] == year].set_index(["state", "utility_number"])
        key = pd.MultiIndex.from_frame(df[["state", "utility_number"]])
        has_hand = key.isin(hand.index)
        for col, manual_col in [
            ("fixed_charge_monthly", "fixed_charge_monthly"),
            ("schedule", "schedule"),
            ("reference", "source_url"),
            ("effective_date", "effective_date"),
        ]:
            df.loc[has_hand, col] = hand.loc[key[has_hand], manual_col].values
        df.loc[has_hand, "fixed_charge_source"] = "Hand check: " + hand.loc[key[has_hand], "evidence"].values
        assert len(df) == len(main)
        frames.append(df)

    utility_fixed = pd.concat(frames, ignore_index=True)
    utility_fixed["fixed_charge_source"] = utility_fixed["fixed_charge_source"].fillna("None found")
    assert utility_fixed["fixed_charge_monthly"].dropna().between(0, MAX_FIXED_CHARGE).all()

    has = utility_fixed["fixed_charge_monthly"].notna()
    covered = utility_fixed["customers"].where(has, 0)
    state_fixed = (
        utility_fixed.assign(
            covered=covered,
            weighted=(utility_fixed["fixed_charge_monthly"] * utility_fixed["customers"]).where(has, 0),
        )
        .groupby(["year", "state"], as_index=False)
        .agg(
            main_utilities=("utility_number", "size"),
            main_utility_customers=("customers", "sum"),
            covered=("covered", "sum"),
            weighted=("weighted", "sum"),
        )
    )
    state_fixed["fixed_charge_coverage"] = state_fixed["covered"] / state_fixed["main_utility_customers"]
    state_fixed["own_charge"] = state_fixed["weighted"] / state_fixed["covered"].where(state_fixed["covered"] > 0)

    # Below MIN_COVERAGE: borrow the nearest year that passes (earlier on ties).
    passing = state_fixed[state_fixed["fixed_charge_coverage"] >= MIN_COVERAGE]
    use_year = {}
    for _, r in state_fixed.iterrows():
        if r["fixed_charge_coverage"] >= MIN_COVERAGE:
            use_year[(r["year"], r["state"])] = r["year"]
            continue
        candidates = passing.loc[passing["state"] == r["state"], "year"]
        if candidates.empty:
            raise ValueError(f"{r['state']}: no year with fixed-charge coverage >= {MIN_COVERAGE}")
        use_year[(r["year"], r["state"])] = min(candidates, key=lambda y: (abs(y - r["year"]), y))
    state_fixed["fixed_charge_year"] = [use_year[(y, s)] for y, s in zip(state_fixed["year"], state_fixed["state"])]
    own = state_fixed.set_index(["year", "state"])["own_charge"]
    state_fixed["fixed_charge_monthly"] = [
        own[(y, s)] for y, s in zip(state_fixed["fixed_charge_year"], state_fixed["state"])
    ]
    state_fixed = state_fixed.drop(columns=["covered", "weighted", "own_charge"])
    assert state_fixed["fixed_charge_monthly"].notna().all()

    return utility_fixed, state_fixed


def add_energy_price(state_prices: pd.DataFrame, state_fixed: pd.DataFrame) -> pd.DataFrame:
    """
    Add the assumed fixed charge and the energy price without it to residential rows.

    Fixed charges are converted to the prices' dollar-year (the latest year in
    state_prices, as in build_eia861_prices) with the CPI of the year they come
    from (fixed_charge_year). The US row uses the customer-weighted mean of the
    state charges. Rows outside state_fixed's years, and non-residential rows,
    get blanks.

    Adds columns: fixed_charge_monthly (dollars of the latest year), fixed_charge_year,
    fixed_charge_coverage, energy_price_per_kwh.
    """
    infl = _build_inflation_multipliers(reference_year=int(state_prices["year"].max()))
    fixed = state_fixed.merge(
        infl.rename(columns={"year": "fixed_charge_year"}), on="fixed_charge_year", how="left"
    )
    fixed["fixed_charge_monthly"] = fixed["fixed_charge_monthly"] * fixed["inflation_multiplier"]
    fixed = fixed[["year", "state", "fixed_charge_monthly", "fixed_charge_year", "fixed_charge_coverage"]]
    fixed["sector"] = "Residential"

    res = state_prices[(state_prices["sector"] == "Residential") & (state_prices["state"] != "US")]
    us = (
        res.merge(fixed, on=["year", "state", "sector"])
        .assign(w=lambda d: d["fixed_charge_monthly"] * d["customers"])
        .groupby("year", as_index=False)
        .agg(w=("w", "sum"), customers=("customers", "sum"))
    )
    us = us.assign(
        state="US", sector="Residential", fixed_charge_monthly=us["w"] / us["customers"],
    ).drop(columns=["w", "customers"])
    fixed = pd.concat([fixed, us], ignore_index=True)

    out = state_prices.merge(fixed, on=["year", "state", "sector"], how="left")
    assert len(out) == len(state_prices)
    kwh_per_customer = out["avg_annual_bill"] / out["price_per_kwh"]
    out["energy_price_per_kwh"] = out["price_per_kwh"] - 12 * out["fixed_charge_monthly"] / kwh_per_customer
    has = out["fixed_charge_monthly"].notna()
    assert (out.loc[has, "energy_price_per_kwh"] > 0).all()
    assert (out.loc[has, "energy_price_per_kwh"] < out.loc[has, "price_per_kwh"]).all()
    return out

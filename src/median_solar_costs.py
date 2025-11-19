"""
median_solar_costs.py

Comprehensive module for processing LBNL Tracking the Sun datasets and generating
inflation-adjusted median residential solar costs by state and year.

This module performs the following end-to-end pipeline:

1. Google Drive Download
   - Downloads an LBNL ZIP archive from a public Google Drive link.
   - Handles Drive’s non-direct download URLs via `gdown`.

2. ZIP Extraction
   - Extracts the downloaded ZIP.
   - Recursively searches the extracted directory for the LBNL CSV file.

3. CPI Inflation Data (FRED)
   - Fetches monthly CPI (CPIAUCSL) observations from the FRED API
     back to a configurable start year (default: 2000).
   - Builds a consistent inflation multiplier table:
         inflation_multiplier = CPI_latest / CPI_month
     so all historical prices are scaled to the latest CPI month.

4. Solar Cost Processing
   - Filters the LBNL dataset to residential PV-only and PV+storage systems.
   - Joins inflation multipliers by (year, month).
   - Inflation-adjusts system prices and computes $/kW.
   - Aggregates to median price_per_kw by state × year.
   - Computes year-over-year percent changes within each state.

5. Full Pipeline Wrapper
   - A single function call downloads, extracts, fetches CPI, builds inflation
     multipliers, loads LBNL, and returns the final median solar cost table.

Dependencies:
    pip install gdown pandas requests
"""

import os
import zipfile
from typing import Optional

import gdown
import pandas as pd
import requests
import shutil
import tempfile

# ---------------------------------------------------------------------------
# CPI FETCH + INFLATION MULTIPLIERS
# ---------------------------------------------------------------------------


def fetch_cpi_from_fred(api_key: str, start_year: int = 2000) -> pd.DataFrame:
    """
    Fetch monthly CPI (CPIAUCSL) from the St. Louis Fed FRED API.

    Parameters
    ----------
    api_key : str
        Your FRED API key (free to register).

    start_year : int, optional
        Earliest year of CPI data to load. Default is 2000.

    Returns
    -------
    pd.DataFrame
        DataFrame containing columns:
            - date  (Timestamp)
            - year  (int)
            - month (int)
            - cpi   (float)

    Notes
    -----
    - CPIAUCSL is the Consumer Price Index for All Urban Consumers:
      All Items (seasonally adjusted), monthly.
    - This function does not perform any inflation adjustment yet; it only
      retrieves raw CPI values.
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "CPIAUCSL",
        "api_key": api_key,
        "file_type": "json",
        "observation_start": f"{start_year}-01-01",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json().get("observations", [])
    if not data:
        raise RuntimeError("No CPI observations returned from FRED.")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["cpi"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["date", "cpi"])[["date", "cpi"]]
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    return df


def build_inflation_multipliers(api_key: str, start_year: int = 2000) -> pd.DataFrame:
    """
    Create an inflation multiplier table using FRED CPI monthly data.

    The multiplier is computed as:

        inflation_multiplier = CPI_latest / CPI_month

    This scales historical prices up to the latest CPI month in the series.

    Parameters
    ----------
    api_key : str
        FRED API key used to retrieve CPI data.

    start_year : int, optional
        Year from which to begin CPI retrieval. Default is 2000.

    Returns
    -------
    pd.DataFrame
        Columns:
            - year
            - month
            - cpi
            - inflation_multiplier

    Raises
    ------
    RuntimeError
        If CPI data cannot be retrieved or is empty.
    """
    cpi = fetch_cpi_from_fred(api_key, start_year=start_year)
    if cpi.empty:
        raise RuntimeError("CPI DataFrame is empty; cannot build multipliers.")

    latest_cpi = cpi["cpi"].iloc[-1]

    cpi["inflation_multiplier"] = latest_cpi / cpi["cpi"]

    return cpi[["year", "month", "cpi", "inflation_multiplier"]]


# ---------------------------------------------------------------------------
# DOWNLOAD + EXTRACT LBNL ZIP
# ---------------------------------------------------------------------------


def download_lbnl_zip(drive_url: str, output_dir: str = "lbnl_data") -> str:
    """
    Download and extract the LBNL dataset ZIP from a Google Drive sharing link.

    Parameters
    ----------
    drive_url : str
        Google Drive sharing link of the form:
        https://drive.google.com/file/d/<ID>/view

    output_dir : str, optional
        Directory in which to save the ZIP and extracted contents.
        Default is "lbnl_data".

    Returns
    -------
    str
        Path to the directory containing the extracted ZIP contents.

    Notes
    -----
    Google Drive does not allow simple direct-download URLs;
    `gdown` handles the confirmation token logic and large file downloads.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract file ID from shared URL
    try:
        file_id = drive_url.split("/d/")[1].split("/")[0]
    except IndexError as exc:
        raise ValueError(f"Could not parse file ID from Google Drive URL: {drive_url}") from exc

    direct_url = f"https://drive.google.com/uc?id={file_id}"
    zip_path = os.path.join(output_dir, "lbnl_latest.zip")

    print("Downloading LBNL ZIP from Google Drive …")
    gdown.download(direct_url, zip_path, quiet=False)

    print("Extracting LBNL ZIP …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    return output_dir


# ---------------------------------------------------------------------------
# LOAD LBNL CSV
# ---------------------------------------------------------------------------


def load_lbnl_csv(folder: str, preferred_substring: Optional[str] = None) -> pd.DataFrame:
    """
    Search recursively for a CSV file inside the extracted LBNL dataset folder
    and load it as a pandas DataFrame.

    Parameters
    ----------
    folder : str
        Root extraction directory.

    preferred_substring : str, optional
        If provided, the search will prefer CSVs whose filename contains this
        substring (case-insensitive). If none match, the first CSV found is used.

    Returns
    -------
    pd.DataFrame
        The loaded LBNL dataset.

    Raises
    ------
    FileNotFoundError
        If no CSV file is found anywhere inside the folder.
    """
    preferred_substring = (
        preferred_substring.lower() if preferred_substring is not None else None
    )

    first_csv_path: Optional[str] = None
    preferred_csv_path: Optional[str] = None

    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(".csv"):
                path = os.path.join(root, fname)
                if first_csv_path is None:
                    first_csv_path = path
                if preferred_substring and preferred_substring in fname.lower():
                    preferred_csv_path = path
                    break  # found preferred; stop early
        if preferred_csv_path:
            break

    csv_path = preferred_csv_path or first_csv_path
    print(csv_path)
    if not csv_path:
        raise FileNotFoundError("No CSV file found inside extracted LBNL archive.")

    print(f"Loading LBNL CSV: {csv_path}")
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# MEDIAN SOLAR COSTS
# ---------------------------------------------------------------------------


def compute_median_solar_costs_all_states(
    lbnl: pd.DataFrame,
    inflation: pd.DataFrame,
    min_year: int = 2019,
) -> pd.DataFrame:
    """
    Compute inflation-adjusted median PV system cost ($/kW) by state × year.

    Filtering rules:
        - technology_type in {"pv-only", "pv+storage"}
        - customer_segment in {"RES", "RES_SF"}
        - total_installed_price > 0
        - PV_system_size_DC > 0

    Inflation adjustment:
        adj_price = total_installed_price * inflation_multiplier

    Aggregation:
        - price_per_kw = adj_price / PV_system_size_DC
        - median_price_per_kw = median(price_per_kw) per (state, year)
        - pct_change = year-over-year percent change in median_price_per_kw,
                       grouped by state

    Parameters
    ----------
    lbnl : pd.DataFrame
        Raw LBNL dataset.

    inflation : pd.DataFrame
        Inflation table from `build_inflation_multipliers()`,
        with columns: year, month, inflation_multiplier.

    min_year : int, optional
        Earliest installation year to include (default: 2019).

    Returns
    -------
    pd.DataFrame
        Columns:
            - state
            - year
            - median_price_per_kw
            - pct_change
    """
    required_cols = {
        "technology_type",
        "customer_segment",
        "total_installed_price",
        "PV_system_size_DC",
        "installation_date",
        "state",
    }
    missing = required_cols - set(lbnl.columns)
    if missing:
        raise ValueError(f"LBNL DataFrame missing required columns: {sorted(missing)}")

    inf_required = {"year", "month", "inflation_multiplier"}
    inf_missing = inf_required - set(inflation.columns)
    if inf_missing:
        raise ValueError(
            f"Inflation DataFrame missing required columns: {sorted(inf_missing)}"
        )

    # Filter to relevant systems
    filt = lbnl[
        (lbnl["technology_type"].isin(["pv-only", "pv+storage"]))
        & (lbnl["customer_segment"].isin(["RES", "RES_SF"]))
        & (lbnl["total_installed_price"] > 0)
        & (lbnl["PV_system_size_DC"] > 0)
    ].copy()

    # Parse dates and attach year/month
    filt["parsed_date"] = pd.to_datetime(filt["installation_date"], errors="coerce")
    filt["year"] = filt["parsed_date"].dt.year
    filt["month"] = filt["parsed_date"].dt.month

    # Restrict to recent data
    filt = filt[filt["year"] >= min_year]

    # Merge inflation multipliers on (year, month)
    merged = filt.merge(inflation, on=["year", "month"], how="left")

    # Inflation-adjust prices
    merged["adj_price"] = merged["total_installed_price"] * merged["inflation_multiplier"]
    merged["price_per_kw"] = merged["adj_price"] / merged["PV_system_size_DC"]

    # Groupby median
    med = (
        merged.groupby(["state", "year"], as_index=False)["price_per_kw"]
        .median()
        .rename(columns={"price_per_kw": "median_price_per_kw"})
        .sort_values(["state", "year"])
    )

    # Year-over-year percent change within each state
    med["pct_change"] = (
        med.groupby("state")["median_price_per_kw"].pct_change()
    )

    return med


# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------


def load_and_process_lbnl(
    drive_url: str,
    fred_api_key: str,
    start_year: int = 2000,
    min_install_year: int = 2019,
    output_dir: str = "lbnl_data",
    preferred_csv_substring: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full pipeline wrapper for:
        - Downloading LBNL ZIP from Google Drive
        - Extracting and loading the LBNL CSV
        - Fetching CPI from FRED and building inflation multipliers
        - Computing median solar costs by state × year

    Parameters
    ----------
    drive_url : str
        Google Drive sharing link to the LBNL ZIP file.

    fred_api_key : str
        FRED API key used to retrieve CPI data for inflation adjustments.

    start_year : int, optional
        Earliest year of CPI data to retrieve from FRED. Default: 2000.

    min_install_year : int, optional
        Earliest installation year from LBNL data to include
        in cost calculations. Default: 2019.

    output_dir : str, optional
        Directory where the LBNL ZIP will be downloaded and extracted.
        Default: "lbnl_data".

    preferred_csv_substring : str, optional
        If provided, the CSV loader will prefer files whose name
        contains this substring (case-insensitive).

    Returns
    -------
    pd.DataFrame
        Median solar costs by state × year with inflation adjustment,
        including year-over-year percent changes.
    """
    # 1. Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="lbnl_")

    try:
        # 2. Download + extract ZIP into the temp folder
        extracted_folder = download_lbnl_zip(drive_url, output_dir=temp_dir)

        # 3. Load the *correct* LBNL CSV from the clean folder
        lbnl_df = load_lbnl_csv(extracted_folder, preferred_substring=preferred_csv_substring)

        # 4. Build inflation multipliers
        inflation = build_inflation_multipliers(fred_api_key, start_year=start_year)

        # 5. Compute median solar costs
        results = compute_median_solar_costs_all_states(
            lbnl=lbnl_df,
            inflation=inflation,
            min_year=min_install_year,
        )

        return results

    finally:
        # 6. Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

"""
median_solar_costs.py

Median residential solar installed price and system size by state and year,
from LBNL's Tracking the Sun data tool export (data/tts_state_download_data.xlsx,
downloaded from https://trackingthesun.lbl.gov).

The export already contains LBNL's state-level statistics, computed with
LBNL's own methodology and already adjusted for inflation by LBNL, so this
module only reshapes them. Prices are in 2025 dollars: the 2025 values are
nominal and LBNL adjusts earlier years to 2025 (concluded by comparing with
the earlier 2024-edition export; the workbook itself doesn't state the dollar
year). (It replaces processing the full ~2 GB Tracking the
Sun public file, which needed a Google Drive download and a FRED CPI series.)

Output columns
    state                  two-letter state code ("US" for the national row)
    year                   installation year
    median_price_per_kw    median installed price, $/kW DC (LBNL's $/W x 1000)
    median_system_size_kw  median system size, kW DC
    pct_change_price       year-over-year % change in median_price_per_kw, by state
    price_n                systems behind the price median
    system_size_n          systems behind the size median

LBNL leaves a value blank when a state has too few systems, so many states
have no price or size in a given year; some recent medians it does publish
rest on very few systems (see MIN_SYSTEMS).
"""

from pathlib import Path

import pandas as pd

TTS_XLSX = Path(__file__).resolve().parents[1] / "data" / "tts_state_download_data.xlsx"


def load_tts_state_data(path: Path = TTS_XLSX, min_year: int = 2000) -> pd.DataFrame:
    """
    Residential median price and system size by state and year from the
    Tracking the Sun data tool export.

    Parameters
    ----------
    path : Path
        The data tool's Excel export.
    min_year : int
        Earliest installation year to keep.

    Returns
    -------
    pd.DataFrame
        Columns as described in the module docstring, sorted by state and year.
    """
    # The "Data" sheet has three title rows above the header.
    df = pd.read_excel(path, sheet_name="Data", header=3)
    df = df[(df["type"] == "Residential") & (df["year"] >= min_year)].copy()

    out = pd.DataFrame({
        "state": df["state"].replace({"National": "US"}),
        "year": df["year"].astype(int),
        "median_price_per_kw": df["price_dollar_per_watt_median"] * 1000,
        "median_system_size_kw": df["system_size_kw_median"],
        "price_n": df["price_N"],
        "system_size_n": df["system_size_N"],
    }).sort_values(["state", "year"]).reset_index(drop=True)

    out["pct_change_price"] = out.groupby("state")["median_price_per_kw"].pct_change(fill_method=None) * 100
    return out[["state", "year", "median_price_per_kw", "median_system_size_kw", "pct_change_price",
                "price_n", "system_size_n"]]


# Minimum number of systems behind a median for latest_by_state to use it.
# Some recent state medians rest on a few dozen systems (e.g. MA and MD in
# 2025, after most of their data stopped reaching LBNL) and aren't reliable.
MIN_SYSTEMS = 200


def latest_by_state(costs: pd.DataFrame, years=(2025, 2024), min_systems: int = MIN_SYSTEMS) -> pd.DataFrame:
    """
    Most recent median price and system size per state, each taken from the
    first year in `years` whose median is based on at least `min_systems`
    systems (so price and size can come from different years). A state with
    no qualifying year is left blank.

    Returns
    -------
    pd.DataFrame
        Columns: state, median_price_per_kw, price_year, median_system_size_kw, size_year
    """
    out = pd.DataFrame({"state": sorted(costs["state"].unique())})
    for col, n_col, year_col in (("median_price_per_kw", "price_n", "price_year"),
                                 ("median_system_size_kw", "system_size_n", "size_year")):
        have = costs[costs["year"].isin(years) & costs[col].notna() & (costs[n_col] >= min_systems)].copy()
        have["rank"] = have["year"].map({y: i for i, y in enumerate(years)})
        latest = have.sort_values("rank").drop_duplicates("state")[["state", col, "year"]]
        out = out.merge(latest.rename(columns={"year": year_col}), on="state", how="left")
    out[["price_year", "size_year"]] = out[["price_year", "size_year"]].astype("Int64")
    return out

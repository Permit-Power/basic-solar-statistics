"""
storage_attachment_rate.py

Process Ohm Analytics monthly storage attachment rates (share of residential
solar installations that include storage) by state.

The raw file is wide: one row per state, one column per month ("Jan-19" …),
with values formatted as percent strings and blanks for months with no data.

Two public entry points:

    load_attachment_rates(path)
        -> long format: state, month, attachment_rate

    summarize_attachment_rate(long_df, windows=(6, 12))
        -> one row per state with mean/median/std/min/max and month counts
           over the trailing N months, for each window in `windows`

NOTES:
- Rates are returned as fractions (0-1), not percents.
- The source file carries no denominator (installs per month), so the spread
  columns are the only available signal for how thin a state's sample is.
  A state whose monthly rate swings between 0% and 100% is reporting on a
  handful of installs, not moving its market.
- Months are parsed to the first day of the month (Timestamp).
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


ID_COLS = ["AddressStateId", "State"]


def _parse_percent(series: pd.Series) -> pd.Series:
    """Convert '4.55%' / '' / NaN to float fraction (0.0455 / NaN)."""
    s = series.astype(str).str.strip().str.rstrip("%")
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(s, errors="coerce") / 100.0


def load_attachment_rates(
    path: str = "../data/ohm_analytics_storage_attachment_rate.csv",
    state_abbr_path: str | None = "../data/state_name_abbr.csv",
) -> pd.DataFrame:
    """
    Read the wide Ohm attachment-rate file and return long format.

    Returns
    -------
    pd.DataFrame
        Columns: state (abbr, if mapping available), state_name, month, attachment_rate
        Rows with no reported rate are dropped.
    """
    raw = pd.read_csv(path)

    month_cols = [c for c in raw.columns if c not in ID_COLS]

    long = raw.melt(
        id_vars=["State"],
        value_vars=month_cols,
        var_name="month",
        value_name="attachment_rate",
    ).rename(columns={"State": "state_name"})

    long["attachment_rate"] = _parse_percent(long["attachment_rate"])
    long["month"] = pd.to_datetime(long["month"], format="%b-%y")

    long = long.dropna(subset=["attachment_rate"])

    if state_abbr_path is not None:
        name_abbr = pd.read_csv(state_abbr_path)
        # The Ohm file writes DC as "Washington, DC"
        long["state_name"] = long["state_name"].replace(
            {"Washington, DC": "District of Columbia"}
        )
        long = long.merge(
            name_abbr.rename(columns={"state": "state_name", "state_abbr": "state"}),
            on="state_name",
            how="left",
        )

    return long.sort_values(["state_name", "month"]).reset_index(drop=True)


def summarize_attachment_rate(
    long_df: pd.DataFrame,
    windows: Iterable[int] = (6, 12),
    asof: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """
    Summarize each state's attachment rate over trailing windows.

    Parameters
    ----------
    long_df : pd.DataFrame
        Output of load_attachment_rates().
    windows : iterable of int
        Trailing window lengths in months (e.g. (6, 12)).
    asof : optional
        Last month to include. Defaults to the latest month in the data.

    Returns
    -------
    pd.DataFrame
        One row per state. For each window N:
            mean_{N}m, median_{N}m, std_{N}m, min_{N}m, max_{N}m,
            range_{N}m (max - min), months_reported_{N}m
        plus latest_month and latest_rate.
    """
    df = long_df.copy()
    last_month = pd.Timestamp(asof) if asof is not None else df["month"].max()

    keys = ["state", "state_name"] if "state" in df.columns else ["state_name"]

    out: pd.DataFrame | None = None

    for n in windows:
        start = last_month - pd.DateOffset(months=n - 1)
        win = df[(df["month"] >= start) & (df["month"] <= last_month)]

        agg = (
            win.groupby(keys)["attachment_rate"]
            .agg(
                **{
                    f"mean_{n}m": "mean",
                    f"median_{n}m": "median",
                    f"std_{n}m": "std",
                    f"min_{n}m": "min",
                    f"max_{n}m": "max",
                    f"months_reported_{n}m": "count",
                }
            )
            .reset_index()
        )
        agg[f"range_{n}m"] = agg[f"max_{n}m"] - agg[f"min_{n}m"]

        # Small-sample tells:
        #   pct_months_0_or_100 - share of months reported as exactly 0% or 100%
        #   hits_0_and_100      - the window contains both a 0% and a 100% month,
        #                         which no real market does; it means the month's
        #                         denominator was a couple of installs.
        # The second is the stricter test: a genuinely low-attachment state (NJ,
        # Iowa) posts legitimate 0% months but never a 100% month.
        degenerate = (
            win.assign(
                _deg=win["attachment_rate"].isin([0.0, 1.0]),
                _zero=win["attachment_rate"].eq(0.0),
                _hund=win["attachment_rate"].eq(1.0),
            )
            .groupby(keys)
            .agg(**{
                f"pct_months_0_or_100_{n}m": ("_deg", "mean"),
                "_any_zero": ("_zero", "any"),
                "_any_hund": ("_hund", "any"),
            })
            .reset_index()
        )
        degenerate[f"hits_0_and_100_{n}m"] = (
            degenerate["_any_zero"] & degenerate["_any_hund"]
        )
        degenerate = degenerate.drop(columns=["_any_zero", "_any_hund"])
        agg = agg.merge(degenerate, on=keys, how="left")

        out = agg if out is None else out.merge(agg, on=keys, how="outer")

    latest = (
        df[df["month"] == last_month][keys + ["attachment_rate"]]
        .rename(columns={"attachment_rate": "latest_rate"})
    )
    out = out.merge(latest, on=keys, how="left")
    out["latest_month"] = last_month

    return out.sort_values(keys[0]).reset_index(drop=True)


def attach_install_counts(
    summary: pd.DataFrame,
    capacity_csv: str = "../output_csvs/solar_storage_capacity_installations_by_state_sector.csv",
    sector: str = "Residential",
) -> pd.DataFrame:
    """
    Join EIA 861M residential install counts onto an attachment-rate summary,
    as a proxy for how large each state's monthly sample is.

    Adds:
        cumulative_installs      cumulative residential PV customers, latest year
        annual_new_installs      latest year minus prior year (new installs)
        monthly_new_installs     annual_new_installs / 12
    """
    cap = pd.read_csv(capacity_csv)
    cap = cap[cap["sector"] == sector].copy()

    # A few states carry inf in pv_customers (non-net estimation divides by a
    # NEM average system size of 0). Treat those as missing rather than
    # propagating inf through the deltas.
    cap["pv_customers"] = cap["pv_customers"].replace([np.inf, -np.inf], np.nan)

    latest_year = cap["year"].max()
    prior_year = latest_year - 1

    latest = cap[cap["year"] == latest_year][
        ["state", "pv_customers", "pv_capacity_mw"]
    ].rename(columns={"pv_customers": "cumulative_installs"})
    prior = cap[cap["year"] == prior_year][
        ["state", "pv_customers", "pv_capacity_mw"]
    ].rename(
        columns={"pv_customers": "prior_installs", "pv_capacity_mw": "prior_capacity_mw"}
    )

    counts = latest.merge(prior, on="state", how="left")
    counts["annual_new_installs"] = counts["cumulative_installs"] - counts["prior_installs"]
    counts["monthly_new_installs"] = counts["annual_new_installs"] / 12.0
    # Capacity delta is a backup sample proxy where pv_customers is broken.
    counts["annual_new_capacity_mw"] = (
        counts["pv_capacity_mw"] - counts["prior_capacity_mw"]
    )
    counts = counts.drop(columns=["prior_installs", "prior_capacity_mw", "pv_capacity_mw"])

    return summary.merge(counts, on="state", how="left")

"""
process_solartrace_timelines.py

Process the Solar TRACE dataset into two tidy long-format DataFrames.

Source sheets (SolarTRACE Dataset v9-9-2025.xlsx):
    - AHJ-Utility Timelines  → utility_df  (weighted median aggregation AHJ → utility)
    - State-Level Timelines  → state_df    (reshape/rename only)

utility_df columns:
    state, utility, eia_id, year, size_class, tech_class,
    installs, permit_time_days, pre_install_ix_days,
    inspection_time_days, final_ix_to_pto_days

state_df columns:
    state, year, size_class, tech_class,
    installs, permit_time_days, pre_install_ix_days, install_time_days,
    inspection_time_days, final_ix_to_pto_days, project_time_days

size_class values : "0_10kw", "10_20kw"
tech_class values : "pv_only", "pv_storage"
"""

import re

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AHJ_SHEET = "AHJ-Utility Timelines"
_STATE_SHEET = "State-Level Timelines"

_METRIC_MAP = {
    "Installs": "installs",
    "Median AHJ Permit Time": "permit_time_days",
    "Median Pre-Install IX Time": "pre_install_ix_days",
    "Median Install Time": "install_time_days",
    "Median Inspection Time": "inspection_time_days",
    "Median Final IX to PTO": "final_ix_to_pto_days",
    "Median Project Time": "project_time_days",
}

_SIZE_MAP = {"0-10kW": "0_10kw", "10-20kW": "10_20kw"}
_TECH_MAP = {"PV Only": "pv_only", "PV+Storage": "pv_storage"}

_COL_RE = re.compile(
    r"^(Installs"
    r"|Median AHJ Permit Time"
    r"|Median Pre-Install IX Time"
    r"|Median Install Time"
    r"|Median Inspection Time"
    r"|Median Final IX to PTO"
    r"|Median Project Time)"
    r" (\d{4}) (0-10kW|10-20kW) (PV Only|PV\+Storage)$"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_data_columns(columns):
    """Return list of (col_name, metric, year, size_class, tech_class) for matching columns."""
    parsed = []
    for col in columns:
        m = _COL_RE.match(col)
        if m:
            metric = _METRIC_MAP[m.group(1)]
            year = int(m.group(2))
            size = _SIZE_MAP[m.group(3)]
            tech = _TECH_MAP[m.group(4)]
            parsed.append((col, metric, year, size, tech))
    return parsed


def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
    df = pd.DataFrame({"v": values, "w": weights}).dropna(subset=["v"])
    if len(df) == 0 or df["w"].sum() == 0:
        return float("nan")
    df = df.sort_values("v")
    cum_w = df["w"].cumsum()
    cutoff = df["w"].sum() * 0.5
    return df.loc[cum_w >= cutoff, "v"].iloc[0]


# ---------------------------------------------------------------------------
# AHJ sheet → utility_df
# ---------------------------------------------------------------------------


def _reshape_ahj_to_long(ahj: pd.DataFrame) -> pd.DataFrame:
    """Melt AHJ-Utility sheet to one row per (AHJ × year × size_class × tech_class)."""
    id_cols = ["state", "ahj", "geo_id", "utility", "eia_id"]
    parsed = _parse_data_columns(ahj.columns)

    records = []
    for _, row in ahj.iterrows():
        base = {c: row[c] for c in id_cols if c in ahj.columns}
        groups: dict = {}
        for col, metric, year, size, tech in parsed:
            groups.setdefault((year, size, tech), {})[metric] = row[col]
        for (year, size, tech), metrics in groups.items():
            records.append({**base, "year": year, "size_class": size, "tech_class": tech, **metrics})

    return pd.DataFrame(records)


def compute_utility_timelines(ahj: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate AHJ-level data to utility level using install-weighted medians.

    Groups by (state, utility, eia_id, year, size_class, tech_class).
    Rows with zero installs are dropped.

    Returns
    -------
    pd.DataFrame
    """
    long = _reshape_ahj_to_long(ahj)

    timeline_cols = [
        "permit_time_days",
        "pre_install_ix_days",
        "inspection_time_days",
        "final_ix_to_pto_days",
    ]

    group_keys = ["state", "utility", "eia_id", "year", "size_class", "tech_class"]
    results = []

    for keys, sub in long.groupby(group_keys, dropna=False):
        weights = pd.to_numeric(sub["installs"], errors="coerce").fillna(0)
        total_installs = weights.sum()
        if total_installs == 0:
            continue

        rec = dict(zip(group_keys, keys))
        rec["installs"] = total_installs

        for col in timeline_cols:
            if col in sub.columns:
                values = pd.to_numeric(sub[col], errors="coerce")
                rec[col] = _weighted_median(values, weights)
                rec[f"{col}_installs"] = weights[values.notna()].sum()

        results.append(rec)

    col_order = ["state", "utility", "eia_id", "year", "size_class", "tech_class", "installs"]
    for col in timeline_cols:
        col_order += [col, f"{col}_installs"]
    df = pd.DataFrame(results)
    present = [c for c in col_order if c in df.columns]
    return df[present].sort_values(["state", "utility", "year", "size_class", "tech_class"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# State sheet → state_df
# ---------------------------------------------------------------------------


def compute_state_timelines(state: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the State-Level Timelines sheet into a tidy long-format DataFrame.
    No aggregation — values come directly from the source sheet.

    Returns
    -------
    pd.DataFrame
    """
    parsed = _parse_data_columns(state.columns)

    records = []
    for _, row in state.iterrows():
        groups: dict = {}
        for col, metric, year, size, tech in parsed:
            groups.setdefault((year, size, tech), {})[metric] = row[col]
        for (year, size, tech), metrics in groups.items():
            records.append({"state": row["state"], "year": year, "size_class": size, "tech_class": tech, **metrics})

    col_order = [
        "state", "year", "size_class", "tech_class",
        "installs", "permit_time_days", "pre_install_ix_days",
        "install_time_days", "inspection_time_days",
        "final_ix_to_pto_days", "project_time_days",
    ]
    df = pd.DataFrame(records)
    present = [c for c in col_order if c in df.columns]
    return df[present].sort_values(["state", "year", "size_class", "tech_class"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def load_solartrace(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load AHJ-Utility Timelines and State-Level Timelines sheets from the xlsx."""
    ahj = pd.read_excel(path, sheet_name=_AHJ_SHEET)
    state = pd.read_excel(path, sheet_name=_STATE_SHEET)
    return ahj, state


def run_pipeline(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load xlsx → utility_df, state_df.

    Parameters
    ----------
    path : str
        Path to the SolarTRACE xlsx file.

    Returns
    -------
    utility_df, state_df
    """
    ahj, state = load_solartrace(path)
    utility_df = compute_utility_timelines(ahj)
    state_df = compute_state_timelines(state)
    return utility_df, state_df

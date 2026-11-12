"""
process_solartrace_timelines.py

Process NLR's Solar TRACE timelines file into tidy long-format DataFrames.

Source: data/SolarTRACE Residential PII Requirements Timelines Dataset_Final.xlsx
(NLR, completed 8/31/2026): residential systems up to 20 kW, 2017-2025, PV only
and PV+storage. Medians are in business days; NLR leaves a median blank below
10 projects. Each sheet has two header rows (display names, then coded names
like `2024_pv_med_permit`); this module reads the coded row.
    - State-Level Timelines  -> state_df   (reshape only)
    - AHJ-Utility Timelines  -> utility_df (install-weighted median over AHJs)
                             -> ahj_df     (install-weighted median over utilities)

Timeline values are treated as per AHJ-utility pair: IX values vary across
the AHJs a utility serves, and permit/inspection values vary across a few
utilities within one AHJ, so both roll-ups weight by installs.

state_df columns:
    state, year, tech_class, installs, permit_time_days, pre_install_ix_days,
    inspection_time_days, final_ix_to_pto_days, project_time_days

utility_df columns:
    state, utility, eia_id, year, tech_class, installs, then for each of
    permit_time_days, pre_install_ix_days, inspection_time_days,
    final_ix_to_pto_days: the median and `<col>_installs` (installs in the
    AHJs that report it)

ahj_df columns:
    state, ahj, geo_id, year, tech_class, installs, permit_time_days,
    permit_time_days_installs, inspection_time_days, inspection_time_days_installs

latest_df (from latest_by_state) columns, one row per state:
    for each tech prefix (pv_only_, pv_storage_): year, installs and the four
    shown medians from the chosen year, then the same five fields from 2024
    with a _2024 suffix

tech_class values: "pv_only", "pv_storage"
"""

import re

import pandas as pd

_AHJ_SHEET = "AHJ-Utility Timelines"
_STATE_SHEET = "State-Level Timelines"

_METRIC_MAP = {
    "install_count": "installs",
    "med_permit": "permit_time_days",
    "med_pre_IX": "pre_install_ix_days",
    "med_inspection": "inspection_time_days",
    "med_post_IX": "final_ix_to_pto_days",
    "med_total_project": "project_time_days",
}
_TECH_MAP = {"pv": "pv_only", "pv+st": "pv_storage"}
_COL_RE = re.compile(r"^(\d{4})_(pv|pv\+st)_(" + "|".join(_METRIC_MAP) + r")$")

# The four medians shown per state (total project time is not shown).
SHOWN = ["permit_time_days", "pre_install_ix_days", "inspection_time_days", "final_ix_to_pto_days"]
# Year rule (Wael, 2026-11-12): latest year up to LATEST_YEAR with at least
# MIN_INSTALLS installs and all four SHOWN medians, chosen per state and tech.
LATEST_YEAR = 2025
MIN_INSTALLS = 100
HOVER_YEAR = 2024


def _parse_data_columns(columns):
    """Return list of (col_name, metric, year, tech_class) for matching columns."""
    parsed = []
    for col in columns:
        m = _COL_RE.match(str(col))
        if m:
            parsed.append((col, _METRIC_MAP[m.group(3)], int(m.group(1)), _TECH_MAP[m.group(2)]))
    return parsed


def _reshape_to_long(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """Melt a sheet to one row per (id_cols x year x tech_class)."""
    parsed = _parse_data_columns(df.columns)
    records = []
    for _, row in df.iterrows():
        base = {c: row[c] for c in id_cols}
        groups: dict = {}
        for col, metric, year, tech in parsed:
            groups.setdefault((year, tech), {})[metric] = row[col]
        for (year, tech), metrics in groups.items():
            records.append({**base, "year": year, "tech_class": tech, **metrics})
    return pd.DataFrame(records)


def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
    df = pd.DataFrame({"v": values, "w": weights}).dropna(subset=["v"])
    if len(df) == 0 or df["w"].sum() == 0:
        return float("nan")
    df = df.sort_values("v")
    cum_w = df["w"].cumsum()
    cutoff = df["w"].sum() * 0.5
    return df.loc[cum_w >= cutoff, "v"].iloc[0]


def _rollup(long: pd.DataFrame, group_keys: list[str], timeline_cols: list[str]) -> pd.DataFrame:
    """Install-weighted median of each timeline column per group; zero-install groups dropped."""
    results = []
    for keys, sub in long.groupby(group_keys, dropna=False):
        weights = pd.to_numeric(sub["installs"], errors="coerce").fillna(0)
        if weights.sum() == 0:
            continue
        rec = dict(zip(group_keys, keys))
        rec["installs"] = weights.sum()
        for col in timeline_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            rec[col] = _weighted_median(values, weights)
            rec[f"{col}_installs"] = weights[values.notna()].sum()
        results.append(rec)
    return pd.DataFrame(results).sort_values(group_keys).reset_index(drop=True)


def compute_utility_timelines(ahj_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate AHJ-utility pairs to utility level using install-weighted medians."""
    return _rollup(ahj_long, ["state", "utility", "eia_id", "year", "tech_class"], SHOWN)


def compute_ahj_timelines(ahj_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate AHJ-utility pairs to AHJ level (permit and inspection only)."""
    return _rollup(ahj_long, ["state", "ahj", "geo_id", "year", "tech_class"],
                   ["permit_time_days", "inspection_time_days"])


def compute_state_timelines(state: pd.DataFrame) -> pd.DataFrame:
    """Reshape the State-Level Timelines sheet; values come directly from the source."""
    df = _reshape_to_long(state, ["state"])
    cols = ["state", "year", "tech_class", "installs"] + SHOWN + ["project_time_days"]
    return df[cols].sort_values(["state", "year", "tech_class"]).reset_index(drop=True)


def latest_by_state(state_df: pd.DataFrame, latest_year: int = LATEST_YEAR,
                    min_installs: int = MIN_INSTALLS, hover_year: int = HOVER_YEAR) -> pd.DataFrame:
    """
    Per state and tech class, the latest year up to `latest_year` with at least
    `min_installs` installs and all four SHOWN medians; a state with no such
    year is left blank. Also carries every state's `hover_year` values
    (whether or not that year passes the rule) for comparison.
    """
    fields = ["installs"] + SHOWN
    out = pd.DataFrame({"state": sorted(state_df["state"].unique())})
    for tech in ["pv_only", "pv_storage"]:
        t = state_df[state_df["tech_class"] == tech]
        ok = t[(t["year"] <= latest_year) & (t["installs"] >= min_installs) & t[SHOWN].notna().all(axis=1)]
        chosen = ok.sort_values("year").drop_duplicates("state", keep="last")[["state", "year"] + fields]
        hover = t[t["year"] == hover_year][["state"] + fields]
        out = (out
               .merge(chosen.rename(columns={c: f"{tech}_{c}" for c in ["year"] + fields}), on="state", how="left")
               .merge(hover.rename(columns={f: f"{tech}_{f}_{hover_year}" for f in fields}), on="state", how="left"))
        out[f"{tech}_year"] = out[f"{tech}_year"].astype("Int64")
    return out


def load_solartrace(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two timeline sheets, using the coded (second) header row."""
    ahj = pd.read_excel(path, sheet_name=_AHJ_SHEET, header=1)
    ahj.columns = ["state", "ahj", "geo_id", "utility", "eia_id"] + list(ahj.columns[5:])
    state = pd.read_excel(path, sheet_name=_STATE_SHEET, header=1)
    state = state.rename(columns={state.columns[0]: "state"})
    return ahj, state


def run_pipeline(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full pipeline: xlsx -> utility_df, state_df, ahj_df, latest_df."""
    ahj, state = load_solartrace(path)
    ahj_long = _reshape_to_long(ahj, ["state", "ahj", "geo_id", "utility", "eia_id"])
    utility_df = compute_utility_timelines(ahj_long)
    ahj_df = compute_ahj_timelines(ahj_long)
    state_df = compute_state_timelines(state)
    latest_df = latest_by_state(state_df)
    return utility_df, state_df, ahj_df, latest_df

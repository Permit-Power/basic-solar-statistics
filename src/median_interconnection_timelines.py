"""
median_interconnection_timelines.py

Compute annual population-weighted median and weighted average 
interconnection timelines (Final IX to PTO) for PV-only and PV+Storage 
systems under 10 kW.

Inputs:
-------
df : DataFrame containing AHJ-level fields, including:
    - state
    - ahj
    - geoid
    - For each year YYYY in 2017–2023:
        Installs YYYY 0-10kW PV Only
        Median Final IX to PTO YYYY 0-10kW PV Only
        Installs YYYY 0-10kW PV+Storage
        Median Final IX to PTO YYYY 0-10kW PV+Storage

Outputs:
--------
ahj_df : long-form dataframe with:
    state, ahj, geoid, year, tech_class, 
    installs, median_ix, weighted_avg_ix, weighted_median_ix

state_df : state-level aggregation with same metrics.

Only 0–10 kW classes are included; 10–20 kW fields are ignored.

"""

import pandas as pd


# ---------------------------------------------------------------------
# Weighted Median Helper
# ---------------------------------------------------------------------
def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
    """
    Compute weighted median:
        value where cumulative weight >= 50% of total weight.

    values: timelines
    weights: installs (>= 0)
    """
    df = pd.DataFrame({"v": values, "w": weights}).dropna(subset=["v"])
    if len(df) == 0 or df["w"].sum() == 0:
        return float("nan")

    df = df.sort_values("v")
    cum_w = df["w"].cumsum()
    cutoff = df["w"].sum() * 0.5
    return df.loc[cum_w >= cutoff, "v"].iloc[0]


# ---------------------------------------------------------------------
# Long-Format Expansion
# ---------------------------------------------------------------------
def reshape_interconnection_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the wide interconnection timeline table into a tidy long format:
    
    Returns rows with:
        state, ahj, geoid, year, tech_class, installs, median_ix
    """

    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
    techs = {
        "pv_only": {
            "inst_prefix": "Installs {} 0-10kW PV Only",
            "ix_prefix": "Median Final IX to PTO {} 0-10kW PV Only",
        },
        "pv_storage": {
            "inst_prefix": "Installs {} 0-10kW PV+Storage",
            "ix_prefix": "Median Final IX to PTO {} 0-10kW PV+Storage",
        },
    }

    records = []

    for _, row in df.iterrows():
        for year in years:
            for tech_class, prefixes in techs.items():

                inst_col = prefixes["inst_prefix"].format(year)
                ix_col = prefixes["ix_prefix"].format(year)

                if inst_col not in df.columns or ix_col not in df.columns:
                    continue

                installs = row.get(inst_col, float("nan"))
                median_ix = row.get(ix_col, float("nan"))

                # ignore rows with no installs or NAs
                if pd.isna(installs) or installs <= 0:
                    continue

                records.append({
                    "state": row["state"],
                    "ahj": row["ahj"],
                    "geoid": row["geoid"],
                    "year": year,
                    "tech_class": tech_class,
                    "installs": installs,
                    "median_ix": median_ix,
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Compute AHJ-Level Weighted Stats
# ---------------------------------------------------------------------
def compute_ahj_level_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    AHJ-level summary:
        weighted average = sum(median_ix * installs) / sum(installs)
        weighted median = custom weighted median

    Returns:
        state, ahj, geoid, year, tech_class,
        weighted_median_ix, weighted_average_ix, installs
    """
    results = []

    for keys, sub in long_df.groupby(["state", "ahj", "geoid", "year", "tech_class"]):

        total_inst = sub["installs"].sum()
        if total_inst == 0:
            continue

        weighted_avg = (sub["median_ix"] * sub["installs"]).sum() / total_inst
        weighted_med = _weighted_median(sub["median_ix"], sub["installs"])

        results.append({
            "state": keys[0],
            "ahj": keys[1],
            "geoid": keys[2],
            "year": keys[3],
            "tech_class": keys[4],
            "weighted_average_ix": weighted_avg,
            "weighted_median_ix": weighted_med,
            "total_installs": total_inst,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Compute State-Level Weighted Stats
# ---------------------------------------------------------------------
def compute_state_level_metrics(ahj_df: pd.DataFrame) -> pd.DataFrame:
    """
    State-level weighted metrics by year + tech_class.
    Weighted by AHJ installs.

    Returns:
        state, year, tech_class,
        weighted_average_ix, weighted_median_ix, total_installs
    """
    results = []

    for keys, sub in ahj_df.groupby(["state", "year", "tech_class"]):

        total_inst = sub["total_installs"].sum()
        if total_inst == 0:
            continue

        weighted_avg = (sub["weighted_average_ix"] * sub["total_installs"]).sum() / total_inst
        weighted_med = _weighted_median(sub["weighted_median_ix"], sub["total_installs"])

        results.append({
            "state": keys[0],
            "year": keys[1],
            "tech_class": keys[2],
            "weighted_average_ix": weighted_avg,
            "weighted_median_ix": weighted_med,
            "total_installs": total_inst,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------
def run_pipeline(df: pd.DataFrame):
    """
    Full pipeline:
        1. Convert wide table into long format
        2. Produce AHJ-level weighted medians/averages
        3. Produce state-level weighted medians/averages

    Returns:
        ahj_df, state_df
    """
    long_df = reshape_interconnection_df(df)
    ahj_df = compute_ahj_level_metrics(long_df)
    state_df = compute_state_level_metrics(ahj_df)

    return ahj_df, state_df

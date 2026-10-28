"""
energy_insecurity.py

Share of households with any energy insecurity, by state, from the EIA 2024
Residential Energy Consumption Survey (RECS) microdata.

Uses EIA's definition from Table HC11.1 ("Household energy insecurity, 2024",
preliminary release March 2026). A household counts if, in the past 12 months,
it reported any of:
  - reducing or forgoing food or medicine to pay energy costs (SCALEB > 0)
  - keeping the home at an unhealthy temperature (SCALEG > 0)
  - receiving a disconnect or delivery stop notice (SCALEE > 0)
  - being unable to use heating equipment: broken and unaffordable to repair,
    or electricity, natural gas or bulk fuel cut off for nonpayment
    (NOHEATBROKE, NOHEATEL, NOHEATNG, NOHEATBULK)
  - being unable to use air-conditioning equipment: broken and unaffordable to
    repair, or electricity cut off for nonpayment (NOACBROKE, NOACEL)

"Unable to pay an energy bill" (SCALEP) is in the survey but not in EIA's
definition. Including it would raise the national total from 43.56M to 45.45M.

The committed input, data/recs2024_energy_insecurity_microdata.csv, is a trimmed
copy of the public microdata file (recs2024_public_microdata_v1.csv, kept local)
made with trim_microdata().
"""

import pandas as pd

SCALE_COLS = ["SCALEB", "SCALEG", "SCALEE"]
YES_NO_COLS = ["NOHEATBROKE", "NOHEATEL", "NOHEATNG", "NOHEATBULK", "NOACBROKE", "NOACEL"]
KEEP_COLS = ["DOEID", "state_postal", "NWEIGHT"] + SCALE_COLS + YES_NO_COLS

# Table HC11.1, "All homes" row (million households)
EIA_TOTAL_HOUSEHOLDS_M = 132.54
EIA_INSECURE_HOUSEHOLDS_M = 43.56


def trim_microdata(src: str, dst: str) -> None:
    """Keep only the columns compute_energy_insecurity() needs."""
    pd.read_csv(src, usecols=KEEP_COLS)[KEEP_COLS].to_csv(dst, index=False)


def compute_energy_insecurity(df: pd.DataFrame) -> pd.DataFrame:
    """Weighted households and share with any energy insecurity, by state."""
    insecure = (df[SCALE_COLS] > 0).any(axis=1) | (df[YES_NO_COLS] == 1).any(axis=1)

    # National totals must match the published table
    total_m = df["NWEIGHT"].sum() / 1e6
    insecure_m = df.loc[insecure, "NWEIGHT"].sum() / 1e6
    assert round(total_m, 2) == EIA_TOTAL_HOUSEHOLDS_M, total_m
    assert round(insecure_m, 2) == EIA_INSECURE_HOUSEHOLDS_M, insecure_m

    out = (
        df.assign(insecure_weight=df["NWEIGHT"] * insecure)
        .groupby("state_postal")
        .agg(households=("NWEIGHT", "sum"), energy_insecure_households=("insecure_weight", "sum"))
        .reset_index()
        .rename(columns={"state_postal": "state"})
    )
    out["energy_insecure_pct"] = out["energy_insecure_households"] / out["households"]
    assert len(out) == 51  # 50 states + DC
    return out

"""
resstock_solar_access.py

Module to estimate total households and households eligible for solar
from ResStock metadata.

Eligibility criteria:
- in.tenure == "Owner"
- in.geometry_building_type_acs in {
    "Single-Family Detached", "Single-Family Attached", "2 Unit", "3 or 4 Unit"
  }

Input: pandas DataFrame with required columns.
Output: dictionary with totals and percentage.
"""

import pandas as pd

# Allowed values
TENURE_OK = {"Owner"}
BUILDING_OK = {
    "Single-Family Detached",
    "Single-Family Attached",
    "2 Unit",
    "3 or 4 Unit",
}


def compute_solar_eligibility(df: pd.DataFrame, by_state: bool = True):
    """
    Compute total households and households eligible for solar.

    Parameters
    ----------
    df : pandas.DataFrame
        ResStock metadata with columns:
        - weight
        - in.orientation
        - in.tenure
        - in.geometry_building_type_acs

    Returns
    -------
    dict
        {
          "total_households": float,
          "eligible_households": float,
          "eligible_pct": float,
        }
    """

    required = ["weight", "in.orientation", "in.tenure", "in.geometry_building_type_acs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Total households
    total_households = df["weight"].sum()

    # Eligibility mask
    mask = (
        df["in.tenure"].isin(TENURE_OK)
        & df["in.geometry_building_type_acs"].isin(BUILDING_OK)
    )

    eligible_households = df.loc[mask, "weight"].sum()

    eligible_pct = 0.0
    if total_households > 0:
        eligible_pct = eligible_households / total_households

        if not by_state:
            return {
                "total_households": float(total_households),
                "eligible_households": float(eligible_households),
                "eligible_pct": float(eligible_pct),
            }

    # --- State-level aggregation ---
    if "in.state" not in df.columns:
        raise ValueError("Column 'in.state' required for state-level aggregation.")

    # Precompute eligibility mask for all rows
    df = df.copy()
    df["_eligible"] = mask

    grouped = df.groupby("in.state")

    out = {}
    for state, g in grouped:
        tot = g["weight"].sum()
        elig = g.loc[g["_eligible"], "weight"].sum()
        pct = elig / tot if tot > 0 else 0
        out[state] = {
            "total_households": float(tot),
            "eligible_households": float(elig),
            "eligible_pct": float(pct),
        }
        # Convert to DataFrame
    return pd.DataFrame.from_dict(out, orient="index").reset_index().rename(columns={"index": "state"})

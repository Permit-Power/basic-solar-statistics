"""
resstock_solar_access.py

Module to estimate total households and households eligible for solar
using a two-step funnel:

Step 1 — ResStock filter (compute_solar_eligibility):
  Filters to owner-occupied, residential building types using ResStock
  metadata weights. Produces eligible_households by state.

Step 2 — NREL rooftop suitability filter (apply_rooftop_suitability):
  Applies a state-level rooftop suitability rate derived from NREL's
  ZIP-code-level dataset (Gagnon et al. 2016, NREL/TP-6A20-65298).
  The rate represents the share of small residential buildings with at
  least one roof plane suitable for PV, accounting for shading, tilt,
  azimuth, and minimum usable area. This is a further haircut on the
  ResStock-eligible population.

  Note: The NREL ZIP-level data covers small buildings only (< 5,000 ft²).
  This is consistent with the residential scope of the ResStock filter.
"""

import logging
import warnings

import pandas as pd

logger = logging.getLogger(__name__)

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
    return pd.DataFrame.from_dict(out, orient="index").reset_index().rename(columns={"index": "state"})


def apply_rooftop_suitability(
    state_df: pd.DataFrame,
    suitability_path: str = "../data/nrel_solar_suitability_by_zip.csv",
) -> pd.DataFrame:
    """
    Apply a rooftop suitability haircut to ResStock-eligible household counts.

    Uses NREL's ZIP-code-level rooftop suitability dataset (Gagnon et al. 2016)
    to compute the share of small residential buildings with at least one roof
    plane suitable for PV in each state. This rate is applied as a multiplier
    to eligible_households and eligible_pct.

    Parameters
    ----------
    state_df : pd.DataFrame
        Output of compute_solar_eligibility(by_state=True). Must contain columns:
        state, total_households, eligible_households, eligible_pct.

    suitability_path : str
        Path to nrel_solar_suitability_by_zip.csv.

    Returns
    -------
    pd.DataFrame
        Same schema as state_df, with eligible_households and eligible_pct
        updated to reflect the rooftop suitability haircut.
    """
    import pgeocode

    suit = pd.read_csv(suitability_path, dtype={"zip": str})
    suit["zip"] = suit["zip"].str.zfill(5)

    # Drop rows where suitability rate is zero or null (avoid division by zero)
    suit = suit[suit["pct.suitable"].notna() & (suit["pct.suitable"] > 0)].copy()

    # Map ZIP codes to states via pgeocode
    nomi = pgeocode.Nominatim("us")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zip_info = nomi.query_postal_code(suit["zip"].tolist())
    suit["state"] = zip_info["state_code"].values

    suit = suit[suit["state"].notna()]

    # Back-calculate total buildings per ZIP
    suit["total_buildings"] = suit["nbld"] / suit["pct.suitable"]

    # Aggregate to state level
    state_suit = (
        suit.groupby("state")
        .agg(suitable_buildings=("nbld", "sum"), total_buildings=("total_buildings", "sum"))
        .reset_index()
    )
    state_suit["rooftop_suitability_rate"] = (
        state_suit["suitable_buildings"] / state_suit["total_buildings"]
    )

    # Merge and apply haircut
    result = state_df.copy().merge(
        state_suit[["state", "rooftop_suitability_rate"]], on="state", how="left"
    )

    unmatched = result[result["rooftop_suitability_rate"].isna()]["state"].tolist()
    if unmatched:
        logger.warning("No NREL suitability data for states: %s — leaving unchanged.", unmatched)
        result["rooftop_suitability_rate"] = result["rooftop_suitability_rate"].fillna(1.0)

    result["eligible_households"] = (
        result["eligible_households"] * result["rooftop_suitability_rate"]
    )
    result["eligible_pct"] = result["eligible_households"] / result["total_households"]

    return result.drop(columns=["rooftop_suitability_rate"])

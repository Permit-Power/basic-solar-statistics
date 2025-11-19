"""
ahj_permitting.py

Tools for merging AHJ-level permitting cost data with population data and
computing state-level population-weighted median permitting costs.

Expected input DataFrames:

df_permits:
    state                str
    name                 str
    geoid                str or int
    median_permit_cost   float

df_population:
    state                str
    name                 str
    geoid                str or int
    population           int or float

Outputs:
    1. ahj_df: merged AHJ-level dataframe (one row per AHJ)
    2. state_df: state-level population-weighted median permit cost

Population-weighted median:
    For each state:
        Sort AHJs by median_permit_cost
        Compute cumulative population shares
        The weighted median is the value where cumulative share ≥ 0.5
"""

import pandas as pd


def merge_ahj_permit_population(df_permits: pd.DataFrame,
                                df_population: pd.DataFrame, 
                                name_abbr: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the permits dataframe with population using state,name,geoid keys.

    First merges state abbreviations to state names

    Returns a dataframe with:
        state, name, geoid, median_permit_cost, population
    """
    required_permit_cols = {"state", "geoid", "median_permit_cost"}
    required_pop_cols = {"state", "geoid", "population"}

    # Merge state abbreviations to permitting fees df
    df_permits = (
        df_permits
        .merge(name_abbr, on = 'state_abbr')
    )

    if not required_permit_cols.issubset(df_permits.columns):
        missing = required_permit_cols - set(df_permits.columns)
        raise ValueError(f"df_permits missing required columns: {missing}")

    if not required_pop_cols.issubset(df_population.columns):
        missing = required_pop_cols - set(df_population.columns)
        raise ValueError(f"df_population missing required columns: {missing}")
    
    merged = pd.merge(
        df_permits,
        df_population,
        on=["state", "geoid"],
        how="left",
    )

    return merged


def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
    """
    Compute a weighted median.

    Assumes:
        - values and weights aligned
        - weights >= 0
    """
    df = pd.DataFrame({"v": values, "w": weights}).sort_values("v")
    cum_weights = df["w"].cumsum()
    cutoff = df["w"].sum() * 0.5
    return df.loc[cum_weights >= cutoff, "v"].iloc[0]


def compute_state_weighted_medians(ahj_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute state-level population-weighted median permit costs.
    """

    results = []

    for state, sub in ahj_df.groupby("state"):

        # Drop AHJs with no permit cost
        sub = sub.dropna(subset=["median_permit_cost"])

        if len(sub) == 0:
            results.append({
                "state": state,
                "weighted_median_permit_cost": float("nan")
            })
            continue

        median_val = _weighted_median(
            values=sub["median_permit_cost"],
            weights=sub["population"].fillna(0),
        )

        results.append({
            "state": state,
            "weighted_median_permit_cost": median_val
        })

    return pd.DataFrame(results)

def compute_state_weighted_averages(ahj_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute state-level population-weighted *average* permit costs.

    weighted_average = sum(cost_i * pop_i) / sum(pop_i)

    Returns:
        DataFrame with:
            state
            weighted_average_permit_cost
    """

    results = []

    for state, sub in ahj_df.groupby("state"):

        sub = sub.dropna(subset=["median_permit_cost"])

        if len(sub) == 0:
            results.append({
                "state": state,
                "weighted_average_permit_cost": float("nan")
            })
            continue

        weights = sub["population"].fillna(0)
        values = sub["median_permit_cost"]

        avg_val = (values * weights).sum() / weights.sum() if weights.sum() > 0 else float("nan")

        results.append({
            "state": state,
            "weighted_average_permit_cost": avg_val
        })

    return pd.DataFrame(results)



def run_pipeline(df_permits: pd.DataFrame,
                 df_population: pd.DataFrame,
                 name_abbr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute the full pipeline:

    1. Merge AHJ permit costs with population (AHJ-level output)
    2. Compute state-level population-weighted median permit costs
    3. Compute state-level population-weighted average permit costs

    Returns:
        ahj_df (AHJ-level merged data)
        median_df (state-level weighted medians)
        average_df (state-level weighted averages)
    """

    ahj_df = (
        merge_ahj_permit_population(
            df_permits, df_population[['state', 'geoid', 'population']], name_abbr)[['state', 'geoid', 'name', 'population', 'median_permit_cost']]
            .sort_values(['state', 'population'], ascending=[True,False])
    )

    median_df = compute_state_weighted_medians(ahj_df)
    average_df = compute_state_weighted_averages(ahj_df)

    state_df = (
        median_df
        .merge(average_df, on = 'state', how = 'left')
    )

    return ahj_df, state_df


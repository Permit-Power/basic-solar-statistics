"""
solar_technical_potential.py

Loads residential PV technical generation potential by state from the
NREL SLOPE tool (https://maps.nrel.gov/slope/data-viewer).
Data represents technical potential for 2020.

Source: NREL SLOPE — State and Local Planning for Energy
"""

import pandas as pd


def load_solar_technical_potential(
    path: str = "../data/techpot_baseline_state.csv",
    mapping_path: str = "../data/state_name_abbr.csv",
) -> pd.DataFrame:
    """
    Load residential PV technical generation potential by state.

    Filters to Technology == 'residential_pv', converts MWh to GWh,
    and maps full state names to two-letter abbreviations.

    Parameters
    ----------
    path : str
        Path to techpot_baseline_state.csv.
    mapping_path : str
        Path to state_name_abbr.csv (columns: state, state_abbr).

    Returns
    -------
    pd.DataFrame
        Columns: state (abbreviation), technical_potential_gwh
    """
    df = pd.read_csv(path)
    df = df[df["Technology"] == "residential_pv"].copy()

    df = df.rename(columns={
        "State Name": "state_name",
        "Technical Generation Potential - MWh MWh": "technical_potential_gwh",
    })

    df["technical_potential_gwh"] = df["technical_potential_gwh"] / 1_000

    name_abbr = pd.read_csv(mapping_path)
    df = df.merge(name_abbr, left_on="state_name", right_on="state", how="left")

    return df[["state_abbr", "technical_potential_gwh"]].rename(columns={"state_abbr": "state"}).reset_index(drop=True)

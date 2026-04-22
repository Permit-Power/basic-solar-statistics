"""
pull_solar_installations_and_capacity.py

Pulls annual residential net metering customer counts and capacity by state
from the EIA API v2 (Form EIA-861, State Electricity Profiles Table 11).
"""

import os
import requests
import pandas as pd

EIA_API_URL = "https://api.eia.gov/v2/electricity/state-electricity-profiles/net-metering/data/"


def fetch_solar_installations_and_capacity(
    api_key: str,
    year: int = 2024,
) -> pd.DataFrame:
    """
    Fetch annual residential net metering customers and capacity by state
    from EIA API v2 for a single year.

    Parameters
    ----------
    api_key : str
        EIA API key.
    year : int
        Year to pull data for. Default: 2024.

    Returns
    -------
    pd.DataFrame
        Columns: period, stateid, stateDescription, sector, capacity, customers
    """
    params = {
        "api_key": api_key,
        "frequency": "annual",
        "data[0]": "capacity",
        "data[1]": "customers",
        "facets[sector][]": "RES",
        "start": str(year),
        "end": str(year),
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }

    response = requests.get(EIA_API_URL, params=params)
    response.raise_for_status()

    data = response.json().get("response", {}).get("data", [])
    if not data:
        raise RuntimeError("No data returned from EIA API.")

    df = pd.DataFrame(data)

    for col in ["capacity", "customers"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={"stateid": "state", "customers": "solar_customers", "capacity": "solar_capacity_mw"})

    return df[["state", "solar_customers", "solar_capacity_mw"]]


def load_solar_installations_and_capacity(year: int = 2024) -> pd.DataFrame:
    """
    Load solar net metering data using EIA_API_KEY from environment.

    Returns
    -------
    pd.DataFrame
        Columns: period, stateid, stateDescription, sector, capacity, customers
    """
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        raise RuntimeError("EIA_API_KEY environment variable not set.")

    return fetch_solar_installations_and_capacity(api_key=api_key, year=year)

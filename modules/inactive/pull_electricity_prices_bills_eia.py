"""
pull_electricity_prices_bills_eia.py

Pulls annual residential electricity retail sales data by state from the
EIA API v2 (Forms EIA-826, EIA-861, EIA-861M).

Returns customers, average price (cents/kWh), and revenue by state and year.
"""

import os
import requests
import pandas as pd

EIA_API_URL = "https://api.eia.gov/v2/electricity/retail-sales/data/"


def fetch_residential_electricity_prices(
    api_key: str,
    year: int = 2025,
) -> pd.DataFrame:
    """
    Fetch annual residential electricity retail sales data from EIA API v2
    for a single year.

    Parameters
    ----------
    api_key : str
        EIA API key.
    year : int
        Year to pull data for. Default: 2025.

    Returns
    -------
    pd.DataFrame
        Columns: period, stateid, stateDescription, customers, price, revenue
    """
    params = {
        "api_key": api_key,
        "frequency": "annual",
        "data[0]": "customers",
        "data[1]": "price",
        "data[2]": "revenue",
        "facets[sectorid][]": "RES",
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

    for col in ["customers", "price", "revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_residential_electricity_prices(year: int = 2025) -> pd.DataFrame:
    """
    Load residential electricity prices using EIA_API_KEY from environment.

    Returns
    -------
    pd.DataFrame
        Columns: period, stateid, stateDescription, customers, price, revenue
    """
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        raise RuntimeError("EIA_API_KEY environment variable not set.")

    return fetch_residential_electricity_prices(api_key=api_key, year=year)


def add_annual_bill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price_per_kwh and annual_bill columns to the DataFrame.
    EIA reports price in cents/kWh and revenue in million dollars.
    """
    df = df.copy()
    df["price_per_kwh"] = df["price"] / 100
    df["annual_bill"] = (df["revenue"] * 1_000_000) / df["customers"]
    return df

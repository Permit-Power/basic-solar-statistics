"""
pull_solar_count_and_capacity.py

Download and aggregate EIA 861M distributed solar (PV) and battery statistics
for U.S. states, combining non-net-metered and net-metered data.

Supported years: 2017–2025

Final output schema (one row per year, state, sector):
    year
    state
    sector                # 'Residential', 'Commercial', 'Industrial'
    pv_capacity_mw        # PV capacity (MW), net + non-net
    pv_customers          # PV customers (net) + non-net generators (residential only)
    battery_capacity_mw   # Battery/storage capacity (MW), net + non-net
    battery_energy_mwh    # Battery energy capacity (MWh), if available; NaN otherwise
    battery_customers     # Battery customers (net only; non-net is always 0)

NOTES:
- Only Residential, Commercial, Industrial sectors are included.
- Non-net number of generators is treated as PV customers in the Residential sector.
- Battery capacity from older "Storage" fields and newer "Battery" fields
  are summed.
- Battery energy (MWh) is only populated in years where EIA reports it.
"""

from __future__ import annotations

from io import BytesIO
from typing import Iterable, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

ARCHIVE_BASE_URL = "https://www.eia.gov/electricity/data/eia861m/archive/xls/"
CURRENT_BASE_URL = "https://www.eia.gov/electricity/data/eia861m/xls/"

SECTORS = ["Residential", "Commercial", "Industrial"]


# ---------------------------------------------------------------------------
# DOWNLOAD HELPERS
# ---------------------------------------------------------------------------

def _download_excel(url: str) -> pd.DataFrame:
    """
    Download an Excel file from a URL and return the 'Monthly_Totals-States' sheet
    as a DataFrame with a 2-level MultiIndex header.

    Parameters
    ----------
    url : str

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError if the download or parse fails.
    """
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200 or not resp.content:
        raise ValueError(f"Failed to download {url} (status {resp.status_code})")
    return pd.read_excel(
        BytesIO(resp.content),
        sheet_name="Monthly_Totals-States",
        header=[1, 2],
    )


def _download_non_net_raw(year: int) -> pd.DataFrame:
    """
    Download the non-net-metering Excel for a given year and return the raw table.

    URL patterns:
    - 2017–2024: archive/xls/non_netmetering{year}.xlsx or non_netmetering_{year}.xlsx
    - 2025:      xls/non_netmetering2025.xlsx
    """
    candidates: List[str] = []

    if year == 2025:
        candidates.append(CURRENT_BASE_URL + f"non_netmetering{year}.xlsx")
    else:
        # archive patterns
        candidates.append(ARCHIVE_BASE_URL + f"non_netmetering{year}.xlsx")
        candidates.append(ARCHIVE_BASE_URL + f"non_netmetering_{year}.xlsx")

    last_error = None
    for url in candidates:
        try:
            return _download_excel(url)
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue

    raise ValueError(f"Could not download non-net-metering file for year={year}: {last_error}")


def _download_net_raw(year: int) -> pd.DataFrame:
    """
    Download the net-metering Excel for a given year and return the raw table.

    URL patterns:
    - 2017–2024: archive/xls/net_metering{year}.xlsx or net_metering_{year}.xlsx
    - 2025:      xls/net_metering2025.xlsx
    """
    candidates: List[str] = []

    if year == 2025:
        candidates.append(CURRENT_BASE_URL + f"net_metering{year}.xlsx")
    else:
        candidates.append(ARCHIVE_BASE_URL + f"net_metering{year}.xlsx")
        candidates.append(ARCHIVE_BASE_URL + f"net_metering_{year}.xlsx")

    last_error = None
    for url in candidates:
        try:
            return _download_excel(url)
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue

    raise ValueError(f"Could not download net-metering file for year={year}: {last_error}")


# ---------------------------------------------------------------------------
# GENERIC HELPERS
# ---------------------------------------------------------------------------

def _select_december_indices(
    df_raw: pd.DataFrame,
    month_col: Tuple[str, str],
    state_col: Tuple[str, str],
) -> List[int]:
    """
    Given a raw dataframe, identify the row index for each state corresponding
    to December (month=12) if present, otherwise the latest available month.

    Month column may come from either:
    - direct ("Utility Characteristics","Month") for non-net files, or
    - a column under "Utility Characteristics" whose row-0 label is 'Month'
      for net-metering files.

    Parameters
    ----------
    df_raw : pd.DataFrame
    month_col : tuple
        Column indicating the month (top-level, sub-level).
    state_col : tuple
        Column containing state abbreviations.

    Returns
    -------
    list of row indices (int)
    """
    month_series = pd.to_numeric(df_raw[month_col], errors="coerce")
    data_mask = month_series.notna()
    df_data = df_raw.loc[data_mask]

    selected_indices: List[int] = []

    # group by state; pick December or highest month
    for state_val, sub in df_data.groupby(state_col):
        m = month_series.loc[sub.index]
        if (m == 12).any():
            idx = m[m == 12].index[0]
        else:
            idx = m.idxmax()
        selected_indices.append(idx)

    return selected_indices


# ---------------------------------------------------------------------------
# NON-NET-METERING PARSER
# ---------------------------------------------------------------------------

def parse_non_net_metering(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parse NON-NET-METERING "Monthly_Totals-States" sheet for a single year
    into long-format sector data.

    The sheet is already aggregated by state and month. This function:
      - Selects December (or latest month) per state.
      - Extracts sector-level PV and battery capacity/energy.
      - Uses number of generators as PV customers in the Residential sector.

    Output columns:
        year
        state
        sector
        pv_capacity_mw_non_net
        pv_customers_non_net
        battery_capacity_mw_non_net
        battery_energy_mwh_non_net
        battery_customers_non_net
    """
    # Identify month and UC columns directly (non-net has clean UC labels)
    uc_month_col = ("Utility Characteristics", "Month")
    uc_year_col = ("Utility Characteristics", "Year")
    uc_state_col = ("Utility Characteristics", "State")

    # Drop footer rows (notes) where Month isn't numeric
    month_series = pd.to_numeric(df_raw[uc_month_col], errors="coerce")
    df = df_raw.loc[month_series.notna()].copy()

    # Year is constant within this sheet
    year_val = pd.to_numeric(df[uc_year_col], errors="coerce").dropna().iloc[0]
    year = int(year_val)

    # Determine December row index per state
    dec_indices = _select_december_indices(df, uc_month_col, uc_state_col)
    dec_df = df.loc[dec_indices].copy()

    # Collect top-level block names to detect PV, storage, battery blocks dynamically
    blocks = {c[0] for c in dec_df.columns}

    # Photovoltaic block name (e.g. "Photovoltaic" or "Photovoltaic  (MW)")
    pv_block = None
    for b in blocks:
        if "photovoltaic" in b.lower():
            pv_block = b
            break

    # Storage (older years)
    storage_block = None
    for b in blocks:
        if b.lower() == "storage":
            storage_block = b
            break

    # Battery capacity and energy blocks (newer years)
    battery_block = None
    battery_energy_block = None
    for b in blocks:
        low = b.lower()
        if "battery" in low and "mwh" not in low:
            battery_block = b
        if "battery" in low and "mwh" in low:
            battery_energy_block = b

    def extract_block_sector(
        block_name: str | None,
        sectors: List[str],
        default_value: float,
    ) -> Dict[str, pd.Series]:
        """
        Extract sector-specific series for a given block from dec_df.

        For non-net, the second-level label is typically the sector name.
        """
        out = {s: pd.Series(default_value, index=dec_df.index) for s in sectors}
        if block_name is None:
            # Just return default (0 or NaN) if block absent
            return out

        for col in dec_df.columns:
            if col[0] != block_name:
                continue
            sec = str(col[1]).strip()
            if sec not in sectors:
                continue
            vals = pd.to_numeric(dec_df[col], errors="coerce")
            # If default is NaN, ensure we start from 0 when summing
            if np.isnan(default_value):
                out[sec] = vals
            else:
                out[sec] = vals.fillna(0)
        return out

    # PV capacity (MW) by sector
    pv_capacity = {s: pd.Series(0.0, index=dec_df.index) for s in SECTORS}
    if pv_block is not None:
        for col in dec_df.columns:
            if col[0] == pv_block:
                sec = str(col[1]).strip()
                if sec in SECTORS:
                    vals = pd.to_numeric(dec_df[col], errors="coerce").fillna(0)
                    pv_capacity[sec] = pv_capacity[sec] + vals

    # PV customers (non-net has no explicit customers; use number of generators
    # assigned to the Residential sector only)
    pv_customers = {s: pd.Series(0.0, index=dec_df.index) for s in SECTORS}
    numgen_col = ("Number and Capacity (MW)", "Number of Generators")
    if numgen_col in dec_df.columns:
        gens = pd.to_numeric(dec_df[numgen_col], errors="coerce").fillna(0)
        pv_customers["Residential"] = gens

    # Battery capacity (MW)
    # = Storage (if present) + Battery (if present)
    storage_capacity = extract_block_sector(storage_block, SECTORS, default_value=0.0)
    battery_capacity = extract_block_sector(battery_block, SECTORS, default_value=0.0)
    combined_batt_capacity = {
        s: storage_capacity[s] + battery_capacity[s] for s in SECTORS
    }

    # Battery energy (MWh) — use NaN if not available
    if battery_energy_block is not None:
        battery_energy = extract_block_sector(
            battery_energy_block, SECTORS, default_value=np.nan
        )
    else:
        battery_energy = {s: pd.Series(np.nan, index=dec_df.index) for s in SECTORS}

    # Non-net has no battery customers field; set to 0
    battery_customers = {s: pd.Series(0.0, index=dec_df.index) for s in SECTORS}

    # Build final long-format output
    out_rows: List[Dict[str, object]] = []
    for idx, row in dec_df.iterrows():
        state = str(row[uc_state_col])
        for s in SECTORS:
            out_rows.append(
                {
                    "year": year,
                    "state": state,
                    "sector": s,
                    "pv_capacity_mw_non_net": float(pv_capacity[s].loc[idx]),
                    #"pv_customers_non_net": float(pv_customers[s].loc[idx]),
                    "pv_customers_non_net": [0],
                    "battery_capacity_mw_non_net": float(
                        combined_batt_capacity[s].loc[idx]
                    ),
                    "battery_energy_mwh_non_net": battery_energy[s].loc[idx],
                    "battery_customers_non_net": float(
                        battery_customers[s].loc[idx]
                    ),
                }
            )

    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# NET-METERING PARSER
# ---------------------------------------------------------------------------

def parse_net_metering(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parse NET-METERING "Monthly_Totals-States" sheet for a single year
    into long-format sector data.

    Net-metering files use a slightly different convention:
      - The first row (index 0) contains "real" labels for UC fields and
        sector labels for technology blocks.
      - Subsequent rows contain numeric data.

    This parser:
      - Uses row 0 to detect Year, Month, State columns, sector labels.
      - Selects December (or latest month) per state.
      - Extracts PV capacity (MW) and customers.
      - Extracts battery/storage capacity (MW), energy (MWh where available),
        and customers.

    Output columns:
        year
        state
        sector
        pv_capacity_mw_nem
        pv_customers_nem
        battery_capacity_mw_nem
        battery_energy_mwh_nem
        battery_customers_nem
    """
    header_row = df_raw.iloc[0]

    # ---- Find UC columns by row-0 label (Year/Month/State) ----
    def find_uc_col(label: str) -> Tuple[str, str]:
        for col in df_raw.columns:
            if col[0] != "Utility Characteristics":
                continue
            val = str(header_row[col]).strip().lower()
            if val == label.lower():
                return col
        raise KeyError(f"Could not find UC column for label={label!r}")

    month_col = find_uc_col("month")
    year_col = find_uc_col("year")
    state_col = find_uc_col("state")

    # Month series; filter out header row and footer notes
    month_series = pd.to_numeric(df_raw[month_col], errors="coerce")
    data_mask = month_series.notna()
    df_data = df_raw.loc[data_mask].copy()

    # Year is constant; extract once
    year_val = pd.to_numeric(df_raw[year_col], errors="coerce").dropna().iloc[0]
    year = int(year_val)

    # Determine December row index per state
    dec_indices = _select_december_indices(df_raw, month_col, state_col)
    dec_df = df_raw.loc[dec_indices].copy()

    # Sector labels are in row 0
    blocks = {c[0] for c in df_raw.columns}

    def extract_sector_block(
        block_name: str,
        subfield_prefix: str,
        default_value: float,
    ) -> Dict[str, pd.Series]:
        """
        Extract sector-specific series for a block and subfield.

        - block_name: e.g. "Photovoltaic", "Battery"
        - subfield_prefix: e.g. "Capacity MW", "Customers", "Storage Capacity (MW)"

        Returns a dict of sector → Series aligned to dec_df.index.
        """
        out = {s: pd.Series(default_value, index=dec_df.index) for s in SECTORS}

        if block_name not in blocks:
            return out

        for col in df_raw.columns:
            if col[0] != block_name:
                continue
            if not str(col[1]).startswith(subfield_prefix):
                continue

            sector_label = str(header_row[col]).strip()
            if sector_label not in SECTORS:
                continue

            vals = pd.to_numeric(df_raw[col], errors="coerce")
            vals = vals.loc[dec_df.index]
            if np.isnan(default_value):
                out[sector_label] = vals
            else:
                out[sector_label] = out[sector_label] + vals.fillna(0)
        return out

    # ---- PV capacity and customers ----
    pv_block = None
    for b in blocks:
        if "photovoltaic" in b.lower():
            pv_block = b
            break
    if pv_block is None:
        raise ValueError("No Photovoltaic block found in net-metering sheet.")

    pv_capacity = extract_sector_block(pv_block, "Capacity MW", default_value=0.0)
    pv_customers = extract_sector_block(pv_block, "Customers", default_value=0.0)

    # ---- Storage under Photovoltaic (older years) ----
    storage_capacity = extract_sector_block(
        pv_block, "Storage Capacity", default_value=0.0
    )
    storage_customers = extract_sector_block(
        pv_block, "Storage Customers", default_value=0.0
    )

    # ---- Battery block (newer years) ----
    battery_block = None
    for b in blocks:
        if "battery" in b.lower():
            battery_block = b
            break

    pvpaired_capacity = extract_sector_block(
        battery_block or "", "PV-Paired Battery Capacity", default_value=0.0
    )
    pvpaired_customers = extract_sector_block(
        battery_block or "", "PV-Paired Installations", default_value=0.0
    )
    pvpaired_energy = extract_sector_block(
        battery_block or "", "PV-Paired Energy Capacity", default_value=np.nan
    )

    nopv_capacity = extract_sector_block(
        battery_block or "", "Not PV-Paired Battery Capacity", default_value=0.0
    )
    nopv_customers = extract_sector_block(
        battery_block or "", "Not PV-Paired Installations", default_value=0.0
    )
    nopv_energy = extract_sector_block(
        battery_block or "", "Not PV-Paired Energy Capacity", default_value=np.nan
    )

    # Combine battery capacity/customers/energy
    battery_capacity: Dict[str, pd.Series] = {}
    battery_customers: Dict[str, pd.Series] = {}
    battery_energy: Dict[str, pd.Series] = {}

    for s in SECTORS:
        battery_capacity[s] = (
            storage_capacity[s]
            + pvpaired_capacity[s]
            + nopv_capacity[s]
        )
        battery_customers[s] = (
            storage_customers[s]
            + pvpaired_customers[s]
            + nopv_customers[s]
        )

        # For energy, if both series are NaN, result stays NaN (sum of NaNs)
        energy_vals = pvpaired_energy[s] + nopv_energy[s]
        battery_energy[s] = energy_vals

    # ---- Build long-format output ----
    out_rows: List[Dict[str, object]] = []
    for idx, row in dec_df.iterrows():
        state = str(row[state_col])
        for s in SECTORS:
            out_rows.append(
                {
                    "year": year,
                    "state": state,
                    "sector": s,
                    "pv_capacity_mw_nem": float(pv_capacity[s].loc[idx]),
                    "pv_customers_nem": float(pv_customers[s].loc[idx]),
                    "battery_capacity_mw_nem": float(battery_capacity[s].loc[idx]),
                    "battery_energy_mwh_nem": battery_energy[s].loc[idx],
                    "battery_customers_nem": float(battery_customers[s].loc[idx]),
                }
            )

    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# MAIN PUBLIC API
# ---------------------------------------------------------------------------

def download_and_aggregate_distributed_solar(years: Iterable[int]) -> pd.DataFrame:
    """
    Download and aggregate net-metered + non-net-metered distributed solar
    and battery statistics for the given years.

    Data are pulled directly from EIA 861M Excel files.

    Parameters
    ----------
    years : Iterable[int]
        Years in [2017, 2025].

    Returns
    -------
    pd.DataFrame
        Columns:
            year
            state
            sector                # 'Residential', 'Commercial', 'Industrial'
            pv_capacity_mw
            pv_customers
            battery_capacity_mw
            battery_energy_mwh
            battery_customers
    """
    all_rows: List[Dict[str, object]] = []

    for year in years:
        # Download raw files
        non_net_raw = _download_non_net_raw(year)
        net_raw = _download_net_raw(year)

        # Parse each side
        df_non = parse_non_net_metering(non_net_raw)
        df_nem = parse_net_metering(net_raw)

        # Merge on year, state, sector
        merged = pd.merge(
            df_non,
            df_nem,
            on=["year", "state", "sector"],
            how="outer",
        )

        # Fill numeric NaNs with 0 where appropriate for capacities/customers.
        # For battery_energy_mwh we leave NaN as "no data in either source".
        for col in [
            "pv_capacity_mw_non_net",
            "pv_customers_non_net",
            "battery_capacity_mw_non_net",
            "battery_customers_non_net",
            "pv_capacity_mw_nem",
            "pv_customers_nem",
            "battery_capacity_mw_nem",
            "battery_customers_nem",
        ]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

        # Sum up net + non-net
        merged["pv_capacity_mw"] = (
            merged.get("pv_capacity_mw_non_net", 0.0)
            + merged.get("pv_capacity_mw_nem", 0.0)
        )
        merged["pv_customers"] = (
            merged.get("pv_customers_non_net", 0.0)
            + merged.get("pv_customers_nem", 0.0)
        )
        merged["battery_capacity_mw"] = (
            merged.get("battery_capacity_mw_non_net", 0.0)
            + merged.get("battery_capacity_mw_nem", 0.0)
        )
        # For energy, sum but keep NaN if both were NaN
        energy_non = pd.to_numeric(
            merged.get("battery_energy_mwh_non_net", np.nan), errors="coerce"
        )
        energy_nem = pd.to_numeric(
            merged.get("battery_energy_mwh_nem", np.nan), errors="coerce"
        )
        merged["battery_energy_mwh"] = energy_non + energy_nem

        merged["battery_customers"] = (
            merged.get("battery_customers_non_net", 0.0)
            + merged.get("battery_customers_nem", 0.0)
        )

        # Append minimal schema rows
        keep_cols = [
            "year",
            "state",
            "sector",
            "pv_capacity_mw",
            "pv_customers",
            "battery_capacity_mw",
            "battery_energy_mwh",
            "battery_customers",
        ]
        for _, row in merged[keep_cols].iterrows():
            all_rows.append(row.to_dict())

    out = pd.DataFrame(all_rows)
    out = out.sort_values(["year", "state", "sector"]).reset_index(drop=True)
    return out

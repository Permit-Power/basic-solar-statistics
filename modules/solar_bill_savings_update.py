"""
solar_bill_savings_update.py

Computes weighted median Year-1 and lifetime bill savings for the 2026
cohort of solar adopters, using dGen baseline scenario outputs.

Unlike the original solar_bill_savings.py, this module:
  - Uses baseline.csv instead of policy.csv
  - Filters to year == 2026 only (first cohort of adopters)
  - Uses weighted median (not weighted average) to match analysis_functions.py
  - Uses cf_energy_value_pv_only for lifetime savings (gross energy value)

Directory structure expected:
    {base_directory}/{state_abbr}/{run_name}/baseline.csv
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


BASE_DIRECTORY = (
    "/Users/wael/Library/CloudStorage/GoogleDrive-wael@permitpower.org"
    "/Shared drives/PP (All)/Research/$1 watt solar/Results/Raw"
)
RUN_NAME = "run_all_states_net_savings_add_itc"
COHORT_YEAR = 2026


def _parse_array(arr_str: str) -> list:
    if not isinstance(arr_str, str):
        return []
    cleaned = arr_str.strip().lstrip("{").rstrip("}")
    if not cleaned:
        return []
    try:
        return [float(x) for x in cleaned.split(",")]
    except ValueError:
        return []


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & (weights > 0)
    if not mask.any():
        return float("nan")
    v, w = values[mask], weights[mask]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cumw = np.cumsum(w)
    return float(v[np.searchsorted(cumw, 0.5 * w.sum(), side="left")])


def compute_state_bill_savings_baseline(
    base_directory: str = BASE_DIRECTORY,
    run_name: str = RUN_NAME,
    cohort_year: int = COHORT_YEAR,
) -> pd.DataFrame:
    """
    Compute weighted median Year-1 and lifetime bill savings by state,
    using only the first cohort of adopters (cohort_year).

    Year-1 savings  = utility_bill_wo_sys_pv_only[1] - utility_bill_w_sys_pv_only[1]
    Lifetime savings = sum of cf_energy_value_pv_only (25 years, gross energy value)

    Returns
    -------
    pd.DataFrame
        Columns: state_abbr, year_1_savings, lifetime_savings
    """
    results = []

    for state_abbr in sorted(os.listdir(base_directory)):
        state_path = os.path.join(base_directory, state_abbr)
        if not os.path.isdir(state_path):
            continue

        baseline_path = os.path.join(state_path, run_name, "baseline.csv")
        if not os.path.isfile(baseline_path):
            continue

        df = pd.read_csv(baseline_path)
        df = df[df["year"] == cohort_year]

        if df.empty or df["new_adopters"].sum() == 0:
            continue

        wo = df["utility_bill_wo_sys_pv_only"].apply(_parse_array)
        w  = df["utility_bill_w_sys_pv_only"].apply(_parse_array)
        ev = df["cf_energy_value_pv_only"].apply(_parse_array)

        year1 = np.array([
            (wo_arr[1] - w_arr[1]) if len(wo_arr) >= 2 and len(w_arr) >= 2 else np.nan
            for wo_arr, w_arr in zip(wo, w)
        ])
        bill_wo_year1 = np.array([
            wo_arr[1] if len(wo_arr) >= 2 else np.nan
            for wo_arr in wo
        ])
        lifetime = np.array([
            sum(ev_arr[:25]) if len(ev_arr) >= 1 else np.nan
            for ev_arr in ev
        ])
        weights = df["new_adopters"].values.astype(float)

        results.append({
            "state_abbr": state_abbr.upper(),
            "year_1_savings": _weighted_median(year1, weights),
            "lifetime_savings": _weighted_median(lifetime, weights),
            "median_bill_year_1": _weighted_median(bill_wo_year1, weights),
        })

    return pd.DataFrame(results)


def export_state_bill_savings_to_csv(
    export_directory: str,
    output_filename: Optional[str] = None,
    base_directory: str = BASE_DIRECTORY,
    run_name: str = RUN_NAME,
    cohort_year: int = COHORT_YEAR,
) -> str:
    """
    Compute state-level bill savings and write to a CSV for use in CI.

    Returns
    -------
    str
        Absolute path to the written CSV.
    """
    export_dir_path = Path(export_directory).resolve()
    export_dir_path.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = f"{run_name}_baseline_{cohort_year}_state_bill_savings.csv"

    output_path = export_dir_path / output_filename

    df = compute_state_bill_savings_baseline(
        base_directory=base_directory,
        run_name=run_name,
        cohort_year=cohort_year,
    )
    df.to_csv(output_path, index=False)
    return str(output_path)


def load_from_export(
    export_directory: str,
    output_filename: Optional[str] = None,
    run_name: str = RUN_NAME,
    cohort_year: int = COHORT_YEAR,
) -> pd.DataFrame:
    """
    Load a pre-computed bill savings CSV (for use in CI / basic_statistics.ipynb).

    Returns
    -------
    pd.DataFrame
        Columns: state_abbr, year_1_savings, lifetime_savings
    """
    if output_filename is None:
        output_filename = f"{run_name}_baseline_{cohort_year}_state_bill_savings.csv"

    csv_path = Path(export_directory).resolve() / output_filename

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Bill savings CSV not found at: {csv_path}. "
            "Run export_state_bill_savings_to_csv() locally and commit the CSV before running the pipeline."
        )

    return pd.read_csv(csv_path)

"""
Module: weighted_solar_bill_savings
===================================

This module computes weighted average Year-1 bill savings and weighted
25-year lifetime savings from rooftop solar using dGen per-state outputs.

Directory structure expected:

    /Seagate Portabl/permit_power/dgen_runs/per_state_outputs/
        {state_abbr}/
            {run_name}/
                policy.csv

The caller must specify `run_name`. For each state directory, the module
will look for:

    {base_directory}/{state_abbr}/{run_name}/policy.csv

and skip states where this file does not exist.

The module parses the 25-year bill arrays (index 1 = Year-1),
sums elements 1–24 for lifetime savings, and weights all metrics
by `new_adopters`.

Returned DataFrame has one row per state.

"""

import os
import pandas as pd
from typing import List, Dict


def _parse_bill_array(arr_str: str) -> List[float]:
    """
    Parse a 25-element bill array of form:
        "{0,354.6,374.65,...}"

    Returns a list of floats.

    Parameters
    ----------
    arr_str : str
        String representation of a PySAM bill array.

    Returns
    -------
    list of float
    """
    if not isinstance(arr_str, str):
        return []

    arr_str = arr_str.strip().lstrip("{").rstrip("}")
    if not arr_str:
        return []

    try:
        return [float(x) for x in arr_str.split(",")]
    except ValueError:
        return []


def _compute_row_metrics(row: pd.Series) -> Dict[str, float]:
    """
    Compute Year-1 bills and lifetime savings for one policy row.

    Parameters
    ----------
    row : pandas.Series
        Row containing:
        - new_adopters
        - utility_bill_wo_sys_pv_only
        - utility_bill_w_sys_pv_only

    Returns
    -------
    dict with:
        - bill_wo_y1
        - bill_w_y1
        - life_savings
        - weight
    """
    wo_arr = _parse_bill_array(row["utility_bill_wo_sys_pv_only"])
    w_arr = _parse_bill_array(row["utility_bill_w_sys_pv_only"])

    if len(wo_arr) < 2 or len(w_arr) < 2:
        return {
            "bill_wo_y1": 0.0,
            "bill_w_y1": 0.0,
            "life_savings": 0.0,
            "weight": 0.0,
        }

    # Year-1 = index 1 (exclude index 0 which is always zero)
    bill_wo_y1 = wo_arr[1]
    bill_w_y1 = w_arr[1]

    # Lifetime savings = sum of years 1–24 (indices 1..24)
    life_wo = sum(wo_arr[1:])
    life_w = sum(w_arr[1:])
    life_savings = life_wo - life_w

    weight = float(row["new_adopters"])

    return {
        "bill_wo_y1": bill_wo_y1,
        "bill_w_y1": bill_w_y1,
        "life_savings": life_savings,
        "weight": weight,
    }


def _process_policy_csv(csv_path: str) -> pd.DataFrame:
    """
    Load and process one policy.csv file.

    Parameters
    ----------
    csv_path : str
        Path to a policy.csv file.

    Returns
    -------
    pandas.DataFrame with:
        - bill_wo_y1
        - bill_w_y1
        - life_savings
        - weight
    """
    df = pd.read_csv(csv_path)

    required = [
        "new_adopters",
        "utility_bill_wo_sys_pv_only",
        "utility_bill_w_sys_pv_only",
    ]

    for col in required:
        if col not in df.columns:
            return pd.DataFrame(columns=["bill_wo_y1", "bill_w_y1", "life_savings", "weight"])

    return df.apply(_compute_row_metrics, axis=1, result_type="expand")


def compute_state_bill_savings(base_directory: str, run_name: str) -> pd.DataFrame:
    """
    Compute weighted average Year-1 bill savings and 25-year lifetime
    savings by state using dGen per-state outputs.

    Parameters
    ----------
    base_directory : str
        Path to the per_state_outputs directory, e.g.:
        "/Seagate Portabl/permit_power/dgen_runs/per_state_outputs"

    run_name : str
        The name of the run directory under each state, e.g.:
        "run_2024_base" or "scenario_a".

        The module looks for:
            {base_directory}/{state_abbr}/{run_name}/policy.csv

    Returns
    -------
    pandas.DataFrame
        Columns:
            - state_abbr
            - weighted_avg_bill_without_pv_year1
            - weighted_avg_bill_with_pv_year1
            - pct_savings_year1
            - lifetime_savings_weighted
            - total_new_adopters
    """
    results = []

    # Iterate over state directories
    for state_abbr in sorted(os.listdir(base_directory)):
        state_path = os.path.join(base_directory, state_abbr)
        if not os.path.isdir(state_path):
            continue

        policy_path = os.path.join(state_path, run_name, "policy.csv")
        if not os.path.isfile(policy_path):
            continue  # skip missing states

        df_metrics = _process_policy_csv(policy_path)
        if df_metrics.empty:
            continue

        w = df_metrics["weight"].sum()
        if w == 0:
            continue

        # Weighted Year-1 bills
        bill_wo_y1 = (df_metrics["bill_wo_y1"] * df_metrics["weight"]).sum() / w
        bill_w_y1  = (df_metrics["bill_w_y1"]  * df_metrics["weight"]).sum() / w

        pct_savings_y1 = (bill_wo_y1 - bill_w_y1) / bill_wo_y1 if bill_wo_y1 > 0 else 0.0

        # Weighted lifetime savings
        life_savings_w = (df_metrics["life_savings"] * df_metrics["weight"]).sum() / w

        results.append({
            "state_abbr": state_abbr.upper(),
            "weighted_avg_bill_without_pv_year1": bill_wo_y1,
            "weighted_avg_bill_with_pv_year1": bill_w_y1,
            "weighted_avg_savings_year1": (bill_wo_y1 - bill_w_y1),
            "pct_savings_year1": pct_savings_y1,
            "lifetime_savings_weighted": life_savings_w
        })

    return pd.DataFrame(results)

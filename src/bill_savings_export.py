"""
Module: bill_savings_export
===========================

This module provides a helper to pre-compute and export state-level solar
bill savings from dGen per-state outputs into a CSV inside the repository.

Typical workflow:

1. Run this locally on a machine that has the external drive mounted, e.g.:

       base_directory = "/Volumes/Seagate Portabl/permit_power/dgen_runs/per_state_outputs"
       run_name = "run_all_states_net_savings_adjust_loan_params"
       export_directory = "../data/bill_savings_csvs"

       from bill_savings_export import export_state_bill_savings_to_csv
       csv_path = export_state_bill_savings_to_csv(base_directory, run_name, export_directory)

2. Commit the resulting CSV into the repo (if appropriate).

3. In CI (GitHub Actions), instead of reading from the external drive,
   the pipeline uses the exported CSV via `compute_state_bill_savings_from_export`
   in solar_bill_savings.py.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from solar_bill_savings import compute_state_bill_savings


def export_state_bill_savings_to_csv(
    base_directory: str,
    run_name: str,
    export_directory: str,
    output_filename: Optional[str] = None,
) -> str:
    """
    Compute state-level bill savings from dGen per-state outputs and write
    the aggregated results to a CSV inside the repository.

    This function is intended to be run locally on a machine with access
    to the external drive. CI can then read the exported CSV without
    needing access to the external filesystem.

    Parameters
    ----------
    base_directory : str
        Path to the per_state_outputs directory on the external drive, e.g.:
        "/Volumes/Seagate Portabl/permit_power/dgen_runs/per_state_outputs"

    run_name : str
        Name of the dGen run directory under each state, e.g.:
        "run_all_states_net_savings_adjust_loan_params".

    export_directory : str
        Path (relative or absolute) to the directory where the aggregated
        CSV should be written. This should typically point inside the repo,
        e.g. "../data/bill_savings_csvs".

    output_filename : str, optional
        Name of the output CSV file. If None, defaults to:
        f"{run_name}_state_bill_savings.csv".

    Returns
    -------
    str
        Absolute path to the written CSV file.
    """
    export_dir_path = Path(export_directory).resolve()
    export_dir_path.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = f"{run_name}_state_bill_savings.csv"

    output_path = export_dir_path / output_filename

    df = compute_state_bill_savings(base_directory=base_directory, run_name=run_name)

    # Ensure deterministic column ordering for easier diff/inspection.
    desired_order = [
        "state_abbr",
        "weighted_avg_bill_without_pv_year1",
        "weighted_avg_bill_with_pv_year1",
        "weighted_avg_savings_year1",
        "pct_savings_year1",
        "lifetime_savings_weighted",
    ]
    cols = [c for c in desired_order if c in df.columns] + [
        c for c in df.columns if c not in desired_order
    ]
    df = df[cols]

    df.to_csv(output_path, index=False)
    return str(output_path)

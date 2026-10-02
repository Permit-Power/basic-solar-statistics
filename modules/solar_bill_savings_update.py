"""
solar_bill_savings_update.py

Computes weighted median Year-1 and lifetime (25-year) bill savings for the 2026
cohort of solar adopters, using dGen baseline scenario outputs, matched to each
state's residential export compensation policy.

dGen was run twice, differing only in how exported solar is credited:
  - NET_METERING_RUN: exports credited at full retail (all states but CA)
  - NET_BILLING_RUN:  exports credited at hourly wholesale prices (CA is on its
                      NEM 3.0 net billing in both runs)
Each state uses the run matching its policy in DSIRE_CSV, or the average of
the two where its export credit is between retail and avoided cost.

Unlike the original solar_bill_savings.py, this module:
  - Uses baseline.csv instead of policy.csv
  - Filters to year == 2026 only (first cohort of adopters)
  - Uses weighted median (not weighted average) to match analysis_functions.py
  - Reports lifetime savings as 25 years in today's (2026) dollars

cf_energy_value_pv_only only covers years 1-24 (slot 0 is year 0, always 0),
but cf_discounted_savings_pv_only covers years 1-25. It is the same energy
value discounted by ((1 + real_discount_rate) * (1 + inflation_rate))^t, so
multiplying back by (1 + real_discount_rate)^t leaves savings deflated by
inflation only, i.e. in today's dollars.

Directory structure expected:
    {base_directory}/{state_abbr}/{run_name}/baseline.csv

Both runs are weighted by the net metering run's 2026 adopters (matched by
agent_id; the two runs model the same households), so each state's numbers
describe a typical solar customer under its export policy. Weighting the net
billing run by its own adopters would instead pick a smaller, better-economics
group (savings could even rise, e.g. AZ), and DC, NV and UT have no 2026
adopters in that run at all.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


BASE_DIRECTORY = (
    "/Users/wael/Library/CloudStorage/GoogleDrive-wael@permitpower.org"
    "/Shared drives/PP (All)/Research/$1 watt solar/2025/Results/Updated tariffs"
)
NET_METERING_RUN = "synapse_attachrate_75"
NET_BILLING_RUN = "synapse_attachrate_75_net_billing"
COHORT_YEAR = 2026
OUTPUT_FILENAME = f"state_bill_savings_{COHORT_YEAR}_by_export_policy.csv"

# State residential export compensation (DSIRE, investor-owned utilities).
DSIRE_CSV = Path(__file__).resolve().parents[1] / "data" / "dsire_iou_net_metering_may2026.csv"

NET_METERING = "net metering"
NET_BILLING = "net billing"
AVERAGE = "average of net metering and net billing"

# A net metering state whose export credit is below this share of retail is
# treated as between retail and avoided cost (e.g. NV at 75%).
FULL_RETAIL_THRESHOLD = 0.95

# States decided by hand instead of from the DSIRE category: (scenario, note).
SCENARIO_OVERRIDES = {
    "NH": (AVERAGE, "Exports credited at supply + transmission + 25% of distribution, "
                    "between retail and avoided cost"),
    "DC": (NET_METERING, "Not in the DSIRE file; DC has full-retail residential net metering"),
}
SCENARIO_NOTES = {
    "TN": "No export compensation; the net billing run's wholesale export credit overstates savings",
    "TX": "No statewide rule; export credit is up to each retail electric provider",
}


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
    run_name: str,
    base_directory: str = BASE_DIRECTORY,
    cohort_year: int = COHORT_YEAR,
    weights_run: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute weighted median Year-1 and lifetime bill savings by state for one
    dGen run, using only the first cohort of adopters (cohort_year).

    weights_run: take the new_adopters weights from this run instead (matched
    by agent_id), so two runs describe the same households.

    Year-1 savings  = utility_bill_wo_sys_pv_only[1] - utility_bill_w_sys_pv_only[1]
    Lifetime savings = sum over years 1-25 of
                       cf_discounted_savings_pv_only[t] * (1 + real_discount_rate)^t
                       (gross energy value in today's dollars; not discounted)

    Savings are weighted by new_adopters, or by customers_in_bin where a state
    has no adopters in cohort_year (weight_basis says which). The bill is
    always weighted the same way as savings.

    Returns
    -------
    pd.DataFrame
        Columns: state_abbr, year_1_savings, lifetime_savings, median_bill_year_1, weight_basis
    """
    results = []

    for state_abbr in sorted(os.listdir(base_directory)):
        baseline_path = os.path.join(base_directory, state_abbr, run_name, "baseline.csv")
        if not os.path.isfile(baseline_path):
            continue

        df = pd.read_csv(baseline_path)
        df = df[df["year"] == cohort_year]
        if df.empty:
            continue
        weights_label = "new_adopters"
        if weights_run is not None:
            w_df = pd.read_csv(os.path.join(base_directory, state_abbr, weights_run, "baseline.csv"),
                               usecols=["year", "agent_id", "new_adopters"])
            w_df = w_df[w_df["year"] == cohort_year]
            if set(w_df["agent_id"]) != set(df["agent_id"]):
                raise ValueError(f"{state_abbr}: {run_name} and {weights_run} model different households")
            df = df.drop(columns="new_adopters").merge(w_df[["agent_id", "new_adopters"]], on="agent_id")
            weights_label = f"new_adopters ({weights_run})"

        wo = df["utility_bill_wo_sys_pv_only"].apply(_parse_array)
        w  = df["utility_bill_w_sys_pv_only"].apply(_parse_array)
        ds = df["cf_discounted_savings_pv_only"].apply(_parse_array)
        real_discount = 1 + df["real_discount_rate"].values

        year1 = np.array([
            (wo_arr[1] - w_arr[1]) if len(wo_arr) >= 2 and len(w_arr) >= 2 else np.nan
            for wo_arr, w_arr in zip(wo, w)
        ])
        bill_wo_year1 = np.array([
            wo_arr[1] if len(wo_arr) >= 2 else np.nan
            for wo_arr in wo
        ])
        lifetime = np.array([
            sum(v * r ** t for t, v in enumerate(ds_arr[1:26], start=1)) if len(ds_arr) >= 26 else np.nan
            for ds_arr, r in zip(ds, real_discount)
        ])
        if df["new_adopters"].sum() > 0:
            weights, weight_basis = df["new_adopters"].values.astype(float), weights_label
        else:
            weights, weight_basis = df["customers_in_bin"].values.astype(float), "customers_in_bin"

        results.append({
            "state_abbr": state_abbr.upper(),
            "year_1_savings": _weighted_median(year1, weights),
            "lifetime_savings": _weighted_median(lifetime, weights),
            "median_bill_year_1": _weighted_median(bill_wo_year1, weights),
            "weight_basis": weight_basis,
        })

    return pd.DataFrame(results)


def state_savings_scenarios(dsire_csv: Path = DSIRE_CSV) -> pd.DataFrame:
    """
    Which run each state's savings come from, from its DSIRE residential
    export compensation category:
      - net_metering at full retail       -> net metering run
      - net_metering below full retail    -> average of the two runs
      - mixed (e.g. TX)                   -> average of the two runs
      - net_billing_or_other, none        -> net billing run
    SCENARIO_OVERRIDES take precedence.
    """
    dsire = pd.read_csv(dsire_csv)
    rows = []
    for r in dsire.itertuples():
        pct = r.export_credit_pct_of_retail
        if r.residential_resolved == "net_metering":
            scenario = NET_METERING if pd.isna(pct) or pct >= FULL_RETAIL_THRESHOLD else AVERAGE
        elif r.residential_resolved == "mixed":
            scenario = AVERAGE
        else:
            scenario = NET_BILLING
        note = SCENARIO_NOTES.get(r.state_abbr, "")
        if pd.notna(pct) and pct < FULL_RETAIL_THRESHOLD:
            note = f"Exports credited at {pct:.0%} of retail"
        rows.append({"state_abbr": r.state_abbr, "export_policy": r.residential_resolved,
                     "savings_scenario": scenario, "scenario_note": note})
    out = pd.DataFrame(rows).set_index("state_abbr")
    for state, (scenario, note) in SCENARIO_OVERRIDES.items():
        out.loc[state, ["savings_scenario", "scenario_note"]] = [scenario, note]
        if pd.isna(out.loc[state, "export_policy"]):
            out.loc[state, "export_policy"] = "not in DSIRE file"
    return out.reset_index()


def compute_state_bill_savings(
    base_directory: str = BASE_DIRECTORY,
    cohort_year: int = COHORT_YEAR,
    dsire_csv: Path = DSIRE_CSV,
) -> pd.DataFrame:
    """
    State bill savings under each state's export compensation policy: the net
    metering run, the net billing run, or the average of the two (see
    state_savings_scenarios). Both runs' values are kept for reference.

    Returns
    -------
    pd.DataFrame
        Columns: state_abbr, year_1_savings, lifetime_savings, median_bill_year_1,
        export_policy, savings_scenario, scenario_note, year_1_savings_net_metering,
        year_1_savings_net_billing, lifetime_savings_net_metering,
        lifetime_savings_net_billing, net_billing_weight_basis
    """
    nm = compute_state_bill_savings_baseline(NET_METERING_RUN, base_directory, cohort_year).set_index("state_abbr")
    nb = compute_state_bill_savings_baseline(NET_BILLING_RUN, base_directory, cohort_year,
                                             weights_run=NET_METERING_RUN).set_index("state_abbr")
    scenarios = state_savings_scenarios(dsire_csv).set_index("state_abbr")

    out = pd.DataFrame({
        "year_1_savings_net_metering": nm["year_1_savings"],
        "year_1_savings_net_billing": nb["year_1_savings"],
        "lifetime_savings_net_metering": nm["lifetime_savings"],
        "lifetime_savings_net_billing": nb["lifetime_savings"],
        # The bill without solar is the same in both runs; use the adopter-weighted one.
        "median_bill_year_1": nm["median_bill_year_1"],
        "net_billing_weight_basis": nb["weight_basis"],
    }).join(scenarios, how="left")

    missing = out.index[out["savings_scenario"].isna()].tolist()
    if missing:
        raise ValueError(f"No export compensation scenario for: {missing}; add them to {dsire_csv.name} or SCENARIO_OVERRIDES")

    for measure in ("year_1_savings", "lifetime_savings"):
        nm_vals, nb_vals = out[f"{measure}_net_metering"], out[f"{measure}_net_billing"]
        out[measure] = np.select(
            [out["savings_scenario"] == NET_METERING, out["savings_scenario"] == NET_BILLING],
            [nm_vals, nb_vals],
            default=(nm_vals + nb_vals) / 2,
        )

    cols = ["year_1_savings", "lifetime_savings", "median_bill_year_1", "export_policy", "savings_scenario",
            "scenario_note", "year_1_savings_net_metering", "year_1_savings_net_billing",
            "lifetime_savings_net_metering", "lifetime_savings_net_billing", "net_billing_weight_basis"]
    return out[cols].reset_index()


def export_state_bill_savings_to_csv(
    export_directory: str,
    output_filename: str = OUTPUT_FILENAME,
    base_directory: str = BASE_DIRECTORY,
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
    output_path = export_dir_path / output_filename
    compute_state_bill_savings(base_directory=base_directory, cohort_year=cohort_year).to_csv(output_path, index=False)
    return str(output_path)


def load_from_export(
    export_directory: str,
    output_filename: str = OUTPUT_FILENAME,
) -> pd.DataFrame:
    """
    Load a pre-computed bill savings CSV (for use in CI / basic_statistics.ipynb).

    Returns
    -------
    pd.DataFrame
        Columns as in compute_state_bill_savings.
    """
    csv_path = Path(export_directory).resolve() / output_filename

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Bill savings CSV not found at: {csv_path}. "
            "Run export_state_bill_savings_to_csv() locally and commit the CSV before running the pipeline."
        )

    return pd.read_csv(csv_path)

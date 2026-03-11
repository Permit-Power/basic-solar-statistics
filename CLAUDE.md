# CLAUDE.md — basic-solar-statistics

## Project purpose

This repository pulls, cleans, and organizes solar-related statistics for U.S. states and jurisdictions, then uploads the results to a Google Drive folder ("The Big Numbers Database"). A GitHub Action runs the full pipeline automatically on the 1st of each month.

## Tech stack

- **Python 3.11** managed with **Poetry**
- **Jupyter notebooks** executed headlessly via **papermill**
- **Google Drive API** (via Workload Identity Federation in CI; service-account credentials locally)
- Data sources: EIA 861M, LBNL Tracking the Sun, Solar TRACE, ResStock, Census

## Repository layout

```
notebooks/
  basic_statistics.ipynb      # Main pipeline notebook — runs all modules and uploads to Drive
  export_bill_savings.ipynb   # Exports bill savings CSVs from dGen outputs (run locally)

src/
  pull_solar_count_and_capacity.py   # Downloads EIA 861M Excel files and aggregates PV/battery stats
  solar_technical_potential.py       # Computes solar-eligible household counts from ResStock metadata
  median_solar_costs.py              # Processes LBNL Tracking the Sun cost data (real $/W)
  average_electricity_prices.py      # Builds state- and utility-level electricity price tables from EIA 861
  median_permit_fees.py              # Aggregates Solar TRACE permitting fee data
  median_interconnection_timelines.py# Aggregates Solar TRACE interconnection timeline data
  solar_bill_savings.py              # Loads pre-computed dGen bill savings exports
  bill_savings_export.py             # Helper to export dGen per_state_outputs to CSV
  drive_uploader.py                  # Google Drive helpers (authenticate, ensure folder path, upload)

data/
  *.csv                              # Static and pre-processed input datasets
  bill_savings_csvs/                 # Pre-computed dGen bill savings (must be present before pipeline runs)

.github/workflows/
  run-monthly.yml                    # Scheduled GitHub Action (1st of each month, 12:00 UTC)
```

## Running the pipeline

### Prerequisites

```bash
poetry install --no-interaction
```

For the monthly notebook you also need:

```bash
poetry run pip install papermill
```

### Local notebook execution

```bash
cd notebooks
poetry run papermill basic_statistics.ipynb basic_statistics-output.ipynb
```

The output notebook (`*-output.ipynb`) is git-ignored.

### Google Drive authentication

In CI, authentication is handled automatically via [Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation). Locally, place a service-account JSON key somewhere accessible and set `GOOGLE_APPLICATION_CREDENTIALS` to its path, or rely on `gcloud auth application-default login`.

## Key data dependencies

| Dataset | Source | How loaded |
|---|---|---|
| ResStock metadata | Static CSV in `data/` | Read from disk |
| AHJ permitting/inspection data | Static CSVs in `data/` | Read from disk |
| Solar TRACE fees & IX timelines | Static CSVs in `data/` | Read from disk |
| Bill savings | Pre-computed CSVs in `data/bill_savings_csvs/` | Must be present before pipeline runs |
| EIA 861M solar/battery stats | Downloaded live from EIA website | `pull_solar_count_and_capacity.py` |
| LBNL Tracking the Sun costs | Downloaded from Google Drive link | `median_solar_costs.py` |
| EIA 861 electricity prices | Downloaded live from EIA website | `average_electricity_prices.py` |

## Google Drive upload structure

Results are uploaded to folder ID `1DBlVUvspIPTTyZPtVovYtEgUSmQNXmG7` with the following layout:

```
1. Solar/
2. Permitting/
3. Inspection/
4. Interconnection/
5. Rates/
6. Jobs/
```

## Important notes

- The `data/bill_savings_csvs/` directory must contain the pre-computed CSV (`run_all_states_net_savings_adjust_loan_params_state_bill_savings.csv`) before running `basic_statistics.ipynb`. This file is generated locally from dGen outputs using `export_bill_savings.ipynb` and committed to the repo.
- `data/resstock_metadata_technical_potential.csv` is ~34 MB and committed directly to the repo.
- The GitHub Action only has `contents: read` permission; it does not write back to the repository.

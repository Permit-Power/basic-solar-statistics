# basic-solar-statistics

Pulls, processes, and organizes solar statistics for U.S. states and jurisdictions. A GitHub Action runs the full pipeline on the 1st of each month, uploading results to a shared Google Drive folder ("The Big Numbers Database").

## What's in this repo

1. **Monthly Big Numbers pipeline** (`notebooks/`, `modules/`, `data/`): pulls and processes the statistics, uploads them to Drive, run by the GitHub Action.
2. **One-off analyses** (`ad_hocs/`): exploratory notebooks, not part of any pipeline.
3. **Local bill-savings export** (`notebooks/export_bill_savings.ipynb`): regenerates `data/bill_savings_csvs/` from dGen outputs; needs the dGen shared drive.

---

## Pipeline architecture

The pipeline runs in two stages, both executed as Jupyter notebooks via `papermill`.

```
Stage 1: pipeline.ipynb
  - Pulls live data from EIA, LBNL, and other sources
  - Processes static input files from data/
  - Uploads all outputs to Google Drive (skipped with -p UPLOAD False)
  - Writes processed DataFrames to output_csvs/

Stage 2: compile_state_stats.ipynb
  - Reads from output_csvs/ (produced by Stage 1)
  - Joins everything into a single state-level summary table
  - Exports output_csvs/state_stats.csv
```

Stage 2 never pulls from the internet — it only reads from `output_csvs/`. This means you can re-run Stage 2 quickly to adjust joins or column names without re-fetching data.

---

## Stage 1 — pipeline.ipynb

### What it does, in order

| Cell | What it does | Source | Output variable |
|------|-------------|--------|----------------|
| Load data | Reads static input files | `data/` | — |
| Ohm static uploads | Loads pre-processed Ohm CSVs and uploads directly to Drive | Ohm Analytics | `cancellation_*`, `permitting_timelines_*`, `inspection_timelines_ahj` |
| Solar capacity | Downloads annual EIA 861 files for 2017–2025, aggregates PV/battery stats | EIA 861 (live) | `capacity` |
| Solar eligibility | Filters ResStock households for rooftop suitability | ResStock metadata | `eligibility` |
| Solar costs | Reads LBNL Tracking the Sun data tool export (state medians in 2025 dollars, inflation-adjusted by LBNL) | `data/tts_state_download_data.xlsx` | `costs` |
| Electricity prices | Pulls state prices from EIA API (2018 onward), utility prices from EIA 861 files (2018–2024) | EIA API + EIA 861 (live) | `state_prices`, `util_prices` |
| Permitting fees | Aggregates Solar TRACE fee data by AHJ and state | `data/solartrace_fees.csv` | `ahj_df_fees`, `state_df_fees` |
| IX timelines (legacy) | Aggregates old Solar TRACE IX CSV by AHJ and state | `data/solartrace_ix.csv` | `ahj_df_ix`, `state_df_ix` |
| Solar TRACE timelines | Reshapes Solar TRACE xlsx into tidy utility and state tables | `data/SolarTRACE Dataset v9-9-2025.xlsx` | `utility_timelines`, `state_timelines` |
| Bill savings | Loads pre-computed dGen savings | `data/bill_savings_csvs/` | `savings` |
| Export CSVs | Writes all processed DataFrames to `output_csvs/` | — | — |

### output_csvs/ written by Stage 1

| File | Source | Contents |
|------|--------|----------|
| `solar_storage_capacity_installations_by_state_sector.csv` | EIA 861 (annual) | Cumulative solar/storage capacity (MW, AC) and install counts by state, sector, year; residential non-net-metered installs are estimated from capacity |
| `solar_eligible_households_by_state.csv` | ResStock | Eligible and suitable household counts by state |
| `potential_solar_generation_by_state.csv` | NREL SLOPE | Technical generation potential in GWh by state |
| `residential_solar_costs_by_state_over_time.csv` | LBNL TTS data tool | Median residential $/kW (2025 dollars: the 2025 value is nominal and LBNL adjusts earlier years to 2025) and system size (kW) by state and year, with sample sizes and 20th/80th percentiles |
| `annual_and_lifetime_solar_savings_by_state.csv` | dGen | Median year-1 and lifetime (25-year, 2026 $) bill savings for a typical solar customer, by state, under the state's export compensation policy (PV only; see "Solar bill savings" below) |
| `electricity_rates_by_state.csv` | EIA API | Avg price ($/kWh) and annual bill by state, sector, year (2018 on, 2026 $), plus a `US` row (national revenue / national sales) for comparison |
| `electricity_rates_by_utility.csv` | EIA 861 | Avg price ($/kWh) and annual bill by utility, sector, year (2018 on, 2026 $) |
| `solartrace_timelines_by_state.csv` | Solar TRACE xlsx | All timeline metrics by state, year, size class, tech class |
| `solartrace_timelines_by_utility.csv` | Solar TRACE xlsx | All timeline metrics by utility, state, year, size class, tech class (weighted median) |
| `cancellation_rates_by_state.csv` | Ohm Analytics | Solar permit cancellation rates by state |
| `cancellation_rates_by_ahj.csv` | Ohm Analytics | Solar permit cancellation rates by AHJ |
| `permitting_timeline_distribution_by_state.csv` | Ohm Analytics | Percentile distribution of permitting timelines by state |
| `permitting_timeline_distribution_by_ahj.csv` | Ohm Analytics | Percentile distribution of permitting timelines by AHJ |
| `median_permitting_timeline_by_tech_by_ahj.csv` | Ohm Analytics | Median permitting timeline by AHJ and technology |
| `inspection_timelines_by_ahj.csv` | Ohm Analytics | Median inspection timeline by AHJ |
| `permitting_fees_by_state.csv` | Solar TRACE | Median permitting fee by state |
| `permitting_fees_by_ahj.csv` | Solar TRACE | Median permitting fee by AHJ |
| `interconnection_timelines_by_state.csv` | Solar TRACE (legacy CSV) | IX timelines by state — legacy, not used in state_stats |
| `interconnection_timelines_by_ahj.csv` | Solar TRACE (legacy CSV) | IX timelines by AHJ — legacy, not used in state_stats |
| `solar_jobs_by_state.csv` | IREC | Solar jobs by state and category |
| `state_stats.csv` | All of the above | Final joined state-level summary (written by Stage 2) |

---

## Stage 2 — compile_state_stats.ipynb

Reads from `output_csvs/` and a handful of static files in `data/`, joins everything, and writes `state_stats.csv`.

### Columns in state_stats.csv and where they come from

| Column | Source file | Filter applied |
|--------|------------|---------------|
| State, State (Abbr.) | `data/state_name_abbr.csv` | — |
| Median solar savings in first year | `output_csvs/annual_and_lifetime_solar_savings_by_state.csv` | — |
| Median % bill reduction in first year | same | Median of each household's (bill without solar − bill with solar) ÷ bill without solar, year 1 |
| Median solar savings over lifetime | same | 25 years, 2026 dollars; export policy per state |
| Average electricity retail cost ($/kWh) | `output_csvs/electricity_rates_by_state.csv` | Latest year, residential sector |
| Average annual electricity bill | same | Latest year, residential sector |
| Number of solar installations | `output_csvs/solar_storage_capacity_installations_by_state_sector.csv` | Latest year, residential sector |
| Total installed solar capacity (MW) | same | Latest year, residential sector |
| Total households | `output_csvs/solar_eligible_households_by_state.csv` | — |
| Solar eligible and suitable households | same | — |
| Percent of total households suitable | same | — |
| Solar penetration | computed: installs / eligible households | — |
| Annual technical potential (GWh) | `output_csvs/potential_solar_generation_by_state.csv` | — |
| Median solar cost ($/kW) | `output_csvs/residential_solar_costs_by_state_over_time.csv` | Most recent of 2025/2024 with a median based on ≥200 systems; blank otherwise. The year used per state is in `output_csvs/solar_costs_latest_by_state.csv` |
| Median solar system size (kW) | same | Same rule, chosen separately from cost |
| Solar permit cancellation rate | `data/state_cancellation_rates.csv` | 2022–2024 combined |
| Median permitting timeline — solar | `output_csvs/solartrace_timelines_by_state.csv` | 2023, 0-10kW, pv_only, ≥250 installs |
| Median permitting timeline — solar+storage | same | 2023, 0-10kW, pv_storage, ≥250 installs |
| Median interconnection timeline — PV only | same | 2023, 0-10kW, pv_only, ≥250 installs; pre-install IX + final IX to PTO summed |
| Median interconnection timeline — PV+storage | same | 2023, 0-10kW, pv_storage, ≥250 installs; pre-install IX + final IX to PTO summed |
| Median inspection timeline — PV only | same | 2023, 0-10kW, pv_only, ≥250 installs |
| Median inspection timeline — PV+storage | same | 2023, 0-10kW, pv_storage, ≥250 installs |

---

## Data sources: Ohm Analytics vs Solar TRACE

Both Ohm Analytics and Solar TRACE provide permitting, inspection, and interconnection timeline data, and both are partially in use.

**Solar TRACE** (`SolarTRACE Dataset v9-9-2025.xlsx`) is the **primary source** for all timeline metrics used in `state_stats.csv`. It covers permit time, pre-install IX, inspection time, final IX to PTO, install time, and project time — broken out by state, year (2017–2024), size class (0-10kW / 10-20kW), and tech class (PV only / PV+storage).

**Ohm Analytics** (static CSVs in `data/`) is still used for:
- **Cancellation rates** — no Solar TRACE equivalent; used in `state_stats.csv`
- **Permitting timeline distributions** (percentile breakdowns) — uploaded to Drive only, not in `state_stats.csv`
- **AHJ-level permitting and inspection medians** — uploaded to Drive only, not in `state_stats.csv`

The following Ohm files are **no longer used** by the pipeline (replaced by Solar TRACE); `ad_hocs/solartrace_exploration.ipynb` still reads them for comparison:
- `data/state_inspection_timelines_pv.csv`
- `data/state_median_permitting_timelines_by_tech.csv`

---

## Solar bill savings

`data/bill_savings_csvs/state_bill_savings_2026_by_export_policy.csv` is generated locally by `notebooks/export_bill_savings.ipynb` (via `modules/solar_bill_savings_update.py`) from dGen runs on the shared drive (`$1 watt solar/2025/Results/Updated tariffs/{STATE}/{run}/baseline.csv`), then committed.

- **Two dGen runs**, identical except export compensation: `synapse_attachrate_75` credits exports at full retail (net metering) and `synapse_attachrate_75_net_billing` at hourly wholesale prices (net billing). California is on its NEM 3.0 net billing in both.
- **Per-state policy** from `data/dsire_iou_net_metering_may2026.csv` (DSIRE, investor-owned utilities, residential): net metering states use the net metering run, net billing states the net billing run, and states credited between retail and avoided cost (NH, NV at 75% of retail, TX) the average of the two. DC (not in the file) uses net metering; TN (no export compensation) uses net billing, which overstates. Overrides and notes are in `SCENARIO_OVERRIDES` / `SCENARIO_NOTES`.
- **Typical solar customer**: weighted medians over the 2026 adopters of the net metering run, in both runs (same households, matched by `agent_id`). Weighting the net billing run by its own adopters would pick a smaller, better-economics group.
- **First-year savings** = bill without solar − bill with solar (PV only), year 1. **First-year % bill reduction** = that savings ÷ the bill without solar, per household (the median of households' percentages, so not median savings ÷ median bill). **Lifetime** = 25 years of savings in 2026 dollars (dGen's discounted savings with the 5% real discount removed).
- The CSV also keeps both runs' values, the policy, the scenario used, and a note per state.

---

## data/ folder — static inputs

| File | Used by | How produced | Notes |
|------|---------|--------------|-------|
| `SolarTRACE Dataset v9-9-2025.xlsx` | `process_solartrace_timelines.py` | downloaded source snapshot | Primary timeline source |
| `solartrace_fees.csv` | `median_permit_fees.py` | processed outside this repo (no script), from Solar TRACE | Permitting fee data |
| `solartrace_ix.csv` | `median_interconnection_timelines.py` | processed outside this repo (no script), from Solar TRACE | Legacy IX data; still uploaded to Drive but not used in state_stats |
| `tts_state_download_data.xlsx` | `median_solar_costs.py` | downloaded source snapshot | LBNL Tracking the Sun data tool export (trackingthesun.lbl.gov, 2026 edition, installs through 2025); replace with a new export to update |
| `resstock_metadata_technical_potential.csv` | `solar_eligible_households.py` | processed outside this repo (no script) | ~34 MB; ResStock household metadata |
| `bill_savings_csvs/` | `solar_bill_savings_update.py` | generated by `notebooks/export_bill_savings.ipynb` | Pre-computed dGen outputs; must exist before pipeline runs |
| `dsire_iou_net_metering_may2026.csv` | `solar_bill_savings_update.py` | processed outside this repo (no script) | State residential export compensation (DSIRE, IOUs), picks the dGen run per state |
| `state_cancellation_rates.csv` | `compile_state_stats.ipynb` | processed outside this repo (no script) | Ohm; used in state_stats |
| `ahj_cancellation_rates.csv` | `pipeline.ipynb` | processed outside this repo (no script) | Ohm; uploaded to Drive only |
| `ahj_distribution_permitting_timelines_pv.csv` | `pipeline.ipynb`, `median_permit_fees.py` | processed outside this repo (no script) | Ohm; provides AHJ population weights for fees module |
| `state_distribution_permitting_timelines_pv.csv` | `pipeline.ipynb` | processed outside this repo (no script) | Ohm; uploaded to Drive only |
| `ahj_median_permitting_timelines_by_tech.csv` | `pipeline.ipynb` | processed outside this repo (no script) | Ohm; uploaded to Drive only |
| `ahj_inspection_timelines_pv.csv` | `pipeline.ipynb` | processed outside this repo (no script) | Ohm; uploaded to Drive only |
| `state_inspection_timelines_pv.csv` | `ad_hocs/solartrace_exploration.ipynb` | processed outside this repo (no script) | Ohm; **no longer used by the pipeline**, superseded by Solar TRACE |
| `state_median_permitting_timelines_by_tech.csv` | `ad_hocs/solartrace_exploration.ipynb` | processed outside this repo (no script) | Ohm; **no longer used by the pipeline**, superseded by Solar TRACE |
| `solar_jobs.csv` | `pipeline.ipynb` | unknown | IREC; uploaded to Drive only |
| `state_name_abbr.csv` | `pipeline.ipynb`, `compile_state_stats.ipynb` | lookup table | State name ↔ abbreviation mapping |
| `nrel_solar_suitability_by_zip.csv` | `solar_eligible_households.py` | downloaded source snapshot | NREL rooftop suitability by ZIP |
| `techpot_baseline_state.csv` | `solar_technical_potential.py` | downloaded source snapshot | NREL SLOPE state totals |

---

## Modules

| Module | What it does |
|--------|-------------|
| `pull_solar_count_and_capacity.py` | Downloads annual EIA 861 files (861M reader kept, unused); aggregates PV/battery capacity and customer counts by state, sector and year |
| `solar_eligible_households.py` | Filters ResStock metadata to estimate solar-eligible household counts by state |
| `solar_technical_potential.py` | Loads NREL SLOPE residential PV technical generation potential by state |
| `median_solar_costs.py` | Reads LBNL's Tracking the Sun data tool export: residential median $/kW and system size by state and year; `latest_by_state` picks each state's most recent reliable year for state_stats |
| `average_electricity_prices.py` | State prices from EIA API (2018 onward); utility prices from EIA 861 annual files (2018–2024); inflation-adjusted to 2026 dollars |
| `process_solartrace_timelines.py` | Reshapes Solar TRACE xlsx into tidy long-format tables; produces `state_timelines` (direct reshape) and `utility_timelines` (AHJ→utility weighted-median aggregation) |
| `median_permit_fees.py` | Aggregates Solar TRACE fee data; population-weighted median fees by state |
| `median_interconnection_timelines.py` | **Legacy.** Processes old `solartrace_ix.csv`; superseded by `process_solartrace_timelines.py` for state_stats purposes but still runs in pipeline |
| `solar_bill_savings_update.py` | Loads pre-computed dGen bill savings from `data/bill_savings_csvs/` |
| `drive_uploader.py` | Google Drive helpers: authenticate, ensure folder path, upload DataFrame as CSV |
| `manifest.py` | Builds and uploads a JSON manifest of all Drive files |
| `storage_attachment_rate.py` | Loads and summarizes Ohm Analytics monthly storage attachment rates by state (raw export `data/ohm_analytics_*.csv` stays local, gitignored). **Not yet wired into the pipeline** |

---

## Running the pipeline

```bash
poetry install --no-interaction
poetry run pre-commit install   # once per clone: runs ruff (pyflakes rules) on every commit

# Stage 1 (computes outputs without publishing)
cd notebooks
poetry run papermill pipeline.ipynb pipeline-output.ipynb -p UPLOAD False

# Stage 2
poetry run papermill compile_state_stats.ipynb compile_state_stats-output.ipynb
```

`pipeline.ipynb` takes one papermill parameter, `UPLOAD` (default `True`, which the GitHub Action uses). With `-p UPLOAD False` it skips Drive authentication, folder creation, sheet uploads, and the manifest, and still writes `output_csvs/`. Uploading overwrites the shared Big Numbers Drive sheets, so use `UPLOAD False` for local runs. Stage 2 never uploads.

### Tests

```bash
poetry run pytest
```

`tests/test_state_stats.py` checks `output_csvs/state_stats.csv` (run both stages first). Columns built only from `data/` must match `tests/golden/state_stats_static.csv` exactly; the five EIA-derived columns (electricity price and bill, installations, capacity, penetration) get range and null checks.

### Authentication

In CI, Workload Identity Federation handles Google Drive auth automatically. Locally:

```bash
gcloud auth application-default login \
  --scopes=https://www.googleapis.com/auth/drive,https://www.googleapis.com/auth/cloud-platform
```

API keys required (set in `.env` or as environment variables):
- `EIA_API_KEY` — for electricity prices
- `FRED_API_KEY` — for CPI inflation data (electricity prices module)

---

MSA utility approval tracking moved out of this repo in October 2026.

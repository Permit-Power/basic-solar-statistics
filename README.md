# basic-solar-statistics

Pulls, processes, and organizes solar statistics for U.S. states and jurisdictions. A GitHub Action runs the full pipeline on the 1st of each month, uploading results to a shared Google Drive folder ("The Big Numbers Database").

---

## Pipeline architecture

The pipeline runs in two stages, both executed as Jupyter notebooks via `papermill`.

```
Stage 1: pipeline.ipynb
  - Pulls live data from EIA, LBNL, and other sources
  - Processes static input files from data/
  - Uploads all outputs to Google Drive
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
| Solar capacity | Downloads EIA 861M files for 2017–2025, aggregates PV/battery stats | EIA 861M (live) | `capacity` |
| Solar eligibility | Filters ResStock households for rooftop suitability | ResStock metadata | `eligibility` |
| Solar costs | Downloads LBNL Tracking the Sun, applies CPI inflation adjustment | LBNL / FRED API (live) | `costs` |
| Electricity prices | Pulls state prices from EIA API (2015–2026), utility prices from EIA 861 files (2015–2024) | EIA API + EIA 861 (live) | `state_prices`, `util_prices` |
| Permitting fees | Aggregates Solar TRACE fee data by AHJ and state | `data/solartrace_fees.csv` | `ahj_df_fees`, `state_df_fees` |
| IX timelines (legacy) | Aggregates old Solar TRACE IX CSV by AHJ and state | `data/solartrace_ix.csv` | `ahj_df_ix`, `state_df_ix` |
| Solar TRACE timelines | Reshapes Solar TRACE xlsx into tidy utility and state tables | `data/SolarTRACE Dataset v9-9-2025.xlsx` | `utility_timelines`, `state_timelines` |
| Bill savings | Loads pre-computed dGen savings | `data/bill_savings_csvs/` | `savings` |
| Export CSVs | Writes all processed DataFrames to `output_csvs/` | — | — |

### output_csvs/ written by Stage 1

| File | Source | Contents |
|------|--------|----------|
| `solar_storage_capacity_installations_by_state_sector.csv` | EIA 861M | Cumulative solar/storage capacity and install counts by state, sector, year |
| `solar_eligible_households_by_state.csv` | ResStock | Eligible and suitable household counts by state |
| `potential_solar_generation_by_state.csv` | NREL SLOPE | Technical generation potential in GWh by state |
| `residential_solar_costs_by_state_over_time.csv` | LBNL TTS | Median $/kW and system size (kW), inflation-adjusted, by state and year |
| `annual_and_lifetime_solar_savings_by_state.csv` | dGen | Median year-1 and lifetime bill savings by state |
| `electricity_rates_by_state.csv` | EIA API | Avg price ($/kWh) and annual bill by state, sector, year (2026 $) |
| `electricity_rates_by_utility.csv` | EIA 861 | Avg price ($/kWh) and annual bill by utility, sector, year (2026 $) |
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
| Median solar savings over lifetime | same | — |
| Average electricity retail cost ($/kWh) | `output_csvs/electricity_rates_by_state.csv` | Latest year, residential sector |
| Average annual electricity bill | same | Latest year, residential sector |
| Number of solar installations | `output_csvs/solar_storage_capacity_installations_by_state_sector.csv` | Latest year, residential sector |
| Total installed solar capacity (MW) | same | Latest year, residential sector |
| Total households | `output_csvs/solar_eligible_households_by_state.csv` | — |
| Solar eligible and suitable households | same | — |
| Percent of total households suitable | same | — |
| Solar penetration | computed: installs / eligible households | — |
| Annual technical potential (GWh) | `output_csvs/potential_solar_generation_by_state.csv` | — |
| Median solar cost ($/kW) | `output_csvs/residential_solar_costs_by_state_over_time.csv` | 2024 |
| Median solar system size (kW) | same | 2024 |
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

The following Ohm files are **no longer used** anywhere in the pipeline and are kept only as historical reference:
- `data/state_inspection_timelines_pv.csv` — replaced by Solar TRACE
- `data/state_median_permitting_timelines_by_tech.csv` — replaced by Solar TRACE

---

## data/ folder — static inputs

| File | Used by | Notes |
|------|---------|-------|
| `SolarTRACE Dataset v9-9-2025.xlsx` | `process_solartrace_timelines.py` | Primary timeline source |
| `solartrace_fees.csv` | `median_permit_fees.py` | Permitting fee data |
| `solartrace_ix.csv` | `median_interconnection_timelines.py` | Legacy IX data; still uploaded to Drive but not used in state_stats |
| `TTS_LBNL_public_file_29-Sep-2025_all.csv` | `median_solar_costs.py` | LBNL Tracking the Sun (local copy) |
| `resstock_metadata_technical_potential.csv` | `solar_eligible_households.py` | ~34 MB; ResStock household metadata |
| `bill_savings_csvs/` | `solar_bill_savings_update.py` | Pre-computed dGen outputs; must exist before pipeline runs |
| `state_cancellation_rates.csv` | `compile_state_stats.ipynb` | Ohm; used in state_stats |
| `ahj_cancellation_rates.csv` | `pipeline.ipynb` | Ohm; uploaded to Drive only |
| `ahj_distribution_permitting_timelines_pv.csv` | `pipeline.ipynb`, `median_permit_fees.py` | Ohm; provides AHJ population weights for fees module |
| `state_distribution_permitting_timelines_pv.csv` | `pipeline.ipynb` | Ohm; uploaded to Drive only |
| `ahj_median_permitting_timelines_by_tech.csv` | `pipeline.ipynb` | Ohm; uploaded to Drive only |
| `ahj_inspection_timelines_pv.csv` | `pipeline.ipynb` | Ohm; uploaded to Drive only |
| `state_inspection_timelines_pv.csv` | — | Ohm; **no longer used**, superseded by Solar TRACE |
| `state_median_permitting_timelines_by_tech.csv` | — | Ohm; **no longer used**, superseded by Solar TRACE |
| `solar_jobs.csv` | `pipeline.ipynb` | IREC; uploaded to Drive only |
| `state_name_abbr.csv` | `pipeline.ipynb`, `compile_state_stats.ipynb` | State name ↔ abbreviation mapping |
| `utility_interconnection_fees.csv` | — | Reference file for interconnection fee research |
| `nrel_solar_suitability_by_zip.csv` | `solar_eligible_households.py` | NREL rooftop suitability by ZIP |
| `solar_potential.csv` | `solar_technical_potential.py` | NREL SLOPE technical potential |
| `techpot_baseline_state.csv` | `solar_technical_potential.py` | NREL SLOPE state totals |

---

## Modules

| Module | What it does |
|--------|-------------|
| `pull_solar_count_and_capacity.py` | Downloads EIA 861M Excel files; aggregates residential PV/battery capacity and customer counts by state and year |
| `solar_eligible_households.py` | Filters ResStock metadata to estimate solar-eligible household counts by state |
| `solar_technical_potential.py` | Loads NREL SLOPE residential PV technical generation potential by state |
| `median_solar_costs.py` | Processes LBNL Tracking the Sun data; applies CPI inflation adjustment from FRED; computes median $/kW and system size by state and year |
| `average_electricity_prices.py` | State prices from EIA API (2015–2026); utility prices from EIA 861 annual files (2015–2024); inflation-adjusted to 2026 dollars |
| `process_solartrace_timelines.py` | Reshapes Solar TRACE xlsx into tidy long-format tables; produces `state_timelines` (direct reshape) and `utility_timelines` (AHJ→utility weighted-median aggregation) |
| `median_permit_fees.py` | Aggregates Solar TRACE fee data; population-weighted median fees by state |
| `median_interconnection_timelines.py` | **Legacy.** Processes old `solartrace_ix.csv`; superseded by `process_solartrace_timelines.py` for state_stats purposes but still runs in pipeline |
| `solar_bill_savings_update.py` | Loads pre-computed dGen bill savings from `data/bill_savings_csvs/` |
| `drive_uploader.py` | Google Drive helpers: authenticate, ensure folder path, upload DataFrame as CSV |
| `manifest.py` | Builds and uploads a JSON manifest of all Drive files |

---

## Running the pipeline

```bash
poetry install --no-interaction
poetry run pip install papermill

# Stage 1
cd notebooks
poetry run papermill pipeline.ipynb pipeline-output.ipynb

# Stage 2
poetry run papermill compile_state_stats.ipynb compile_state_stats-output.ipynb
```

### Authentication

In CI, Workload Identity Federation handles Google Drive auth automatically. Locally:

```bash
gcloud auth application-default login \
  --scopes=https://www.googleapis.com/auth/drive,https://www.googleapis.com/auth/cloud-platform
```

API keys required (set in `.env` or as environment variables):
- `EIA_API_KEY` — for electricity prices
- `FRED_API_KEY` — for CPI inflation data (solar costs module)

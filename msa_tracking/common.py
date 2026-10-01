"""Paths, settings and run logging shared by the MSA tracking notebooks."""

import json
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Inputs
RAW_MSA_DIR = ROOT / "data" / "raw" / "msa"      # one folder per snapshot date (git-ignored)
RAW_EIA_DIR = ROOT / "data" / "raw" / "eia"      # one folder per EIA-861 year (git-ignored)
CROSSWALK_CSV = ROOT / "data" / "msa_eia_crosswalk.csv"  # hand-reviewed

# Outputs
OUTPUT_DIR = ROOT / "outputs"
MSA_MASTER_CSV = OUTPUT_DIR / "msa_master.csv"
MSA_SUMMARY_CSV = OUTPUT_DIR / "msa_summary.csv"
EIA_CUSTOMERS_CSV = OUTPUT_DIR / "eia_residential_customers.csv"
UTILITIES_ALL_CSV = OUTPUT_DIR / "msa_utilities_all.csv"
STATE_SUMMARY_CSV = OUTPUT_DIR / "msa_state_summary.csv"

LOG_FILE = ROOT / "run_log.txt"

# EIA-861 year used for residential customer counts. 2024 is the latest final
# year; a 2025 early release exists but excludes some utilities.
EIA_YEAR = 2024

# MSA sources: key used for raw file names and fetch_status.json -> display name.
MSA_SOURCES = {"connectder": "ConnectDER", "enphase": "Enphase", "tesla": "Tesla"}
MSA_SOURCE_PAGES = {
    "ConnectDER": "https://connectder.com/utility-approval-status",
    "Enphase": "https://enphase.com/installers/storage/gen4/iq-meter-collar/approvals",
    "Tesla": "https://www.tesla.com/support/energy/powerwall/learn/tesla-backup-switch",
}


def get_logger(name):
    """Logger that appends to run_log.txt and prints. Safe to call again when
    a notebook cell is re-run."""
    log = logging.getLogger(name)
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s  %(name)-12s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
        for handler in (logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)):
            handler.setFormatter(fmt)
            log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


def msa_sources_due(today=None):
    """MSA sources not yet fetched successfully this calendar month, according
    to each snapshot's fetch_status.json."""
    month = (today or date.today().isoformat())[:7]
    done = set()
    for status_file in RAW_MSA_DIR.glob(f"{month}-??/fetch_status.json"):
        done.update(json.loads(status_file.read_text())["ok"])
    return [s for s in MSA_SOURCES if s not in done]


if __name__ == "__main__":
    # Used by run_scheduled.sh: prints the sources still due this month.
    print(" ".join(msa_sources_due()))

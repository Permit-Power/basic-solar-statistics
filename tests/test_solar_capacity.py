"""
Tests on output_csvs/solar_storage_capacity_installations_by_state_sector.csv
(annual EIA 861, see modules/pull_solar_count_and_capacity.py).

Expected 2025 values were summed by hand from Net_Metering_2025.xlsx and
Non_Net_Metering_Distributed_2025.xlsx (all utility rows, including EIA's
Adjustment rows), in October 2026. If EIA revises the 2025 files, update them
and say which values moved.
"""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CAPACITY = ROOT / "output_csvs" / "solar_storage_capacity_installations_by_state_sector.csv"

if not CAPACITY.is_file():
    pytest.skip(
        "solar capacity CSV not found; run the pipeline first (see README, 'Running the pipeline').",
        allow_module_level=True,
    )

# state: (residential installs incl. estimated non-net, residential MW AC)
EXPECTED_2025 = {
    "CA": (2_243_993, 13_802.9),
    "MD": (118_789, 997.7),
    "NJ": (219_450, 1_526.7),
    "PA": (93_934, 760.0),
    "TX": (396_535, 2_890.8),
}


@pytest.fixture(scope="module")
def res():
    df = pd.read_csv(CAPACITY)
    return df[df["sector"] == "Residential"].set_index(["year", "state"])


def test_years_and_states(res):
    counts = res.groupby(level="year").size()
    assert list(counts.index) == list(range(2017, 2026))
    assert (counts == 51).all(), counts.to_dict()


@pytest.mark.parametrize("state", list(EXPECTED_2025))
def test_2025_matches_eia_files(res, state):
    installs, mw = EXPECTED_2025[state]
    row = res.loc[(2025, state)]
    assert row["pv_customers"] == installs
    assert round(row["pv_capacity_mw"], 1) == mw

"""
Tests on output_csvs/state_stats.csv, which feeds funder and legislative materials.

Static columns come only from files committed in data/, so they must match
tests/golden/state_stats_static.csv exactly. If one of these fails after you
replaced an input file, report which values moved before regenerating the golden file.

Live columns depend on EIA data pulled at run time, so they only get sanity checks.
"""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
STATE_STATS = ROOT / "output_csvs" / "state_stats.csv"
GOLDEN = ROOT / "tests" / "golden" / "state_stats_static.csv"

if not STATE_STATS.is_file():
    pytest.skip(
        "output_csvs/state_stats.csv not found; run the pipeline first "
        "(see README, 'Running the pipeline').",
        allow_module_level=True,
    )

# (min, max) inclusive, set from the October 2026 run with wide margins.
LIVE_RANGES = {
    "Average electricity retail cost ($/kWh)": (0.05, 0.80),   # now 0.12 to 0.42
    "Average annual electricity bill": (500, 5000),            # now 1,213 to 2,557
    "Number of solar installations": (0, 5_000_000),           # now 0 (AL) to 2.2M
    "Total installed solar capacity (MW)": (0.1, 40_000),      # now 1.4 to 13,800
    "Solar penetration": (0, 1.0),                             # now 0 (AL) to 0.54
}

# Alabama reports 0 residential installations but 4.5 MW of capacity in EIA 861M.
# Looks like a data bug; allowed here until it's investigated.
KNOWN_ZERO_INSTALLS = {"AL"}


@pytest.fixture(scope="module")
def stats():
    return pd.read_csv(STATE_STATS)


def test_columns_match_golden_plus_live(stats):
    golden_cols = list(pd.read_csv(GOLDEN, nrows=0).columns)
    assert sorted(stats.columns) == sorted(golden_cols + list(LIVE_RANGES))


def test_static_columns_match_golden(stats):
    golden = pd.read_csv(GOLDEN).set_index("State").sort_index()
    current = stats[golden.reset_index().columns].set_index("State").sort_index()
    pd.testing.assert_frame_equal(current, golden, check_exact=False, rtol=1e-9)


def test_one_row_per_state(stats):
    assert len(stats) == 51  # 50 states + DC
    assert not stats["State"].duplicated().any()
    assert not stats["State (Abbr.)"].duplicated().any()


@pytest.mark.parametrize("col", list(LIVE_RANGES))
def test_live_column_has_no_nulls(stats, col):
    assert stats[col].notna().all(), stats.loc[stats[col].isna(), "State"].tolist()


@pytest.mark.parametrize("col", list(LIVE_RANGES))
def test_live_column_in_range(stats, col):
    lo, hi = LIVE_RANGES[col]
    out = stats.loc[~stats[col].between(lo, hi), ["State", col]]
    assert out.empty, out.to_string()


def test_zero_installations_only_known_states(stats):
    zero = set(stats.loc[stats["Number of solar installations"] == 0, "State (Abbr.)"])
    assert zero <= KNOWN_ZERO_INSTALLS, zero - KNOWN_ZERO_INSTALLS

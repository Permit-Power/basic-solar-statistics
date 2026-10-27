"""
Checks on data/residential_fixed_charges_manual.csv, the hand-checked residential
fixed charges that replace URDB values (see modules/average_electricity_prices.py).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANUAL = pd.read_csv(ROOT / "data" / "residential_fixed_charges_manual.csv")
EVIDENCE = {"tariff", "tariff (inferred)", "secondary", "not found"}


def test_one_row_per_utility_year():
    assert not MANUAL.duplicated(["year", "state", "utility_number"]).any()


def test_evidence_labels():
    assert set(MANUAL["evidence"]) <= EVIDENCE


def test_blank_charge_only_when_not_found():
    blank = MANUAL["fixed_charge_monthly"].isna()
    assert (blank == (MANUAL["evidence"] == "not found")).all()


def test_charges_have_a_source_and_are_in_range():
    found = MANUAL[MANUAL["evidence"] != "not found"]
    assert found["source_url"].str.startswith("http").all()
    assert found["fixed_charge_monthly"].between(0, 50).all()

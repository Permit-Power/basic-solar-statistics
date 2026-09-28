"""Parse every raw MSA snapshot and rebuild msa_master.csv and msa_summary.csv.

The master list is regenerated from ALL folders in raw/ on every run (never
appended), so fixing a parser or a mapping below corrects the whole history.

msa_master.csv, long format: one row per snapshot_date x manufacturer x product
x state x utility. Verbatim fields are kept as written by the manufacturer;
*_std columns are added alongside. Nothing is filtered for scope.

msa_summary.csv, wide format: the current picture, one row per state x
utility with one status column per brand (see SUMMARY_STATUS), built from each
brand's latest snapshot. Rows without a state are left out.

    poetry run python msa_tracking/processing.py
"""

import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
OUT_CSV = HERE / "msa_master.csv"
SUMMARY_CSV = HERE / "msa_summary.csv"
LOG_FILE = HERE / "run_log.txt"

SOURCE_URLS = {
    "ConnectDER": "https://connectder.com/utility-approval-status",
    "Enphase": "https://enphase.com/installers/storage/gen4/iq-meter-collar/approvals",
    "Tesla": "https://www.tesla.com/support/energy/powerwall/learn/tesla-backup-switch",
}

# ---------------------------------------------------------------------------
# Standardization dictionaries
# ---------------------------------------------------------------------------

# Approval level as written -> standardized level. Anything not listed here is
# written as "UNMAPPED" and reported in the run log.
# Tesla and Enphase only publish a list of approved utilities, so being on it
# is recorded as "Listed".
APPROVAL_STD = {
    "Listed": "approved",
    "Approved": "approved",
    "Pilot in progress": "pilot",
    "Approvals on case-by-case basis": "case_by_case",
    "In Progress (TBD)": "in_progress",
    "Expected Q, YYYY": "expected",
    "N/A": "not_applicable",
}

# Standardized approval level -> status in the per-brand summary. A brand's
# status for a utility is its best one across all its products, in the order
# listed in SUMMARY_ORDER. Not approved (N/A, or not listed at all) is left
# blank so approvals stand out.
SUMMARY_STATUS = {
    "approved": "Approved",
    "pilot": "Pilot",
    "case_by_case": "Case-by-case",
    "in_progress": "Pending",
    "expected": "Pending",
    "not_applicable": "",
}
SUMMARY_ORDER = ["Approved", "Pilot", "Case-by-case", "Pending", ""]
SUMMARY_BRANDS = ["Tesla", "Enphase", "ConnectDER"]

# Enphase status pills that describe a limitation rather than an approval
# level. These go in `notes` and the approval stays "Listed".
ENPHASE_LIMITATION_PILLS = {"Ring-type meter base only"}

# Standardized utility name -> the spellings manufacturers use for it. Only
# utilities spelled differently across sources need an entry; any other name
# passes through unchanged.
UTILITY_VARIANTS = {
    "Arizona Public Service": ["Arizona Public Service (APS)", "Arizona Public Service Company"],
    "Atlantic City Electric": ["ACE", "Atlantic City Electric (ACE)"],
    "Austin Energy": ["Austin Energy (City of Austin)"],
    "Baltimore Gas and Electric": ["Baltimore Gas & Electric", "Baltimore Gas & Electric (BGE)",
                                   "Baltimore Gas and Electric (BGE)"],
    "Black Hills Energy": ["Black Hills Corporation"],
    "Bluebonnet Electric Cooperative": ["Bluebonnet Electric Coop (BB)"],
    "Buckeye Rural Electric Cooperative": ["Buckeye Rural Electric", "Buckeye Rural Electrical Cooperative"],
    "City of Tallahassee": ["City of Tallahassee Electric Utility"],
    "Commonwealth Edison": ["ComEd"],
    "CoServ Electric Cooperative": ["CoServ"],
    "Delmarva Power": ["Delmarva", "DPL"],
    # Enphase's "Eau Clair Utility" is taken to be the co-op; the city of Eau
    # Claire has no municipal electric utility.
    "Eau Claire Energy Cooperative": ["Eau Clair Utility", "Eau Claire Energy Coop"],
    "Fort Collins Light and Power": ["City of Ft. Collins Utility", "Fort Collins Light & Power",
                                     "Fort Collins Light and Power (FCLP)"],
    "Green Mountain Power": ["Green Mountain Power (GMP)"],
    "Hawaiian Electric": ["Hawaiian Electric Company", "Heco"],
    "Hohokam Irrigation and Drainage District": ["Hohokam Irrigation & Drainage District",
                                                 "Hohokam Irrigation and Power"],
    "Jersey Central Power and Light": ["Jersey Central Power & Light"],
    "La Plata Electric Association": ["La Plata Electric Association (LPEA)"],
    "Los Alamos County Utilities": ["Los Alamos Department of public utilities"],
    "NV Energy": ["NV Energy (South and North)"],
    "Omaha Public Power District": ["Omaha Public Power District (OPPD)"],
    "Orlando Utilities Commission": ["OUC (Orlando Utilities Commission)"],
    "Pacific Gas and Electric": ["Pacific Gas & Electric", "Pacific Gas and Electric Company",
                                 "Pacific Gas and Electric Company (PG&E)"],
    "Pepco": ["PEPCO", "Potomac Electric Power Company (Pepco)"],
    "Poudre Valley Rural Electric Association": ["Poudre Valley Electric Association"],
    "Public Service Electric and Gas": ["Public Service Electirc & Gas (PSE&G) NJ",
                                        "Public Service Electric & Gas Company"],
    "Rocky Mountain Power": ["Rocky Mountain Power (RMP)"],
    "Sacramento Municipal Utility District": ["Sacramento Municipal Utility District (SMUD)"],
    "Salt River Project": ["Salt River Project (SRP)"],
    "San Diego Gas and Electric": ["San Diego Gas & Electric", "San Diego Gas & Electric (SDG&E)"],
    "Southern California Edison": ["Southern California Edison (SCE)"],
    "Sulphur Springs Valley Electric Cooperative": ["Sulphur Spring Valley Electric",
                                                    "Sulphur Springs Valley Electric Coop"],
    "Tucson Electric Power": ["Tucson Electric Power (TEP)"],
    "Vermont Electric Cooperative": ["Vermont Electric Coop"],
    "Washington Electric Cooperative": ["Washington Electric Coop", "Washington Electrical Cooperative"],
    "West River Electric Association": ["West River Electric Association (WREA)"],
    "Westerville Electric": ["City of Westerville OH"],
    "Xcel Energy": ["Xcel Energy-Colorado"],
    "Yellow Springs Electric": ["Village of Yellow Springs"],
}
UTILITY_STD = {variant: std for std, variants in UTILITY_VARIANTS.items() for variant in variants}

STATE_STD = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Washington DC": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "Puerto Rico": "PR",
    # Canada (Enphase lists a few provinces)
    "Nova Scotia": "NS", "Saskatchewan": "SK", "Alberta": "AB", "British Columbia": "BC",
    "Ontario": "ON", "Quebec": "QC", "Manitoba": "MB", "New Brunswick": "NB",
}

COLUMNS = [
    "snapshot_date", "manufacturer", "product", "state", "state_std", "utility", "utility_std",
    "utility_other_names", "approval_raw", "approval_std", "expected_date", "install_type",
    "notes", "source_url",
]

log = logging.getLogger("msa_processing")


def setup_logging():
    fmt = logging.Formatter("%(asctime)s  %(name)-14s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def clean(text):
    # Collapses whitespace and drops zero-width characters (one ConnectDER name
    # carries a trailing U+200B).
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text or "")
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Parsers: one per manufacturer. Each takes a snapshot folder and returns a
# list of row dicts (standardized columns are added later in one place).
# ---------------------------------------------------------------------------

def parse_connectder(snap_dir):
    """Prismic CMS JSON. One utility_provider document per utility, with one
    product_statuses entry per state it serves, each holding three products."""
    docs = []
    for path in sorted(snap_dir.glob("connectder_p*.json")):
        docs += json.loads(path.read_text())["results"]
    if not docs:
        return None

    install_types = {d["id"]: d["data"]["label"] for d in docs if d["type"] == "utility_install_type"}
    products = {"islandder": "IslandDER MSA", "solar_msa": "Solar MSA", "ev_msa": "EV MSA"}

    rows = []
    for doc in docs:
        if doc["type"] != "utility_provider":
            continue
        data = doc["data"]
        other_names = "; ".join(o["other_name"] for o in data.get("other_names") or [] if o.get("other_name"))
        for ps in data["product_statuses"]:
            install_type = install_types.get((ps.get("utility_install_type") or {}).get("id"), "")
            for key, product in products.items():
                rows.append({
                    "manufacturer": "ConnectDER",
                    "product": product,
                    "state": ps.get("state") or "",
                    "utility": clean(data["name"]),
                    "utility_other_names": other_names,
                    "approval_raw": ps.get(f"{key}_status") or "",
                    "expected_date": ps.get(f"{key}_due_date") or "",
                    "install_type": install_type,
                    "notes": "",
                })
    return rows


def parse_enphase(snap_dir):
    """Static Drupal HTML. The page lists utilities twice (sorted by state and
    by utility); only the 'Sort by state' tab is read."""
    path = snap_dir / "enphase.html"
    if not path.exists():
        return None
    soup = BeautifulSoup(path.read_text(), "html.parser")
    tab = soup.select_one('.tabs__item[data-tab="Sort by state"]')
    if tab is None:
        raise ValueError("'Sort by state' tab not found; page layout changed")

    rows = []
    for state_block in tab.select(".state-accordion"):
        state = clean(state_block.select_one(".state-accordion__state-name").get_text())
        for item in state_block.select("li.state-accordion__list-item"):
            pill = item.select_one(".state-accordion__status-pill")
            pill_text = clean(pill.get_text()) if pill else ""
            if pill:
                pill.extract()
            is_limitation = pill_text in ENPHASE_LIMITATION_PILLS
            rows.append({
                "manufacturer": "Enphase",
                "product": "IQ Meter Collar",
                "state": state,
                "utility": clean(item.get_text()),
                "approval_raw": "Listed" if not pill_text or is_limitation else pill_text,
                "notes": pill_text if is_limitation else "",
            })
    return rows


def parse_tesla(snap_dir):
    """Static HTML accordions. Two lists: Backup Switch approved for Powerwall
    installs and for Powershare installs, each under its own h3 heading."""
    path = snap_dir / "tesla.html"
    if not path.exists():
        return None
    soup = BeautifulSoup(path.read_text(), "html.parser")

    rows = []
    for accordion in soup.select("section.tcl-accordion"):
        heading = clean(accordion.find_previous("h3").get_text())
        match = re.search(r"for (Powerwall|Powershare) Installation", heading)
        if not match:
            raise ValueError(f"unexpected heading above accordion: {heading!r}")
        product = f"Backup Switch ({match.group(1)})"
        for item in accordion.select(".tcl-accordion__item"):
            state = clean(item.select_one(".tcl-accordion__title").get_text())
            for li in item.select(".tcl-accordion__panel li"):
                rows.append({
                    "manufacturer": "Tesla",
                    "product": product,
                    "state": state,
                    "utility": clean(li.get_text()),
                    "approval_raw": "Listed",
                    "notes": "",
                })
    if not rows:
        raise ValueError("no utilities found; page layout changed")
    return rows


PARSERS = {"ConnectDER": parse_connectder, "Enphase": parse_enphase, "Tesla": parse_tesla}


def build_summary(master):
    """One row per state x utility, one status column per brand, using each
    brand's most recent snapshot."""
    latest = master.groupby("manufacturer")["snapshot_date"].transform("max")
    current = master[(master["snapshot_date"] == latest) & (master["state_std"] != "")].copy()
    current["status"] = current["approval_std"].map(SUMMARY_STATUS).fillna("UNMAPPED")
    current["rank"] = current["status"].map(
        {s: i for i, s in enumerate(SUMMARY_ORDER)}).fillna(len(SUMMARY_ORDER))

    best = current.sort_values("rank").drop_duplicates(["state_std", "utility_std", "manufacturer"])
    summary = (
        best.pivot(index=["state_std", "utility_std"], columns="manufacturer", values="status")
        .reindex(columns=SUMMARY_BRANDS).fillna("").reset_index()
    )
    state_names = {}
    for name, abbr in STATE_STD.items():
        state_names.setdefault(abbr, name)
    summary.insert(0, "state", summary.pop("state_std").map(state_names))
    summary = summary.rename(columns={"utility_std": "utility"})
    summary.columns.name = None
    return summary.sort_values(["state", "utility"], key=lambda c: c.str.lower())


def main():
    setup_logging()
    snapshots = sorted(p for p in RAW_DIR.glob("????-??-??") if p.is_dir())
    log.info("processing start  %d snapshot(s) in raw/", len(snapshots))

    frames, errors = [], 0
    for snap_dir in snapshots:
        counts = []
        for manufacturer, parse in PARSERS.items():
            try:
                rows = parse(snap_dir)
            except Exception as e:
                errors += 1
                log.error("FAILED  %s %-10s %s: %s", snap_dir.name, manufacturer, type(e).__name__, e)
                continue
            if rows is None:
                counts.append(f"{manufacturer}=missing")
                continue
            df = pd.DataFrame(rows)
            df["snapshot_date"] = snap_dir.name
            df["source_url"] = SOURCE_URLS[manufacturer]
            frames.append(df)
            counts.append(f"{manufacturer}={len(df)}")
        log.info("parsed  %s  %s", snap_dir.name, "  ".join(counts))

    if not frames:
        log.error("processing done  no rows parsed; %s not written", OUT_CSV.name)
        return 1

    master = pd.concat(frames, ignore_index=True).reindex(columns=COLUMNS).fillna("")
    master["state_std"] = master["state"].map(STATE_STD).fillna("")
    master["utility_std"] = master["utility"].map(lambda u: UTILITY_STD.get(u, u))
    master["approval_std"] = master["approval_raw"].map(APPROVAL_STD).fillna("UNMAPPED")

    for col, std in [("approval_raw", "approval_std"), ("state", "state_std")]:
        # A blank state is verbatim from the source, not a missing mapping.
        unmapped = master.loc[master[std].isin(["", "UNMAPPED"]) & (master[col] != ""), col].unique()
        if len(unmapped):
            log.warning("unmapped %s values (add to dictionary): %s", col, sorted(unmapped))

    master = master.sort_values(["snapshot_date", "manufacturer", "product", "state", "utility"])
    master.to_csv(OUT_CSV, index=False)
    summary = build_summary(master)
    summary.to_csv(SUMMARY_CSV, index=False)
    latest = master.groupby("manufacturer")["snapshot_date"].max()
    log.info("summary  %d state x utility rows written to %s (latest snapshots: %s)", len(summary),
             SUMMARY_CSV.name, ", ".join(f"{m} {d}" for m, d in latest.items()))
    log.info("processing done  %d rows written to %s%s", len(master), OUT_CSV.name,
             f"; {errors} parser error(s)" if errors else "")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

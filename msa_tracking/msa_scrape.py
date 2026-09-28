"""Download each manufacturer's meter socket adapter (MSA) utility-approval listing.

Only fetches and saves; no parsing. Each source's raw response is written
untouched to raw/YYYY-MM-DD/ so history survives parser breakage. A failure in
one source is logged and does not stop the others.

    poetry run python msa_tracking/msa_scrape.py              # fetch all sources
    poetry run python msa_tracking/msa_scrape.py --catch-up   # scheduled mode

--catch-up fetches only the sources that have not been fetched successfully
yet this calendar month (per each snapshot's fetch_status.json), and exits
with code 3 without doing anything when the month is already complete. The
scheduler runs it daily, so a missed or failed month is picked up on the next
day the machine is on.
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
LOG_FILE = HERE / "run_log.txt"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)

ENPHASE_URL = "https://enphase.com/installers/storage/gen4/iq-meter-collar/approvals"
TESLA_URL = "https://www.tesla.com/support/energy/powerwall/learn/tesla-backup-switch"
# connectder.com sits behind a Vercel bot checkpoint, but the page's content
# comes from this public Prismic CMS API, which returns the same data as JSON.
CONNECTDER_API = "https://connectder.cdn.prismic.io/api/v2"
CONNECTDER_TYPES = ["utility_provider", "utility_install_type", "utility_approval_status"]

log = logging.getLogger("msa_scrape")


def setup_logging():
    fmt = logging.Formatter("%(asctime)s  %(name)-14s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def fetch_enphase(out_dir):
    resp = requests.get(ENPHASE_URL, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    path = out_dir / "enphase.html"
    path.write_bytes(resp.content)
    return [path]


def fetch_connectder(out_dir):
    session = requests.Session()
    ref = session.get(CONNECTDER_API, timeout=60).json()["refs"][0]["ref"]
    types = ",".join(f'"{t}"' for t in CONNECTDER_TYPES)
    paths, page, total_pages = [], 1, 1
    while page <= total_pages:
        resp = session.get(
            f"{CONNECTDER_API}/documents/search",
            params={"ref": ref, "q": f"[[any(document.type,[{types}])]]", "pageSize": 100, "page": page},
            timeout=60,
        )
        resp.raise_for_status()
        total_pages = resp.json()["total_pages"]
        path = out_dir / f"connectder_p{page}.json"
        path.write_bytes(resp.content)
        paths.append(path)
        page += 1
    return paths


def fetch_tesla(out_dir):
    # Akamai rejects requests, curl, and headless Chromium with 403. A headed
    # Chromium with the automation flag disabled gets through; the window is
    # parked off-screen so it doesn't get in the way. The first response is an
    # Akamai JS challenge that reloads into the real page, so we keep the last
    # document response and only accept it once it contains the listing.
    from playwright.sync_api import sync_playwright

    documents = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--window-position=-2400,-2400"],
        )
        try:
            page = browser.new_context(locale="en-US").new_page()
            page.on("response", lambda r: documents.append(r) if r.request.is_navigation_request()
                    and r.frame == page.main_frame else None)
            page.goto(TESLA_URL, wait_until="domcontentloaded", timeout=90000)
            try:
                page.wait_for_selector("text=Utilities That Have Approved Backup Switch", state="attached", timeout=60000)
            except Exception:
                statuses = [r.status for r in documents]
                raise RuntimeError(f"listing never loaded; document statuses {statuses} (likely Akamai bot block)")
            body = documents[-1].body()
        finally:
            browser.close()
    if b"Backup Switch for Powerwall" not in body:
        raise RuntimeError("final response is missing the Powerwall listing")
    path = out_dir / "tesla.html"
    path.write_bytes(body)
    return [path]


FETCHERS = {"connectder": fetch_connectder, "enphase": fetch_enphase, "tesla": fetch_tesla}


NOTHING_DUE = 3


def fetched_this_month(today):
    """Sources already fetched successfully in any snapshot this month."""
    done = set()
    for status_file in RAW_DIR.glob(f"{today[:7]}-??/fetch_status.json"):
        done.update(json.loads(status_file.read_text())["ok"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catch-up", action="store_true",
                        help="only fetch sources not yet fetched successfully this month")
    args = parser.parse_args()

    snapshot_date = date.today().isoformat()
    sources = list(FETCHERS)
    if args.catch_up:
        done = fetched_this_month(snapshot_date)
        sources = [s for s in FETCHERS if s not in done]
        if not sources:
            print(f"{snapshot_date}: all sources already fetched this month; nothing to do")
            return NOTHING_DUE

    setup_logging()
    out_dir = RAW_DIR / snapshot_date
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("scrape start  snapshot=%s  sources=%s", snapshot_date, ", ".join(sources))

    failed = []
    for name in sources:
        fetch = FETCHERS[name]
        try:
            paths = fetch(out_dir)
            sizes = ", ".join(f"{p.name} ({p.stat().st_size:,} B)" for p in paths)
            log.info("OK      %-10s %s", name, sizes)
        except Exception as e:
            failed.append(name)
            log.error("FAILED  %-10s %s: %s", name, type(e).__name__, e)

    # Merge with an earlier run from the same day so its successes are kept.
    status_file = out_dir / "fetch_status.json"
    ok = [n for n in sources if n not in failed]
    if status_file.exists():
        ok = sorted(set(ok) | (set(json.loads(status_file.read_text())["ok"]) - set(failed)))
    status_file.write_text(json.dumps({"snapshot_date": snapshot_date, "ok": ok, "failed": failed}, indent=1))
    log.info("scrape done   %d/%d sources fetched%s", len(sources) - len(failed), len(sources),
             f"; FAILED: {', '.join(failed)}" if failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

#!/bin/zsh
# Scheduled entry point (called daily by launchd; see README "MSA utility
# approval tracking"). Fetches whatever is still due this month, then rebuilds
# msa_master.csv. Does nothing once the month's snapshot is complete.

cd "$(dirname "$0")/.." || exit 1
POETRY=/opt/homebrew/bin/poetry

"$POETRY" run python msa_tracking/msa_scrape.py --catch-up
scrape_status=$?
[ $scrape_status -eq 3 ] && exit 0  # nothing due this month

"$POETRY" run python msa_tracking/processing.py
process_status=$?

[ $scrape_status -ne 0 ] && exit $scrape_status
exit $process_status

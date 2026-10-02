#!/bin/zsh
# Scheduled entry point, run daily by launchd (see README "MSA utility approval
# tracking"). Runs the six notebooks in order with papermill, but only while
# this month's MSA snapshot is incomplete; once it's complete, it does nothing
# until next month. --force runs everything regardless.
#
# Executed copies of the notebooks are saved in runs/ (git-ignored).

cd "$(dirname "$0")" || exit 1
POETRY=/opt/homebrew/bin/poetry

run() {
  "$POETRY" run papermill "$1.ipynb" "runs/$1-output.ipynb" --cwd . --log-level WARNING "${@:2}"
}

due=$("$POETRY" run python common.py)
if [[ -z "$due" && "$1" != "--force" ]]; then
  exit 0  # this month's snapshot is already complete
fi

mkdir -p runs
run_status=0
# Keep going after a failure so later steps still use whatever was fetched;
# the exit status reports that something failed (details in run_log.txt).
run 1_pull_msa -p catch_up True || run_status=1
run 2_pull_eia || run_status=1
run 3_pull_census || run_status=1
run 4_process_msa || run_status=1
run 5_process_eia || run_status=1
run 6_join_outputs || run_status=1
exit $run_status

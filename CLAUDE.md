# CLAUDE.md — basic-solar-statistics

README.md is the single description of this repo (architecture, data sources,
outputs, how to run). Don't duplicate it here. Update README.md whenever you
change structure, inputs, or outputs.

## Rules for this repo
- `output_csvs/state_stats.csv` feeds funder and legislative materials. Any change
  that could alter a value or column in it needs a plan first, and `poetry run pytest`
  afterward.
- Run the pipeline with `-p UPLOAD False` unless I ask for an upload. Uploading
  overwrites the shared Big Numbers Drive sheets.
- Where code goes: `modules/` only holds code used by the Big Numbers pipeline
  (or about to be, like `storage_attachment_rate.py`). A standalone project gets its
  own top-level folder. One-off analyses go in `ad_hocs/`.
- Files in `data/` are committed snapshots. Replacing one (e.g. a new Tracking the
  Sun or Solar TRACE export) is a data update: plan it, and expect golden tests to
  fail. Report which values moved and by how much before updating the golden file.
- Raw Ohm Analytics exports (`data/ohm_analytics_*.csv`) stay local and are
  gitignored. The aggregated Ohm CSVs already in `data/` are fine to keep.
- Legacy but intentionally kept: `modules/median_interconnection_timelines.py` and
  `data/solartrace_ix.csv` still feed two Drive sheets.

"""
Manifest generator for the Big Numbers Database.

Collects metadata about every DataFrame uploaded to Google Drive and produces
a manifest.json that serves as a machine-readable data dictionary. This file
powers downstream tools (LLM-based search, web apps, etc.) that need to know
what data is available, where it lives, and what the columns mean.

Usage:
    from manifest import ManifestBuilder

    mb = ManifestBuilder()

    # After each upload_df_to_drive call, register the result:
    file_id = upload_df_to_drive(df, folder_id, "Sheet Name")
    mb.register(df, file_id, folder="1. Solar", sheet_name="Sheet Name",
                description="Human-readable description of this dataset",
                source="LBNL Tracking the Sun")

    # At the end of the pipeline, save + upload:
    mb.save("manifest.json")
    mb.upload_to_drive(service, ROOT_ID)
"""

import json
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


# ── Human-readable column descriptions ──────────────────────────────────────
# Maps column names (as they appear in the DataFrames) to plain-English
# descriptions. The manifest generator looks up each column here. If a column
# isn't listed, it falls back to the raw column name.

COLUMN_DESCRIPTIONS = {
    # Identifiers
    "state":           "U.S. state (full name)",
    "state_abbr":      "U.S. state (two-letter abbreviation)",
    "geoid":           "Census GEOID for the jurisdiction (AHJ)",
    "ahj_geoid":       "Census GEOID for the jurisdiction (AHJ)",
    "name":            "Jurisdiction / AHJ name",
    "source_id":       "Internal source identifier",
    "data_source":     "Name and year of the data source (e.g. 'Ohm Analytics, Reporting Year-2025')",
    "population":      "Population of the jurisdiction",
    "year":            "Year the data applies to",
    "distribution_year": "Year the timeline distribution was computed",

    # Solar capacity & installations
    "technology":      "Technology type (solar, storage, solar+storage)",
    "sector":          "Market sector (Residential, Commercial, Industrial, Transportation)",

    # Solar costs
    "install_year":    "Year of installation",
    "median_cost_per_kw_nominal": "Median installed cost per kW in nominal dollars",
    "median_cost_per_kw_real":    "Median installed cost per kW in inflation-adjusted (real) dollars",
    "num_systems":     "Number of systems in the sample",

    # Permitting
    "median_permitting_timeline_days":      "Median number of days to obtain a permit",
    "median_permitting_timeline_all_years": "Median permitting timeline in days (all years combined)",
    "num_records":     "Number of data records used to compute the statistic",
    "pct_cancelled":   "Fraction of applications that were cancelled (0 to 1)",
    "num_applications":"Total number of permit applications",
    "num_finaled":     "Number of applications that were finaled/completed",

    # Permitting timeline distribution
    "pctile_0":   "0th percentile (minimum) of permitting timeline in days",
    "pctile_1":   "1st percentile of permitting timeline in days",
    "pctile_10":  "10th percentile of permitting timeline in days",
    "pctile_25":  "25th percentile of permitting timeline in days",
    "pctile_50":  "50th percentile (median) of permitting timeline in days",
    "pctlile_50": "50th percentile (median) of permitting timeline in days [NOTE: typo in source — should be pctile_50]",
    "pctile_75":  "75th percentile of permitting timeline in days",
    "pctile_90":  "90th percentile of permitting timeline in days",
    "pctile_95":  "95th percentile of permitting timeline in days",

    # Permitting fees
    "median_permit_cost": "Median permit fee in dollars",

    # Inspection
    "median_inspection_timeline_days": "Median number of days for inspection",

    # Interconnection
    "utility":         "Electric utility name",
    "utility_name":    "Electric utility name",
    "utility_number":  "EIA utility identification number",
    "eia_id":          "EIA utility identification number",
    "customers":       "Number of utility customers",
    "solar_interconnection_fee": "Solar interconnection fee in dollars",
    "accurate_as_of":  "Date the interconnection fee was last verified",

    # Electricity rates
    "price_per_kwh":   "Average electricity price in dollars per kWh",
    "yoy_pct_change":  "Year-over-year percentage change in price",

    # Solar potential
    "solar_potential_twh": "Technical solar generation potential in terawatt-hours",

    # Solar jobs
    "jobs":            "Total solar jobs in the state",
    "installation_and_project_development": "Jobs in installation and project development",
    "manufacturing":   "Jobs in solar manufacturing",
    "wholesale_trade_and_distribution":     "Jobs in wholesale trade and distribution",
    "operations_and_maintenance":           "Jobs in operations and maintenance",

    # Bill savings
    "weighted_avg_bill_without_pv_year1": "Weighted average annual electricity bill without solar (Year 1), in dollars",
    "weighted_avg_bill_with_pv_year1":    "Weighted average annual electricity bill with solar (Year 1), in dollars",
    "weighted_avg_savings_year1":         "Weighted average annual savings from solar (Year 1), in dollars",
    "pct_savings_year1":                  "Percentage of bill saved with solar in Year 1 (0 to 1)",
    "lifetime_savings_weighted":          "Weighted average lifetime savings from solar, in dollars",

    # Solar eligibility
    "solar_eligible_count":   "Estimated number of solar-eligible households",
    "solar_eligible_pct":     "Percentage of households that are solar-eligible",
    "total_households":       "Total number of households",
}


def _sheet_url(file_id: str) -> str:
    """Build a Google Sheets URL from a file ID."""
    return f"https://docs.google.com/spreadsheets/d/{file_id}"


def _describe_columns(df: pd.DataFrame) -> list:
    """Build a list of column metadata dicts for each column in the DataFrame."""
    cols = []
    for col in df.columns:
        info = {
            "name": col,
            "description": COLUMN_DESCRIPTIONS.get(col, col),
            "dtype": str(df[col].dtype),
        }
        # Add sample values for non-numeric columns (helps LLMs understand the data)
        if df[col].dtype == "object":
            unique_vals = df[col].dropna().unique()[:5].tolist()
            info["sample_values"] = unique_vals
        else:
            # For numeric columns, show range
            info["min"] = None if df[col].dropna().empty else float(df[col].min())
            info["max"] = None if df[col].dropna().empty else float(df[col].max())
        cols.append(info)
    return cols


class ManifestBuilder:
    """
    Accumulates metadata about each uploaded sheet and produces manifest.json.

    Typical usage in the notebook:

        mb = ManifestBuilder()

        file_id = upload_df_to_drive(df, folder_id, "Sheet Name")
        mb.register(df, file_id, "1. Solar", "Sheet Name",
                    description="...", source="EIA 861M")

        # ... more uploads ...

        mb.save("manifest.json")                       # local copy
        mb.upload_to_drive(service, ROOT_ID)            # to Google Drive
    """

    def __init__(self):
        self.sheets = []
        self.generated_at = None

    def register(
        self,
        df: pd.DataFrame,
        file_id: str,
        folder: str,
        sheet_name: str,
        description: str,
        source: str,
        granularity: str = "state",
        notes: Optional[str] = None,
    ):
        """
        Register a DataFrame that was just uploaded to Google Drive.

        Args:
            df:           The DataFrame that was uploaded.
            file_id:      Google Drive file ID returned by upload_df_to_drive().
            folder:       Which category folder (e.g. "1. Solar").
            sheet_name:   The name of the Google Sheet.
            description:  One-sentence plain-English description of the dataset.
            source:       Data source (e.g. "LBNL Tracking the Sun", "EIA 861M").
            granularity:  Level of geographic detail: "state", "ahj", "utility".
            notes:        Optional extra context.
        """
        entry = {
            "sheet_name": sheet_name,
            "folder": folder,
            "description": description,
            "source": source,
            "granularity": granularity,
            "google_sheet_id": file_id,
            "google_sheet_url": _sheet_url(file_id),
            "row_count": len(df),
            "columns": _describe_columns(df),
        }
        if notes:
            entry["notes"] = notes
        self.sheets.append(entry)

    def build(self) -> dict:
        """Build the full manifest dict."""
        self.generated_at = datetime.now(timezone.utc).isoformat()
        return {
            "name": "Big Numbers Database",
            "description": (
                "A collection of regularly updated solar energy statistics for "
                "U.S. states and jurisdictions, maintained by Permit Power. "
                "Includes data on solar costs, capacity, permitting timelines, "
                "inspection timelines, interconnection, electricity rates, "
                "jobs, bill savings, and more."
            ),
            "generated_at": self.generated_at,
            "total_sheets": len(self.sheets),
            "categories": sorted(set(s["folder"] for s in self.sheets)),
            "sheets": self.sheets,
        }

    def save(self, path: str = "manifest.json"):
        """Write manifest.json to a local file."""
        manifest = self.build()
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"Manifest saved to {path} ({len(self.sheets)} sheets)")
        return path

    def upload_to_drive(self, service, root_folder_id: str):
        """
        Upload manifest.json to the root of the Big Numbers Database folder.
        Overwrites if it already exists.
        """
        import io
        from googleapiclient.http import MediaIoBaseUpload

        manifest = self.build()
        content = json.dumps(manifest, indent=2, default=str).encode("utf-8")
        media = MediaIoBaseUpload(
            io.BytesIO(content),
            mimetype="application/json",
            resumable=False,
        )

        # Check if manifest.json already exists
        query = (
            f"name='manifest.json' and '{root_folder_id}' in parents "
            "and trashed=false"
        )
        result = service.files().list(
            q=query,
            spaces="drive",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="files(id)",
        ).execute()

        existing = result.get("files", [])
        if existing:
            file_id = existing[0]["id"]
            service.files().update(
                fileId=file_id,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            ).execute()
            print(f"Manifest updated on Drive (file ID: {file_id})")
        else:
            metadata = {
                "name": "manifest.json",
                "parents": [root_folder_id],
            }
            created = service.files().create(
                body=metadata,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            ).execute()
            print(f"Manifest created on Drive (file ID: {created['id']})")
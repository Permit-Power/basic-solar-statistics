"""
Utilities for uploading pandas DataFrames to Google Drive as native Google Sheets
and managing folder structures within shared drives. This module is designed for
use with Application Default Credentials (ADC), including Workload Identity
Federation (WIF) in CI environments or user ADC locally. All file and folder
operations are fully compatible with Google Shared Drives.

Key features:
- Upload DataFrames directly as native Google Sheets.
- Overwrite existing files with the same name.
- Automatically create nested folder hierarchies.
- Fully Shared Drive–safe: uses supportsAllDrives and includeItemsFromAllDrives.
"""

import io
from typing import List

import pandas as pd
from google.auth import default
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload


def get_drive_service():
    """
    Initializes and returns a Google Drive API service client using
    Application Default Credentials.

    Returns:
        googleapiclient.discovery.Resource: Authenticated Drive API client.
    """
    creds, _ = default()
    return build("drive", "v3", credentials=creds)


def find_or_create_folder(service, parent_id: str, name: str):
    """
    Finds an existing folder with the given name under parent_id or creates it
    if it does not exist. Fully compatible with Shared Drives.

    Args:
        service: Drive API service instance.
        parent_id (str): ID of the parent folder.
        name (str): Name of the folder to find or create.

    Returns:
        str: Folder ID of the existing or newly created folder.
    """
    query = (
        "mimeType='application/vnd.google-apps.folder' "
        f"and name='{name}' and '{parent_id}' in parents and trashed=false"
    )

    result = service.files().list(
        q=query,
        spaces="drive",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        fields="files(id)"
    ).execute()

    files = result.get("files", [])
    if files:
        return files[0]["id"]

    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }

    created = service.files().create(
        body=metadata,
        fields="id",
        supportsAllDrives=True
    ).execute()

    return created["id"]


def ensure_path(service, root_folder_id: str, path_components: List[str]):
    """
    Ensures that a nested folder hierarchy exists under a given root folder ID.
    Any missing folders are created automatically.

    Args:
        service: Drive API service instance.
        root_folder_id (str): Starting folder ID.
        path_components (List[str]): Ordered list of folder names
            representing the desired path.

    Returns:
        str: ID of the deepest folder in the created or verified path.
    """
    current_parent = root_folder_id
    for part in path_components:
        current_parent = find_or_create_folder(service, current_parent, part)
    return current_parent


def upload_df_to_drive(df: pd.DataFrame, parent_folder_id: str, filename: str):
    """
    Uploads a pandas DataFrame as a native Google Sheet into a specified folder.
    If a file with the same name already exists in that folder, it is overwritten.

    Args:
        df (pd.DataFrame): DataFrame to upload.
        parent_folder_id (str): ID of the destination folder.
        filename (str): Logical filename (without extension enforcement).
            The uploaded file will become a Google Sheet with this name.

    Returns:
        str: File ID of the uploaded or updated Google Sheet.
    """
    service = get_drive_service()

    # Convert DataFrame to CSV as media upload content
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    media = MediaIoBaseUpload(buffer, mimetype="text/csv", resumable=False)

    # File metadata for Google Sheets
    sheet_name = filename.replace(".csv", "").replace(".xlsx", "")
    metadata = {
        "name": sheet_name,
        "mimeType": "application/vnd.google-apps.spreadsheet",
        "parents": [parent_folder_id],
    }

    # Check if file already exists
    query = (
        f"name='{sheet_name}' and '{parent_folder_id}' in parents and trashed=false"
    )

    result = service.files().list(
        q=query,
        spaces="drive",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        fields="files(id)"
    ).execute()

    existing = result.get("files", [])

    if existing:
        file_id = existing[0]["id"]
        service.files().update(
            fileId=file_id,
            media_body=media,
            fields="id",
            supportsAllDrives=True
        ).execute()
        return file_id

    created = service.files().create(
        body=metadata,
        media_body=media,
        fields="id",
        supportsAllDrives=True
    ).execute()

    return created["id"]

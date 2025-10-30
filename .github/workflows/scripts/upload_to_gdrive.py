import argparse
import os
import tempfile
import logging
from datetime import datetime
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)


def get_or_create_folder(service, parent_id: str, folder_name: str) -> str:
    query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(
        q=query, 
        fields="files(id, name)", 
        spaces="drive",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    files = results.get("files", [])
    
    if files:
        logging.info(f"Found existing folder: {folder_name}")
        return files[0]["id"]
    
    logging.info(f"Creating folder: {folder_name}")
    folder_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(
        body=folder_metadata, 
        fields="id",
        supportsAllDrives=True
    ).execute()
    return folder["id"]


def archive_existing_zip(service, folder_id: str, zip_name: str, version: str) -> None:
    query = f"name='{zip_name}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query, 
        fields="files(id, name)", 
        spaces="drive",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    files = results.get("files", [])
    
    if not files:
        logging.info(f"No existing {zip_name} to archive.")
        return
    
    file_id = files[0]["id"]
    archive_folder_id = get_or_create_folder(service, folder_id, "archive")
    date_str = datetime.now().strftime("%b %d %Y").lower()
    date_folder_id = get_or_create_folder(service, archive_folder_id, date_str)
    version_folder_id = get_or_create_folder(service, date_folder_id, version)
    
    service.files().update(
        fileId=file_id,
        addParents=version_folder_id,
        removeParents=folder_id,
        fields="id, parents",
        supportsAllDrives=True
    ).execute()
    logging.info(f"Archived {zip_name} to archive/{date_str}/{version}/")


def upload_zip(platform: str, version: str) -> None:
    zip_name = f"corerl-{platform}-executables.zip"
    zip_path = Path("dist") / zip_name

    if not zip_path.exists():
        logging.error(f"{zip_path} not found.")
        return

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as creds_file:
        creds_file.write(os.environ["GDRIVE_SERVICE_ACCOUNT_KEY"])
        creds_path = creds_file.name

    try:
        scopes = ["https://www.googleapis.com/auth/drive"]
        creds = service_account.Credentials.from_service_account_file(creds_path, scopes=scopes)
        service = build("drive", "v3", credentials=creds)

        folder_id = os.environ["GDRIVE_EXECUTABLES_FOLDER_ID"]

        archive_existing_zip(service, folder_id, zip_name, version)

        logging.info(f"Uploading {zip_path.name} to Google Drive...")

        file_metadata = {
            "name": zip_path.name, 
            "parents": [folder_id]
        }
        media = MediaFileUpload(str(zip_path), mimetype="application/zip", resumable=True)
        request = service.files().create(
            body=file_metadata, 
            media_body=media, 
            fields="id,name,webViewLink",
            supportsAllDrives=True
        )

        # Progress bar for upload
        response = None
        pbar = tqdm(total=zip_path.stat().st_size, unit="B", unit_scale=True, desc="Uploading")
        while response is None:
            status, response = request.next_chunk()
            if status:
                pbar.n = int(status.resumable_progress)
                pbar.refresh()
        pbar.n = zip_path.stat().st_size
        pbar.close()

        logging.info(f"Uploaded {response['name']} - {response['webViewLink']}")
    finally:
        Path(creds_path).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload zipped artifacts to Google Drive")
    parser.add_argument("--platform", required=True, choices=["linux", "windows"], help="Platform identifier")
    parser.add_argument("--version", required=True, help="Release version (e.g., v0.155.0)")
    args = parser.parse_args()
    upload_zip(args.platform, args.version)
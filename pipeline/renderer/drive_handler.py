from __future__ import annotations

import json
import os
import re

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

_SCOPES = ["https://www.googleapis.com/auth/drive"]

_AUDIO_MIMES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/aac", "audio/mp4",
}
_VIDEO_MIMES = {
    "video/mp4", "video/quicktime", "video/x-msvideo", "video/webm",
    "video/3gpp",
}


def build_service(service_account_json: str):
    """
    Build a Google Drive API v3 service from a service account JSON string.
    """
    info = json.loads(service_account_json)
    creds = Credentials.from_service_account_info(info, scopes=_SCOPES)
    return build("drive", "v3", credentials=creds)


def extract_folder_id(drive_url: str) -> str:
    """
    Parse a Google Drive folder sharing URL and return the folder ID.

    Raises:
        ValueError: if no folder ID can be parsed.
    """
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", drive_url)
    if not match:
        raise ValueError(f"Could not extract folder ID from URL: {drive_url}")
    return match.group(1)


def list_drive_assets(service, folder_id: str) -> dict:
    """
    List files in a Drive folder and split into audio + video.

    Returns:
        {
            "audio":  {"id": str, "name": str},
            "videos": [{"id": str, "name": str}, ...]
        }

    Raises:
        ValueError: if no audio or no video files are found.
    """
    resp = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        fields="files(id, name, mimeType)",
        pageSize=100,
    ).execute()

    audio = None
    videos: list[dict] = []

    for f in resp.get("files", []):
        mime = f.get("mimeType", "")
        name = f["name"]
        if mime in _AUDIO_MIMES or _ends_with(name, (".mp3", ".wav", ".aac", ".m4a")):
            if audio is None:
                audio = {"id": f["id"], "name": name}
        elif mime in _VIDEO_MIMES or _ends_with(name, (".mp4", ".mov", ".avi", ".webm")):
            videos.append({"id": f["id"], "name": name})

    if audio is None:
        raise ValueError("No audio file (MP3/WAV) found in the Drive folder.")
    if not videos:
        raise ValueError("No video clips (MP4/MOV) found in the Drive folder.")

    return {"audio": audio, "videos": videos}


def download_file(service, file_id: str, destination_path: str) -> str:
    """
    Download a Drive file to a local path.

    Returns:
        destination_path
    """
    os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    with open(destination_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return destination_path


def upload_file(service, local_path: str, folder_id: str) -> str:
    """
    Upload a local file to a Google Drive folder.

    Returns:
        The Drive file ID of the newly uploaded file.
    """
    metadata = {"name": os.path.basename(local_path), "parents": [folder_id]}
    media = MediaFileUpload(local_path, resumable=True)
    result = service.files().create(
        body=metadata, media_body=media, fields="id"
    ).execute()
    return result.get("id", "")


def _ends_with(name: str, exts: tuple) -> bool:
    return name.lower().endswith(exts)

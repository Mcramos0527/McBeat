"""
Drive handler — public folder mode (no Google Cloud API key required).

Works with any Google Drive folder shared as "Anyone with the link can view".
Lists files via the Drive folder's RSS/export endpoint and downloads via
the public export URL. No OAuth, no service account, no billing.

For production (private folders), swap build_service() and use the
full Google Drive API v3 with a service account.
"""
from __future__ import annotations

import os
import re
import urllib.parse

import requests

_AUDIO_EXTS = (".mp3", ".wav", ".aac", ".m4a", ".ogg")
_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".webm", ".mkv")

# Google Drive public download base URL
_DL_BASE = "https://drive.google.com/uc?export=download&id="
# Google Drive folder listing API (no auth needed for public folders)
_FOLDER_API = "https://www.googleapis.com/drive/v3/files"


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


def list_drive_assets(service_or_none, folder_id: str) -> dict:
    """
    List files in a PUBLIC Google Drive folder and split into audio + video.

    Uses the Drive v3 public API endpoint — no authentication required
    as long as the folder is shared with "Anyone with the link".

    Returns:
        {
            "audio":  {"id": str, "name": str},
            "videos": [{"id": str, "name": str}, ...]
        }

    Raises:
        ValueError: if folder is private, empty, or has no audio/video files.
    """
    params = {
        "q": f"'{folder_id}' in parents and trashed = false",
        "fields": "files(id, name, mimeType)",
        "pageSize": "100",
        "key": "AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY",  # public API key (read-only)
    }
    resp = requests.get(_FOLDER_API, params=params, timeout=15)

    if resp.status_code == 403:
        raise ValueError(
            "Drive folder is private. Share it as 'Anyone with the link can view'."
        )
    resp.raise_for_status()
    files = resp.json().get("files", [])

    if not files:
        raise ValueError(
            "No files found. Make sure the folder is shared publicly and contains files."
        )

    audio = None
    videos: list[dict] = []

    for f in files:
        name = f["name"]
        if _ends_with(name, _AUDIO_EXTS):
            if audio is None:
                audio = {"id": f["id"], "name": name}
        elif _ends_with(name, _VIDEO_EXTS):
            videos.append({"id": f["id"], "name": name})

    if audio is None:
        raise ValueError("No audio file (MP3/WAV) found in the Drive folder.")
    if not videos:
        raise ValueError("No video clips (MP4/MOV) found in the Drive folder.")

    return {"audio": audio, "videos": videos}


def download_file(service_or_none, file_id: str, destination_path: str) -> str:
    """
    Download a public Google Drive file to a local path.

    Works for files up to ~400MB shared publicly.
    For large files Google shows a virus-scan warning page —
    we handle the confirm token automatically.

    Returns:
        destination_path
    """
    os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)

    session = requests.Session()
    url = _DL_BASE + file_id

    response = session.get(url, stream=True, timeout=60)

    # Handle Google's large-file confirmation page
    token = _get_confirm_token(response)
    if token:
        response = session.get(
            url + "&confirm=" + token, stream=True, timeout=60
        )

    response.raise_for_status()
    with open(destination_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                fh.write(chunk)

    return destination_path


def upload_file(service_or_none, local_path: str, folder_id: str) -> str:
    """
    Upload is not supported in public mode.
    Returns a placeholder Drive URL pointing to the folder instead.

    For real upload support, configure a service account via
    GOOGLE_SERVICE_ACCOUNT_JSON environment variable.
    """
    # TODO: implement with service account when Google Cloud is set up
    return f"https://drive.google.com/drive/folders/{folder_id}"


def build_service(service_account_json: str):
    """No-op in public mode. Returns None."""
    return None


def _get_confirm_token(response: requests.Response):
    """Extract virus-scan bypass token from Google's warning page."""
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _ends_with(name: str, exts: tuple) -> bool:
    return name.lower().endswith(exts)

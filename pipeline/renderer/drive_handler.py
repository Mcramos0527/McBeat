"""
Drive handler — uses gdown to download public Google Drive folders.
No API key, no OAuth, no billing. Zero config.

Requirements:
- The folder must be shared as "Anyone with the link can view"
- pip install gdown
"""
from __future__ import annotations

import os
import re

import gdown

_AUDIO_EXTS = (".mp3", ".wav", ".aac", ".m4a", ".ogg")
_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".webm", ".mkv")


def extract_folder_id(drive_url: str) -> str:
    """Extract folder ID from a Google Drive sharing URL."""
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", drive_url)
    if not match:
        raise ValueError(f"Could not extract folder ID from URL: {drive_url}")
    return match.group(1)


def download_folder(drive_url: str, dest_dir: str) -> dict:
    """
    Download all files from a public Google Drive folder using gdown.
    No API key required — folder must be shared publicly.

    Returns:
        {
            "audio":  str,         # path to audio file
            "videos": [str, ...]   # paths to video clips
        }
    """
    os.makedirs(dest_dir, exist_ok=True)
    folder_id = extract_folder_id(drive_url)

    try:
        gdown.download_folder(
            id=folder_id,
            output=dest_dir,
            quiet=False,
            use_cookies=False,
        )
    except Exception as e:
        raise ValueError(
            f"Could not download Drive folder. "
            f"Make sure it's shared as 'Anyone with the link can view'. "
            f"Error: {e}"
        )

    # Scan downloaded files
    audio = None
    videos = []
    for fname in os.listdir(dest_dir):
        path = os.path.join(dest_dir, fname)
        if not os.path.isfile(path):
            continue
        if fname.lower().endswith(_AUDIO_EXTS):
            if audio is None:
                audio = path
        elif fname.lower().endswith(_VIDEO_EXTS):
            videos.append(path)

    if audio is None:
        raise ValueError("No audio file (MP3/WAV) found in the Drive folder.")
    if not videos:
        raise ValueError("No video clips (MP4/MOV) found in the Drive folder.")

    return {"audio": audio, "videos": videos}


# ── Legacy stubs (kept for compatibility with existing api.py) ───────────────

def build_service(service_account_json: str):
    return None


def list_drive_assets(service, folder_id: str) -> dict:
    raise NotImplementedError("Use download_folder() instead.")


def download_file(service, file_id: str, destination_path: str) -> str:
    gdown.download(id=file_id, output=destination_path, quiet=False)
    return destination_path


def upload_file(service, local_path: str, folder_id: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}"

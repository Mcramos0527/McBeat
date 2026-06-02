import os
import pytest
from unittest.mock import MagicMock, patch
from renderer.drive_handler import (
    extract_folder_id,
    list_drive_assets,
    download_file,
    upload_file,
)


def test_extract_standard_url():
    url = "https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"
    assert extract_folder_id(url) == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"


def test_extract_url_with_query_string():
    url = "https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs?usp=sharing"
    assert extract_folder_id(url) == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs"


def test_extract_invalid_url_raises():
    with pytest.raises(ValueError, match="Could not extract"):
        extract_folder_id("https://google.com/not-a-folder")


def test_list_assets_separates_audio_and_video():
    svc = MagicMock()
    svc.files().list().execute.return_value = {
        "files": [
            {"id": "a1", "name": "track.mp3",  "mimeType": "audio/mpeg"},
            {"id": "v1", "name": "clip1.mp4",  "mimeType": "video/mp4"},
            {"id": "v2", "name": "clip2.mov",  "mimeType": "video/quicktime"},
            {"id": "x1", "name": "notes.txt",  "mimeType": "text/plain"},
        ]
    }
    result = list_drive_assets(svc, "folder_id")
    assert result["audio"]["name"] == "track.mp3"
    assert len(result["videos"]) == 2


def test_list_assets_raises_if_no_audio():
    svc = MagicMock()
    svc.files().list().execute.return_value = {
        "files": [{"id": "v1", "name": "clip.mp4", "mimeType": "video/mp4"}]
    }
    with pytest.raises(ValueError, match="No audio"):
        list_drive_assets(svc, "folder_id")


def test_list_assets_raises_if_no_video():
    svc = MagicMock()
    svc.files().list().execute.return_value = {
        "files": [{"id": "a1", "name": "track.mp3", "mimeType": "audio/mpeg"}]
    }
    with pytest.raises(ValueError, match="No video"):
        list_drive_assets(svc, "folder_id")


@patch("renderer.drive_handler.MediaIoBaseDownload")
def test_download_file_calls_next_chunk(mock_cls, tmp_dir):
    svc = MagicMock()
    svc.files().get_media.return_value = MagicMock()
    dl = MagicMock()
    mock_cls.return_value = dl
    dl.next_chunk.side_effect = [
        (MagicMock(progress=lambda: 0.5), False),
        (MagicMock(progress=lambda: 1.0), True),
    ]
    dest = os.path.join(tmp_dir, "audio.mp3")
    download_file(svc, "file123", dest)
    assert dl.next_chunk.call_count == 2

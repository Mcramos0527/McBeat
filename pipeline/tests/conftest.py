import os
import tempfile

import cv2
import numpy as np
import pytest
import soundfile as sf


def make_synthetic_audio(
    bpm: float = 120.0, duration: float = 10.0, sr: int = 22050
) -> tuple[np.ndarray, int]:
    y = np.zeros(int(sr * duration), dtype=np.float32)
    beat_period = 60.0 / bpm
    click_len = int(sr * 0.02)
    for pos in np.arange(0, duration, beat_period):
        idx = int(pos * sr)
        if idx + click_len < len(y):
            y[idx : idx + click_len] = (
                np.sin(np.linspace(0, np.pi * 4, click_len)) * 0.9
            )
    return y, sr


def make_synthetic_video(
    path: str,
    frames: int = 60,
    fps: int = 30,
    width: int = 320,
    height: int = 240,
    motion: bool = True,
) -> str:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(frames):
        frame = np.full((height, width, 3), 100, dtype=np.uint8)
        if motion:
            offset = (i * 5) % width
            end = min(offset + 40, width)
            frame[:, offset:end, :] = 200
        out.write(frame)
    out.release()
    return path


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def audio_120bpm(tmp_dir):
    path = os.path.join(tmp_dir, "track_120bpm.wav")
    y, sr = make_synthetic_audio(bpm=120.0, duration=10.0)
    sf.write(path, y, sr)
    return path


@pytest.fixture
def audio_90bpm(tmp_dir):
    path = os.path.join(tmp_dir, "track_90bpm.wav")
    y, sr = make_synthetic_audio(bpm=90.0, duration=10.0)
    sf.write(path, y, sr)
    return path


@pytest.fixture
def high_motion_clip(tmp_dir):
    path = os.path.join(tmp_dir, "clip_high.mp4")
    return make_synthetic_video(path, motion=True)


@pytest.fixture
def low_motion_clip(tmp_dir):
    path = os.path.join(tmp_dir, "clip_low.mp4")
    return make_synthetic_video(path, motion=False)


@pytest.fixture
def sample_beat_map():
    return {
        "bpm": 120.0,
        "beats": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "peaks": [1.0, 3.0],
        "drops": [2.0],
        "segments": [{"start": 0.0, "end": 4.0, "label": "intro"}],
    }


@pytest.fixture
def sample_clip_scores():
    return [
        {
            "clip_id": "clip_a",
            "clip_path": "/tmp/clip_a.mp4",
            "motion_score": 0.85,
            "brightness": 0.6,
            "contrast": 0.7,
            "duration_ms": 4000,
        },
        {
            "clip_id": "clip_b",
            "clip_path": "/tmp/clip_b.mp4",
            "motion_score": 0.25,
            "brightness": 0.7,
            "contrast": 0.5,
            "duration_ms": 3000,
        },
        {
            "clip_id": "clip_c",
            "clip_path": "/tmp/clip_c.mp4",
            "motion_score": 0.55,
            "brightness": 0.5,
            "contrast": 0.6,
            "duration_ms": 5000,
        },
    ]

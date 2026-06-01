# McBeat Pipeline Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete McBeat beat-sync pipeline — raw audio + video clips → rendered export — using pure Python open-source libraries, fully testable locally before any cloud deployment.

**Architecture:** Five sequential stages run inside a single process: (1) beat_detector extracts BPM/beats/peaks/drops via Librosa, (2) energy_mapper enriches with segment-level energy, (3) visual_scorer evaluates each clip via OpenCV optical flow, (4) beat_matcher builds the ordered timeline in pure Python, (5) ffmpeg_builder cuts, assembles, and encodes the final video. A FastAPI endpoint on Hugging Face Spaces ties them together, pulling inputs from / pushing outputs to the client's Google Drive folder.

**Tech Stack:** Python 3.10, librosa 0.10, soundfile, numpy, opencv-python-headless, ffmpeg (system binary) + ffmpeg-python, google-api-python-client, supabase-py, fastapi, uvicorn, pytest

---

## File Map

```
pipeline/
├── requirements.txt                        CREATE
├── api.py                                  CREATE  — FastAPI POST /process-job
├── audio_analysis/
│   ├── __init__.py                         CREATE  — empty
│   ├── beat_detector.py                    CREATE  — BPM + beat timestamps via Librosa
│   └── energy_mapper.py                    CREATE  — onset strength + segment energy
├── clip_engine/
│   ├── __init__.py                         CREATE  — empty
│   ├── visual_scorer.py                    CREATE  — motion/brightness/contrast per clip
│   └── beat_matcher.py                     CREATE  — timeline builder algorithm
├── renderer/
│   ├── __init__.py                         CREATE  — empty
│   ├── ffmpeg_builder.py                   CREATE  — cut + assemble + encode, 4 profiles
│   └── drive_handler.py                    CREATE  — Drive list/download/upload
└── tests/
    ├── conftest.py                         CREATE  — shared fixtures (synthetic audio/video)
    ├── test_beat_detector.py               CREATE
    ├── test_energy_mapper.py               CREATE
    ├── test_visual_scorer.py               CREATE
    ├── test_beat_matcher.py                CREATE
    ├── test_ffmpeg_builder.py              CREATE
    ├── test_drive_handler.py               CREATE
    ├── test_api.py                         CREATE
    └── test_integration.py                 CREATE  — end-to-end local test

supabase/
└── migrations/
    └── 001_initial_schema.sql              CREATE
```

---

## Task 1: Project Setup + Test Infrastructure

**Files:**
- Create: `pipeline/requirements.txt`
- Create: `pipeline/tests/conftest.py`
- Create: `pipeline/audio_analysis/__init__.py`
- Create: `pipeline/clip_engine/__init__.py`
- Create: `pipeline/renderer/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
# pipeline/requirements.txt
librosa==0.10.1
soundfile==0.12.1
numpy==1.26.4
opencv-python-headless==4.9.0.80
ffmpeg-python==0.2.0
google-api-python-client==2.120.0
google-auth==2.29.0
supabase==2.4.0
fastapi==0.111.0
uvicorn==0.29.0
python-multipart==0.0.9
pytest==8.1.1
pytest-mock==3.14.0
```

- [ ] **Step 2: Install dependencies**

```bash
cd pipeline
pip install -r requirements.txt
```

Expected: All packages install without errors.  
Also verify `ffmpeg` is available as a system binary:
```bash
ffmpeg -version
```
Expected: version string printed. If missing: `brew install ffmpeg` (Mac) or `apt install ffmpeg` (Linux).

- [ ] **Step 3: Create empty `__init__.py` files**

Create these three files, each completely empty:
```
pipeline/audio_analysis/__init__.py
pipeline/clip_engine/__init__.py
pipeline/renderer/__init__.py
```

- [ ] **Step 4: Create conftest.py with shared fixtures**

```python
# pipeline/tests/conftest.py
import os
import tempfile

import cv2
import numpy as np
import pytest
import soundfile as sf


# ── Synthetic audio helpers ─────────────────────────────────────────────────

def make_synthetic_audio(
    bpm: float = 120.0, duration: float = 10.0, sr: int = 22050
) -> tuple[np.ndarray, int]:
    """
    Generate a click track at the given BPM.
    Each click is a 20 ms sine burst — Librosa detects these reliably.
    Returns (audio_array float32, sample_rate).
    """
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


# ── Synthetic video helper ───────────────────────────────────────────────────

def make_synthetic_video(
    path: str,
    frames: int = 60,
    fps: int = 30,
    width: int = 320,
    height: int = 240,
    motion: bool = True,
) -> str:
    """
    Write a synthetic .mp4 to path.
    motion=True  → white stripe shifts 5 px right each frame (high optical flow)
    motion=False → static grey frame (zero optical flow)
    Returns path.
    """
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


# ── Pytest fixtures ──────────────────────────────────────────────────────────

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
```

- [ ] **Step 5: Verify fixtures parse without errors**

```bash
cd pipeline
python -m pytest tests/ --collect-only
```

Expected: `no tests ran` — just confirms conftest.py is valid Python.

- [ ] **Step 6: Commit**

```bash
git add pipeline/requirements.txt \
        pipeline/audio_analysis/__init__.py \
        pipeline/clip_engine/__init__.py \
        pipeline/renderer/__init__.py \
        pipeline/tests/conftest.py
git commit -m "feat: pipeline project setup and shared test fixtures"
```

---

## Task 2: Beat Detector

**Files:**
- Create: `pipeline/audio_analysis/beat_detector.py`
- Create: `pipeline/tests/test_beat_detector.py`

- [ ] **Step 1: Write the failing test**

```python
# pipeline/tests/test_beat_detector.py
import soundfile as sf
import pytest
from audio_analysis.beat_detector import analyze


def test_returns_required_keys(audio_120bpm):
    result = analyze(audio_120bpm)
    for key in ("bpm", "beats", "peaks", "drops", "segments"):
        assert key in result, f"Missing key: {key}"


def test_bpm_within_tolerance_120(audio_120bpm):
    result = analyze(audio_120bpm)
    assert abs(result["bpm"] - 120.0) < 12.0, f"Expected ~120 BPM, got {result['bpm']}"


def test_bpm_within_tolerance_90(audio_90bpm):
    result = analyze(audio_90bpm)
    assert abs(result["bpm"] - 90.0) < 12.0, f"Expected ~90 BPM, got {result['bpm']}"


def test_beats_are_sorted_floats(audio_120bpm):
    beats = analyze(audio_120bpm)["beats"]
    assert len(beats) > 0
    assert all(isinstance(b, float) for b in beats)
    assert beats == sorted(beats)


def test_beats_within_audio_duration(audio_120bpm):
    info = sf.info(audio_120bpm)
    beats = analyze(audio_120bpm)["beats"]
    assert all(b <= info.duration + 0.1 for b in beats)


def test_segments_have_required_keys(audio_120bpm):
    for seg in analyze(audio_120bpm)["segments"]:
        assert "start" in seg and "end" in seg and "label" in seg


def test_segments_cover_duration(audio_120bpm):
    info = sf.info(audio_120bpm)
    segs = analyze(audio_120bpm)["segments"]
    assert segs[0]["start"] == pytest.approx(0.0, abs=1.0)
    assert segs[-1]["end"] == pytest.approx(info.duration, abs=1.0)
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
cd pipeline
python -m pytest tests/test_beat_detector.py -v
```

Expected: `ImportError: No module named 'audio_analysis.beat_detector'`

- [ ] **Step 3: Implement beat_detector.py**

```python
# pipeline/audio_analysis/beat_detector.py
from __future__ import annotations

import librosa
import numpy as np

_SEGMENT_LABELS = ["intro", "verse", "chorus", "bridge", "outro"]


def analyze(audio_path: str) -> dict:
    """
    Analyze an audio file and return a beat_map.

    Returns:
        {
            "bpm":      float,
            "beats":    list[float],   # beat timestamps in seconds
            "peaks":    list[float],   # high-energy moments in seconds
            "drops":    list[float],   # energy-drop moments in seconds
            "segments": list[{"start": float, "end": float, "label": str}]
        }
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # BPM + beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    beats = [float(t) for t in librosa.frames_to_time(beat_frames, sr=sr)]

    # Energy envelope
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    mean_rms = float(np.mean(rms))
    std_rms = float(np.std(rms))

    peak_mask = rms > (mean_rms + 0.8 * std_rms)
    drop_mask = rms < (mean_rms - 0.5 * std_rms)

    peaks = _merge_times(rms_times[peak_mask].tolist(), min_gap=1.0)
    drops = _merge_times(rms_times[drop_mask].tolist(), min_gap=2.0)

    segments = _build_segments(duration, beats)

    return {
        "bpm": bpm,
        "beats": beats,
        "peaks": peaks,
        "drops": drops,
        "segments": segments,
    }


def _merge_times(times: list[float], min_gap: float) -> list[float]:
    """Collapse timestamps closer than min_gap seconds into one."""
    if not times:
        return []
    merged = [times[0]]
    for t in times[1:]:
        if t - merged[-1] >= min_gap:
            merged.append(t)
    return merged


def _build_segments(duration: float, beats: list[float]) -> list[dict]:
    """
    Divide the track into segments of 16 beats each.
    Labels cycle through _SEGMENT_LABELS.
    """
    if not beats:
        return [{"start": 0.0, "end": duration, "label": "intro"}]

    size = 16
    segments = []
    for i in range(0, len(beats), size):
        start = beats[i]
        if i + size < len(beats):
            end = beats[i + size]
        else:
            end = duration
        label = _SEGMENT_LABELS[(i // size) % len(_SEGMENT_LABELS)]
        segments.append({"start": float(start), "end": float(end), "label": label})

    # Guarantee the last segment reaches end of track
    if segments:
        segments[-1]["end"] = duration
    return segments
```

- [ ] **Step 4: Run tests — confirm all pass**

```bash
cd pipeline
python -m pytest tests/test_beat_detector.py -v
```

Expected: **7/7 PASS**

- [ ] **Step 5: Commit**

```bash
git add pipeline/audio_analysis/beat_detector.py \
        pipeline/tests/test_beat_detector.py
git commit -m "feat: beat detector - BPM, beats, peaks, drops, segments via Librosa"
```

---

## Task 3: Energy Mapper

**Files:**
- Create: `pipeline/audio_analysis/energy_mapper.py`
- Create: `pipeline/tests/test_energy_mapper.py`

- [ ] **Step 1: Write the failing test**

```python
# pipeline/tests/test_energy_mapper.py
import soundfile as sf
import pytest
from audio_analysis.energy_mapper import get_onset_strength, segment_by_energy


def test_onset_strength_returns_parallel_lists(audio_120bpm):
    times, strength = get_onset_strength(audio_120bpm)
    assert len(times) > 0
    assert len(times) == len(strength)


def test_onset_strength_all_non_negative(audio_120bpm):
    _, strength = get_onset_strength(audio_120bpm)
    assert all(s >= 0.0 for s in strength)


def test_onset_strength_normalized_max_one(audio_120bpm):
    _, strength = get_onset_strength(audio_120bpm)
    assert max(strength) == pytest.approx(1.0, abs=0.01)


def test_segment_by_energy_count(audio_120bpm):
    segs = segment_by_energy(audio_120bpm, n_segments=4)
    assert len(segs) == 4


def test_segment_by_energy_schema(audio_120bpm):
    for seg in segment_by_energy(audio_120bpm, n_segments=4):
        assert "start" in seg and "end" in seg and "energy" in seg
        assert 0.0 <= seg["energy"] <= 1.0


def test_segments_cover_full_duration(audio_120bpm):
    info = sf.info(audio_120bpm)
    segs = segment_by_energy(audio_120bpm, n_segments=4)
    assert segs[0]["start"] == pytest.approx(0.0, abs=0.1)
    assert segs[-1]["end"] == pytest.approx(info.duration, abs=0.5)
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
cd pipeline
python -m pytest tests/test_energy_mapper.py -v
```

Expected: `ImportError: No module named 'audio_analysis.energy_mapper'`

- [ ] **Step 3: Implement energy_mapper.py**

```python
# pipeline/audio_analysis/energy_mapper.py
from __future__ import annotations

import librosa
import numpy as np


def get_onset_strength(audio_path: str) -> tuple[list[float], list[float]]:
    """
    Compute onset-strength envelope.

    Returns:
        (times, strength) — parallel lists.
        times:    timestamp of each frame in seconds.
        strength: onset strength normalized to [0, 1].
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    hop = 512
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
    max_val = float(np.max(env)) or 1.0
    normalized = (env / max_val).tolist()
    return [float(t) for t in times], normalized


def segment_by_energy(audio_path: str, n_segments: int = 4) -> list[dict]:
    """
    Divide the track into n_segments equal-time windows.
    Each window gets a mean normalized RMS energy score.

    Returns:
        [{"start": float, "end": float, "energy": float}, ...]
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    hop = 512

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    max_rms = float(np.max(rms)) or 1.0
    norm_rms = rms / max_rms

    seg_dur = duration / n_segments
    segments = []
    for i in range(n_segments):
        start = i * seg_dur
        end = (i + 1) * seg_dur if i < n_segments - 1 else duration
        mask = (times >= start) & (times < end)
        energy = float(np.mean(norm_rms[mask])) if mask.any() else 0.0
        segments.append({"start": round(start, 4), "end": round(end, 4), "energy": round(energy, 4)})
    return segments
```

- [ ] **Step 4: Run tests — confirm all pass**

```bash
cd pipeline
python -m pytest tests/test_energy_mapper.py -v
```

Expected: **6/6 PASS**

- [ ] **Step 5: Commit**

```bash
git add pipeline/audio_analysis/energy_mapper.py \
        pipeline/tests/test_energy_mapper.py
git commit -m "feat: energy mapper - onset strength and per-segment RMS energy"
```

---

## Task 4: Visual Scorer

**Files:**
- Create: `pipeline/clip_engine/visual_scorer.py`
- Create: `pipeline/tests/test_visual_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
# pipeline/tests/test_visual_scorer.py
import os
import pytest
from clip_engine.visual_scorer import score_clip, score_clips


def test_score_clip_returns_required_keys(high_motion_clip):
    result = score_clip(high_motion_clip)
    for key in ("clip_id", "clip_path", "motion_score", "brightness", "contrast", "duration_ms"):
        assert key in result, f"Missing key: {key}"


def test_clip_id_is_basename_without_ext(high_motion_clip):
    result = score_clip(high_motion_clip)
    expected = os.path.splitext(os.path.basename(high_motion_clip))[0]
    assert result["clip_id"] == expected


def test_motion_scores_in_range(high_motion_clip, low_motion_clip):
    high = score_clip(high_motion_clip)
    low = score_clip(low_motion_clip)
    assert 0.0 <= high["motion_score"] <= 1.0
    assert 0.0 <= low["motion_score"] <= 1.0


def test_high_motion_greater_than_low_motion(high_motion_clip, low_motion_clip):
    high = score_clip(high_motion_clip)["motion_score"]
    low = score_clip(low_motion_clip)["motion_score"]
    assert high > low, f"Expected high ({high}) > low ({low})"


def test_brightness_in_range(high_motion_clip):
    assert 0.0 <= score_clip(high_motion_clip)["brightness"] <= 1.0


def test_contrast_in_range(high_motion_clip):
    assert 0.0 <= score_clip(high_motion_clip)["contrast"] <= 1.0


def test_duration_ms_positive(high_motion_clip):
    assert score_clip(high_motion_clip)["duration_ms"] > 0


def test_score_clips_processes_multiple(high_motion_clip, low_motion_clip):
    results = score_clips([high_motion_clip, low_motion_clip])
    assert len(results) == 2
    ids = [r["clip_id"] for r in results]
    assert len(set(ids)) == 2  # distinct IDs
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
cd pipeline
python -m pytest tests/test_visual_scorer.py -v
```

Expected: `ImportError: No module named 'clip_engine.visual_scorer'`

- [ ] **Step 3: Implement visual_scorer.py**

```python
# pipeline/clip_engine/visual_scorer.py
from __future__ import annotations

import os

import cv2
import numpy as np


def score_clip(clip_path: str) -> dict:
    """
    Score a video clip on motion, brightness, and contrast using OpenCV.

    Motion:     mean optical-flow magnitude across sampled frame pairs (normalized)
    Brightness: mean luminance across sampled frames, normalized to [0, 1]
    Contrast:   mean luminance std-dev across sampled frames, normalized to [0, 1]

    Returns:
        {
            "clip_id":      str,    # filename without extension
            "clip_path":    str,
            "motion_score": float,  # 0-1
            "brightness":   float,  # 0-1
            "contrast":     float,  # 0-1
            "duration_ms":  int,
        }
    """
    clip_id = os.path.splitext(os.path.basename(clip_path))[0]
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_ms = int((total_frames / fps) * 1000)

    gray_frames: list[np.ndarray] = []
    ret, frame = cap.read()
    while ret:
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        ret, frame = cap.read()
    cap.release()

    if not gray_frames:
        return {"clip_id": clip_id, "clip_path": clip_path,
                "motion_score": 0.0, "brightness": 0.0, "contrast": 0.0, "duration_ms": 0}

    # ── Motion: optical flow over at most 30 frame pairs ──────────────────────
    step = max(1, len(gray_frames) // 30)
    magnitudes: list[float] = []
    for i in range(0, len(gray_frames) - step, step):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i + step],
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(float(np.mean(mag)))

    raw_motion = float(np.mean(magnitudes)) if magnitudes else 0.0
    motion_score = min(raw_motion / 10.0, 1.0)   # ~10 px/frame = max motion

    # ── Brightness + Contrast: sampled every 5 frames ─────────────────────────
    sampled = gray_frames[::5] or gray_frames
    means = [float(np.mean(f)) for f in sampled]
    stds = [float(np.std(f)) for f in sampled]
    brightness = min(float(np.mean(means)) / 255.0, 1.0)
    contrast = min(float(np.mean(stds)) / 128.0, 1.0)

    return {
        "clip_id": clip_id,
        "clip_path": clip_path,
        "motion_score": round(motion_score, 4),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "duration_ms": duration_ms,
    }


def score_clips(clip_paths: list[str]) -> list[dict]:
    """Score multiple clips. Returns results in the same order as input."""
    return [score_clip(p) for p in clip_paths]
```

- [ ] **Step 4: Run tests — confirm all pass**

```bash
cd pipeline
python -m pytest tests/test_visual_scorer.py -v
```

Expected: **8/8 PASS**

- [ ] **Step 5: Commit**

```bash
git add pipeline/clip_engine/visual_scorer.py \
        pipeline/tests/test_visual_scorer.py
git commit -m "feat: visual scorer - optical flow motion, brightness, contrast via OpenCV"
```

---

## Task 5: Beat Matcher

**Files:**
- Create: `pipeline/clip_engine/beat_matcher.py`
- Create: `pipeline/tests/test_beat_matcher.py`

- [ ] **Step 1: Write the failing test**

```python
# pipeline/tests/test_beat_matcher.py
import pytest
from clip_engine.beat_matcher import build_timeline


def test_returns_list(sample_beat_map, sample_clip_scores):
    result = build_timeline(sample_beat_map, sample_clip_scores)
    assert isinstance(result, list)
    assert len(result) > 0


def test_entry_has_required_keys(sample_beat_map, sample_clip_scores):
    for entry in build_timeline(sample_beat_map, sample_clip_scores):
        for key in ("clip_id", "clip_path", "start_sec", "end_sec", "beat_time"):
            assert key in entry, f"Missing key: {key}"


def test_beat_times_match_beats(sample_beat_map, sample_clip_scores):
    beat_set = {round(b, 3) for b in sample_beat_map["beats"]}
    for entry in build_timeline(sample_beat_map, sample_clip_scores):
        assert round(entry["beat_time"], 3) in beat_set


def test_no_consecutive_duplicate_clips(sample_beat_map, sample_clip_scores):
    result = build_timeline(sample_beat_map, sample_clip_scores)
    for i in range(len(result) - 1):
        assert result[i]["clip_id"] != result[i + 1]["clip_id"], (
            f"Consecutive duplicate at positions {i} and {i+1}: {result[i]['clip_id']}"
        )


def test_start_sec_less_than_end_sec(sample_beat_map, sample_clip_scores):
    for entry in build_timeline(sample_beat_map, sample_clip_scores):
        assert entry["start_sec"] < entry["end_sec"]


def test_clip_duration_bounded_by_source(sample_beat_map, sample_clip_scores):
    max_source_dur = max(c["duration_ms"] / 1000 for c in sample_clip_scores)
    for entry in build_timeline(sample_beat_map, sample_clip_scores):
        assert entry["end_sec"] - entry["start_sec"] <= max_source_dur + 0.01


def test_peak_beat_gets_high_motion_clip(sample_clip_scores):
    beat_map = {
        "bpm": 120.0,
        "beats": [0.5, 1.0, 1.5],
        "peaks": [0.5],   # first beat is peak
        "drops": [],
        "segments": [{"start": 0.0, "end": 1.5, "label": "intro"}],
    }
    result = build_timeline(beat_map, sample_clip_scores)
    peak_entry = next(e for e in result if e["beat_time"] == 0.5)
    # clip_a has motion_score=0.85 (highest) — should be assigned to the peak
    assert peak_entry["clip_id"] == "clip_a"


def test_empty_timeline_on_no_beats(sample_clip_scores):
    beat_map = {"bpm": 120.0, "beats": [], "peaks": [], "drops": [], "segments": []}
    assert build_timeline(beat_map, sample_clip_scores) == []


def test_empty_timeline_on_no_clips(sample_beat_map):
    assert build_timeline(sample_beat_map, []) == []
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
cd pipeline
python -m pytest tests/test_beat_matcher.py -v
```

Expected: `ImportError: No module named 'clip_engine.beat_matcher'`

- [ ] **Step 3: Implement beat_matcher.py**

```python
# pipeline/clip_engine/beat_matcher.py
from __future__ import annotations

import copy


def build_timeline(beat_map: dict, clip_scores: list[dict]) -> list[dict]:
    """
    Build an ordered cut list that maps clips to beats.

    Rules (in priority order):
    - peak beats   → highest motion_score clips (motion_score > 0.6)
    - drop beats   → lowest motion_score clips  (motion_score <= 0.4)
    - other beats  → mid-range clips, or any available
    - no two consecutive entries may use the same clip_id
    - each entry's duration = min(beat_slot_duration, clip_duration)

    Args:
        beat_map:    Output of beat_detector.analyze()
        clip_scores: Output of visual_scorer.score_clips()

    Returns:
        List of timeline entries — one per beat interval (len(beats) - 1):
        [
            {
                "clip_id":   str,
                "clip_path": str,
                "start_sec": float,   # cut-in point within the source clip
                "end_sec":   float,   # cut-out point within the source clip
                "beat_time": float,   # playback position in the final video
            },
            ...
        ]
    """
    beats = beat_map.get("beats", [])
    if not beats or not clip_scores:
        return []

    peaks = {round(t, 3) for t in beat_map.get("peaks", [])}
    drops = {round(t, 3) for t in beat_map.get("drops", [])}

    sorted_clips = sorted(clip_scores, key=lambda c: -c["motion_score"])
    high_pool = [copy.copy(c) for c in sorted_clips if c["motion_score"] > 0.6]
    low_pool  = [copy.copy(c) for c in sorted_clips if c["motion_score"] <= 0.4]
    mid_pool  = [copy.copy(c) for c in sorted_clips
                 if 0.4 < c["motion_score"] <= 0.6]

    # Fallback: if a pool is empty use all clips
    if not high_pool:
        high_pool = [copy.copy(c) for c in sorted_clips]
    if not low_pool:
        low_pool  = [copy.copy(c) for c in sorted(clip_scores, key=lambda c: c["motion_score"])]
    if not mid_pool:
        mid_pool  = [copy.copy(c) for c in sorted_clips]

    def pick(pool: list[dict], last_id: str | None) -> dict:
        """Return first clip in pool that isn't last_id; cycle it to the end."""
        for i, clip in enumerate(pool):
            if clip["clip_id"] != last_id:
                pool.append(pool.pop(i))
                return clip
        return pool[0]  # only one clip, no choice

    timeline: list[dict] = []
    last_id: str | None = None

    for i, beat_time in enumerate(beats[:-1]):
        slot_dur = beats[i + 1] - beat_time
        t = round(beat_time, 3)

        if t in peaks:
            chosen = pick(high_pool, last_id)
        elif t in drops:
            chosen = pick(low_pool, last_id)
        else:
            chosen = pick(mid_pool, last_id)

        cut_dur = min(slot_dur, chosen["duration_ms"] / 1000.0)
        timeline.append({
            "clip_id":   chosen["clip_id"],
            "clip_path": chosen.get("clip_path", ""),
            "start_sec": 0.0,
            "end_sec":   round(cut_dur, 4),
            "beat_time": beat_time,
        })
        last_id = chosen["clip_id"]

    return timeline
```

- [ ] **Step 4: Run tests — confirm all pass**

```bash
cd pipeline
python -m pytest tests/test_beat_matcher.py -v
```

Expected: **9/9 PASS**

- [ ] **Step 5: Commit**

```bash
git add pipeline/clip_engine/beat_matcher.py \
        pipeline/tests/test_beat_matcher.py
git commit -m "feat: beat matcher - pure Python timeline builder, peaks to high-motion clips"
```

---

## Task 6: FFmpeg Builder

**Files:**
- Create: `pipeline/renderer/ffmpeg_builder.py`
- Create: `pipeline/tests/test_ffmpeg_builder.py`

- [ ] **Step 1: Write the failing test**

```python
# pipeline/tests/test_ffmpeg_builder.py
import os
import pytest
from renderer.ffmpeg_builder import EXPORT_PROFILES, render, validate_timeline


def test_export_profiles_have_four_platforms():
    for name in ("tiktok", "reels", "shorts", "youtube"):
        assert name in EXPORT_PROFILES


def test_export_profiles_have_required_keys():
    for name, p in EXPORT_PROFILES.items():
        for key in ("width", "height", "fps", "bitrate", "max_duration"):
            assert key in p, f"Profile '{name}' missing '{key}'"


def test_validate_rejects_empty_timeline():
    with pytest.raises(ValueError, match="empty"):
        validate_timeline([])


def test_validate_rejects_missing_keys():
    with pytest.raises(ValueError, match="missing keys"):
        validate_timeline([{"clip_id": "x"}])


def test_validate_rejects_inverted_times():
    entry = {"clip_id": "x", "clip_path": "/p.mp4",
              "start_sec": 2.0, "end_sec": 1.0, "beat_time": 0.0}
    with pytest.raises(ValueError, match="start_sec"):
        validate_timeline([entry])


def test_validate_accepts_valid_entry(sample_beat_map, sample_clip_scores, tmp_dir):
    from clip_engine.beat_matcher import build_timeline
    # Patch clip_path to a real (but non-existent) path — validate doesn't open files
    for cs in sample_clip_scores:
        cs["clip_path"] = os.path.join(tmp_dir, cs["clip_id"] + ".mp4")
    timeline = build_timeline(sample_beat_map, sample_clip_scores)
    validate_timeline(timeline)  # must not raise


def test_render_produces_non_empty_mp4(
    tmp_dir, audio_120bpm, high_motion_clip, low_motion_clip
):
    """End-to-end render using 4 synthetic beats — should complete in <30s."""
    from audio_analysis.beat_detector import analyze
    from clip_engine.visual_scorer import score_clips
    from clip_engine.beat_matcher import build_timeline

    beat_map = analyze(audio_120bpm)
    # Limit to 4 beats to keep test fast
    beat_map["beats"] = beat_map["beats"][:4]
    beat_map["peaks"] = [b for b in beat_map["peaks"] if b <= beat_map["beats"][-1]]
    beat_map["drops"] = [b for b in beat_map["drops"] if b <= beat_map["beats"][-1]]

    clip_scores = score_clips([high_motion_clip, low_motion_clip])
    timeline = build_timeline(beat_map, clip_scores)

    output = os.path.join(tmp_dir, "out_tiktok.mp4")
    result = render(timeline, audio_120bpm, output, export_format="tiktok")

    assert result == output
    assert os.path.exists(output)
    assert os.path.getsize(output) > 1000


def test_render_unknown_format_raises(tmp_dir, audio_120bpm, sample_beat_map,
                                      sample_clip_scores):
    with pytest.raises(ValueError, match="Unknown export format"):
        render([], audio_120bpm, os.path.join(tmp_dir, "out.mp4"),
               export_format="snapchat")
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
cd pipeline
python -m pytest tests/test_ffmpeg_builder.py -v
```

Expected: `ImportError: No module named 'renderer.ffmpeg_builder'`

- [ ] **Step 3: Implement ffmpeg_builder.py**

```python
# pipeline/renderer/ffmpeg_builder.py
from __future__ import annotations

import os
import subprocess
import tempfile

EXPORT_PROFILES: dict[str, dict] = {
    "tiktok": {
        "width": 1080, "height": 1920,
        "fps": 30, "bitrate": "8M",
        "max_duration": 180,
    },
    "reels": {
        "width": 1080, "height": 1920,
        "fps": 30, "bitrate": "8M",
        "max_duration": 90,
    },
    "shorts": {
        "width": 1080, "height": 1920,
        "fps": 60, "bitrate": "12M",
        "max_duration": 60,
    },
    "youtube": {
        "width": 1920, "height": 1080,
        "fps": 30, "bitrate": "15M",
        "max_duration": None,
    },
}

_REQUIRED_KEYS = {"clip_id", "clip_path", "start_sec", "end_sec", "beat_time"}


def validate_timeline(timeline: list[dict]) -> None:
    """
    Raise ValueError if the timeline is empty or malformed.
    Does NOT check that clip_path files exist — that is the caller's responsibility.
    """
    if not timeline:
        raise ValueError("Timeline is empty")
    for i, entry in enumerate(timeline):
        missing = _REQUIRED_KEYS - set(entry.keys())
        if missing:
            raise ValueError(f"Entry {i} missing keys: {missing}")
        if entry["start_sec"] >= entry["end_sec"]:
            raise ValueError(
                f"Entry {i}: start_sec ({entry['start_sec']}) >= end_sec ({entry['end_sec']})"
            )


def render(
    timeline: list[dict],
    audio_path: str,
    output_path: str,
    export_format: str = "tiktok",
) -> str:
    """
    Cut clips per timeline, mix audio, encode to output_path.

    Steps:
    1. Trim each clip segment to a temp file (no audio, scaled to target resolution)
    2. Write an ffmpeg concat list
    3. Concatenate all segments (stream copy — fast)
    4. Mux with audio track, final encode with target profile settings

    Args:
        timeline:       list from beat_matcher.build_timeline()
        audio_path:     music file (.mp3 or .wav)
        output_path:    destination .mp4 path
        export_format:  "tiktok" | "reels" | "shorts" | "youtube"

    Returns:
        output_path
    """
    if not timeline:
        raise ValueError("Timeline is empty")

    profile = EXPORT_PROFILES.get(export_format)
    if profile is None:
        raise ValueError(
            f"Unknown export format '{export_format}'. "
            f"Valid options: {list(EXPORT_PROFILES)}"
        )

    w, h = profile["width"], profile["height"]
    fps = profile["fps"]
    bitrate = profile["bitrate"]
    max_dur = profile["max_duration"]

    with tempfile.TemporaryDirectory() as tmp:
        # ── Step 1: Cut + scale each segment ──────────────────────────────────
        seg_paths: list[str] = []
        for idx, entry in enumerate(timeline):
            seg = os.path.join(tmp, f"seg_{idx:05d}.mp4")
            dur = entry["end_sec"] - entry["start_sec"]
            _run([
                "ffmpeg", "-y",
                "-ss", str(entry["start_sec"]),
                "-i", entry["clip_path"],
                "-t", str(dur),
                "-vf",
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1",
                "-r", str(fps),
                "-an",
                "-c:v", "libx264", "-preset", "ultrafast",
                seg,
            ])
            seg_paths.append(seg)

        # ── Step 2: Write concat list ─────────────────────────────────────────
        concat_list = os.path.join(tmp, "concat.txt")
        with open(concat_list, "w") as f:
            for seg in seg_paths:
                f.write(f"file '{seg}'\n")

        # ── Step 3: Concatenate (stream copy) ─────────────────────────────────
        concat_out = os.path.join(tmp, "merged.mp4")
        _run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            concat_out,
        ])

        # ── Step 4: Mux audio + final encode ──────────────────────────────────
        video_dur = sum(e["end_sec"] - e["start_sec"] for e in timeline)
        if max_dur is not None:
            video_dur = min(video_dur, float(max_dur))

        _run([
            "ffmpeg", "-y",
            "-i", concat_out,
            "-i", audio_path,
            "-t", str(video_dur),
            "-c:v", "libx264", "-preset", "fast",
            "-b:v", bitrate,
            "-r", str(fps),
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path,
        ])

    return output_path


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg error:\n  cmd: {' '.join(cmd)}\n  stderr: {result.stderr[-800:]}"
        )
```

- [ ] **Step 4: Run tests — confirm all pass**

```bash
cd pipeline
python -m pytest tests/test_ffmpeg_builder.py -v
```

Expected: **8/8 PASS** (the render test may take up to 30 s)

- [ ] **Step 5: Commit**

```bash
git add pipeline/renderer/ffmpeg_builder.py \
        pipeline/tests/test_ffmpeg_builder.py
git commit -m "feat: FFmpeg builder - 4 export profiles, cut + assemble + encode"
```

---

## Task 7: Drive Handler

**Files:**
- Create: `pipeline/renderer/drive_handler.py`
- Create: `pipeline/tests/test_drive_handler.py`

- [ ] **Step 1: Write the failing test**

```python
# pipeline/tests/test_drive_handler.py
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
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
cd pipeline
python -m pytest tests/test_drive_handler.py -v
```

Expected: `ImportError: No module named 'renderer.drive_handler'`

- [ ] **Step 3: Implement drive_handler.py**

```python
# pipeline/renderer/drive_handler.py
from __future__ import annotations

import io
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

    Args:
        service_account_json: Contents of the service account key file as a string.

    Returns:
        Authenticated Drive v3 service object.
    """
    info = json.loads(service_account_json)
    creds = Credentials.from_service_account_info(info, scopes=_SCOPES)
    return build("drive", "v3", credentials=creds)


def extract_folder_id(drive_url: str) -> str:
    """
    Parse a Google Drive folder sharing URL and return the folder ID.

    Supports:
      https://drive.google.com/drive/folders/<id>
      https://drive.google.com/drive/folders/<id>?usp=sharing

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
```

- [ ] **Step 4: Run tests — confirm all pass**

```bash
cd pipeline
python -m pytest tests/test_drive_handler.py -v
```

Expected: **7/7 PASS**

- [ ] **Step 5: Commit**

```bash
git add pipeline/renderer/drive_handler.py \
        pipeline/tests/test_drive_handler.py
git commit -m "feat: Google Drive handler - list assets, download, upload"
```

---

## Task 8: Supabase Schema

**Files:**
- Create: `supabase/migrations/001_initial_schema.sql`

- [ ] **Step 1: Create migration file**

```sql
-- supabase/migrations/001_initial_schema.sql

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Users ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email               TEXT UNIQUE NOT NULL,
  plan                TEXT NOT NULL DEFAULT 'trial'
    CHECK (plan IN ('trial', 'single', 'pack10', 'unlimited')),
  trial_exports_used  INT NOT NULL DEFAULT 0,
  stripe_customer_id  TEXT,
  created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── Projects ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS projects (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title            TEXT,
  drive_folder_url TEXT NOT NULL,
  export_format    TEXT NOT NULL DEFAULT 'tiktok'
    CHECK (export_format IN ('tiktok', 'reels', 'shorts', 'youtube')),
  status           TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'processing', 'complete', 'failed')),
  created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ── Jobs ──────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS jobs (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id       UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  status           TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN (
      'queued', 'downloading', 'analyzing', 'matching',
      'rendering', 'uploading', 'complete', 'failed'
    )),
  progress         INT NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
  error_message    TEXT,
  output_drive_url TEXT,
  started_at       TIMESTAMPTZ,
  completed_at     TIMESTAMPTZ
);

-- ── API Keys (BYOK — Phase 2 only) ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS api_keys (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  provider      TEXT NOT NULL CHECK (provider IN ('anthropic', 'openai')),
  encrypted_key TEXT NOT NULL,
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (user_id, provider)
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_projects_user_id  ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_project_id   ON jobs(project_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status       ON jobs(status);
```

- [ ] **Step 2: Apply to Supabase**

In Supabase dashboard → **SQL Editor** → paste and run the migration above.

- [ ] **Step 3: Verify in Table Editor**

Confirm four tables exist: `users`, `projects`, `jobs`, `api_keys`.

- [ ] **Step 4: Commit**

```bash
git add supabase/migrations/001_initial_schema.sql
git commit -m "feat: supabase schema - users, projects, jobs, api_keys"
```

---

## Task 9: FastAPI Endpoint

**Files:**
- Create: `pipeline/api.py`
- Create: `pipeline/tests/test_api.py`

- [ ] **Step 1: Write the failing test**

```python
# pipeline/tests/test_api.py
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_check():
    assert client.get("/health").json() == {"status": "ok"}


def test_process_job_empty_body():
    resp = client.post("/process-job", json={})
    assert resp.status_code == 422


def test_process_job_blank_job_id():
    resp = client.post("/process-job", json={"job_id": "", "project_id": "proj-1"})
    assert resp.status_code == 422


@patch("api.run_pipeline")
def test_process_job_returns_202(mock_run):
    resp = client.post("/process-job",
                       json={"job_id": "job-abc", "project_id": "proj-xyz"})
    assert resp.status_code == 202


@patch("api.run_pipeline")
def test_process_job_calls_run_pipeline(mock_run):
    client.post("/process-job",
                json={"job_id": "job-abc", "project_id": "proj-xyz"})
    mock_run.assert_called_once_with("job-abc", "proj-xyz")


@patch("api.run_pipeline")
def test_process_job_response_body(mock_run):
    resp = client.post("/process-job",
                       json={"job_id": "job-abc", "project_id": "proj-xyz"})
    data = resp.json()
    assert data["job_id"] == "job-abc"
    assert data["status"] == "accepted"
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
cd pipeline
python -m pytest tests/test_api.py -v
```

Expected: `ImportError: No module named 'api'`

- [ ] **Step 3: Implement api.py**

```python
# pipeline/api.py
from __future__ import annotations

import os
import tempfile
import threading
from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, Field
from supabase import create_client

from audio_analysis.beat_detector import analyze as detect_beats
from clip_engine.beat_matcher import build_timeline
from clip_engine.visual_scorer import score_clips
from renderer.drive_handler import (
    build_service,
    download_file,
    extract_folder_id,
    list_drive_assets,
    upload_file,
)
from renderer.ffmpeg_builder import render

app = FastAPI(title="McBeat Pipeline", version="1.0.0")

_SUPABASE_URL  = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY  = os.environ.get("SUPABASE_SERVICE_KEY", "")
_GOOGLE_SA_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")


class JobRequest(BaseModel):
    job_id:     Annotated[str, Field(min_length=1)]
    project_id: Annotated[str, Field(min_length=1)]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process-job", status_code=202)
def process_job(req: JobRequest):
    """
    Accepts a job and starts the pipeline in a background thread.
    Returns 202 immediately — clients poll Supabase for status updates.
    """
    t = threading.Thread(
        target=run_pipeline,
        args=(req.job_id, req.project_id),
        daemon=True,
    )
    t.start()
    return {"job_id": req.job_id, "status": "accepted"}


def run_pipeline(job_id: str, project_id: str) -> None:
    """
    Full pipeline:
    downloading → analyzing → matching → rendering → uploading → complete

    Cleans up /tmp on success or failure.
    Updates job.status + job.progress at each stage via Supabase.
    """
    db = create_client(_SUPABASE_URL, _SUPABASE_KEY)

    def set_status(status: str, progress: int) -> None:
        db.table("jobs").update(
            {"status": status, "progress": progress}
        ).eq("id", job_id).execute()

    try:
        set_status("downloading", 5)

        # ── Load project from Supabase ─────────────────────────────────────────
        proj = (
            db.table("projects")
            .select("*")
            .eq("id", project_id)
            .single()
            .execute()
            .data
        )
        if not proj:
            raise ValueError(f"Project {project_id} not found")

        drive_url     = proj["drive_folder_url"]
        export_format = proj.get("export_format", "tiktok")

        # ── Download assets from client Drive ─────────────────────────────────
        service   = build_service(_GOOGLE_SA_JSON)
        folder_id = extract_folder_id(drive_url)
        assets    = list_drive_assets(service, folder_id)

        with tempfile.TemporaryDirectory() as tmp:
            audio_path = f"{tmp}/{assets['audio']['name']}"
            download_file(service, assets["audio"]["id"], audio_path)

            clip_paths: list[str] = []
            for video in assets["videos"]:
                dest = f"{tmp}/{video['name']}"
                download_file(service, video["id"], dest)
                clip_paths.append(dest)

            set_status("analyzing", 30)

            # ── Audio analysis ────────────────────────────────────────────────
            beat_map = detect_beats(audio_path)

            set_status("matching", 50)

            # ── Clip scoring + beat matching ───────────────────────────────────
            clip_scores = score_clips(clip_paths)
            timeline    = build_timeline(beat_map, clip_scores)

            set_status("rendering", 65)

            # ── FFmpeg render ─────────────────────────────────────────────────
            output_path = f"{tmp}/mcbeat_{job_id}.mp4"
            render(timeline, audio_path, output_path, export_format=export_format)

            set_status("uploading", 90)

            # ── Upload output to Drive ─────────────────────────────────────────
            output_file_id   = upload_file(service, output_path, folder_id)
            output_drive_url = f"https://drive.google.com/file/d/{output_file_id}/view"

        # /tmp cleaned up — update Supabase to complete
        db.table("jobs").update({
            "status": "complete",
            "progress": 100,
            "output_drive_url": output_drive_url,
        }).eq("id", job_id).execute()

        db.table("projects").update({"status": "complete"}).eq("id", project_id).execute()

    except Exception as exc:
        db.table("jobs").update({
            "status": "failed",
            "error_message": str(exc)[:1000],
        }).eq("id", job_id).execute()
        db.table("projects").update({"status": "failed"}).eq("id", project_id).execute()
        raise
```

- [ ] **Step 4: Run tests — confirm all pass**

```bash
cd pipeline
python -m pytest tests/test_api.py -v
```

Expected: **6/6 PASS**

- [ ] **Step 5: Commit**

```bash
git add pipeline/api.py pipeline/tests/test_api.py
git commit -m "feat: FastAPI POST /process-job - triggers full pipeline in background thread"
```

---

## Task 10: End-to-End Integration Test

**Files:**
- Create: `pipeline/tests/test_integration.py`

- [ ] **Step 1: Write the integration test**

```python
# pipeline/tests/test_integration.py
"""
End-to-end pipeline test using only local synthetic files.
No Google Drive, no Supabase, no network calls.
Each test may take 10-30 seconds.
"""
import os
import pytest
from audio_analysis.beat_detector import analyze as detect_beats
from clip_engine.visual_scorer import score_clips
from clip_engine.beat_matcher import build_timeline
from renderer.ffmpeg_builder import render


def _run_pipeline(audio, clips, tmp_dir, fmt):
    """Helper: run full pipeline with first 5 beats only (speed)."""
    beat_map = detect_beats(audio)
    beat_map["beats"] = beat_map["beats"][:5]
    cutoff = beat_map["beats"][-1] if beat_map["beats"] else 0
    beat_map["peaks"] = [b for b in beat_map["peaks"] if b <= cutoff]
    beat_map["drops"] = [b for b in beat_map["drops"] if b <= cutoff]

    scores   = score_clips(clips)
    timeline = build_timeline(beat_map, scores)
    assert len(timeline) >= 1, "Timeline must have at least one entry"

    output = os.path.join(tmp_dir, f"final_{fmt}.mp4")
    result = render(timeline, audio, output, export_format=fmt)
    return result


def test_pipeline_tiktok(audio_120bpm, high_motion_clip, low_motion_clip, tmp_dir):
    result = _run_pipeline(audio_120bpm, [high_motion_clip, low_motion_clip], tmp_dir, "tiktok")
    assert os.path.exists(result)
    assert os.path.getsize(result) > 5_000, "Output file suspiciously small"


def test_pipeline_youtube(audio_120bpm, high_motion_clip, low_motion_clip, tmp_dir):
    result = _run_pipeline(audio_120bpm, [high_motion_clip, low_motion_clip], tmp_dir, "youtube")
    assert os.path.exists(result)
    assert os.path.getsize(result) > 5_000


def test_pipeline_90bpm(audio_90bpm, high_motion_clip, low_motion_clip, tmp_dir):
    result = _run_pipeline(audio_90bpm, [high_motion_clip, low_motion_clip], tmp_dir, "reels")
    assert os.path.exists(result)
    assert os.path.getsize(result) > 5_000


def test_beat_matcher_assigns_high_motion_to_peaks(audio_120bpm, high_motion_clip, low_motion_clip):
    beat_map = detect_beats(audio_120bpm)
    scores   = score_clips([high_motion_clip, low_motion_clip])

    # Identify the high/low motion clip IDs
    high_id = max(scores, key=lambda c: c["motion_score"])["clip_id"]

    # Force a peak at the first beat to test assignment
    beat_map["peaks"] = [beat_map["beats"][0]]
    beat_map["beats"] = beat_map["beats"][:3]
    beat_map["drops"]  = []

    timeline = build_timeline(beat_map, scores)
    first_entry = timeline[0]
    assert first_entry["clip_id"] == high_id, (
        f"Expected high-motion clip '{high_id}' at peak, got '{first_entry['clip_id']}'"
    )
```

- [ ] **Step 2: Run integration tests**

```bash
cd pipeline
python -m pytest tests/test_integration.py -v -s
```

Expected: **4/4 PASS** (allow up to 2 minutes total)

- [ ] **Step 3: Run full test suite**

```bash
cd pipeline
python -m pytest tests/ -v --tb=short
```

Expected: **All tests PASS, 0 failures**

- [ ] **Step 4: Commit**

```bash
git add pipeline/tests/test_integration.py
git commit -m "test: end-to-end integration tests for full McBeat pipeline"
```

---

## Spec Coverage

| Spec requirement | Covered in task |
|---|---|
| BPM + beat grid via Librosa | Task 2 |
| `beat_map` schema matches spec exactly | Task 2 |
| Peaks + drops + segments | Task 2, 3 |
| `clip_score` schema matches spec | Task 4 |
| OpenCV motion/brightness/contrast | Task 4 |
| High-motion clips → peaks | Task 5, integration test |
| Low-motion clips → drops | Task 5 |
| No consecutive duplicate clips | Task 5 (test + impl) |
| Duration matches music exactly | Task 5 (slot duration logic) |
| FFmpeg TikTok 1080×1920 30fps 8Mbps max 3min | Task 6 |
| FFmpeg Reels 1080×1920 30fps 8Mbps max 90s | Task 6 |
| FFmpeg Shorts 1080×1920 60fps 12Mbps max 60s | Task 6 |
| FFmpeg YouTube 1920×1080 30fps 15Mbps | Task 6 |
| Google Drive: list/download/upload | Task 7 |
| Supabase users/projects/jobs/api_keys | Task 8 |
| Job status stages: queued→…→complete/failed | Task 9 |
| `POST /process-job` FastAPI endpoint | Task 9 |
| /tmp cleanup after job | Task 9 (TemporaryDirectory) |
| Zero AI APIs — Librosa/OpenCV/FFmpeg only | All tasks |
| BYOK api_keys table (Phase 2) | Task 8 (schema created, not wired) |

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

    # Motion: mean absolute frame difference over sampled consecutive pairs
    # Frame diff is robust across codecs and synthetic content where optical
    # flow (Farneback) can return zero on simple/low-texture frames.
    step = max(1, len(gray_frames) // 30)
    diffs: list[float] = []
    for i in range(0, len(gray_frames) - step, step):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i + step])
        diffs.append(float(np.mean(diff)))

    raw_motion = float(np.mean(diffs)) if diffs else 0.0
    # ~10 mean pixel difference per frame = full motion (255 max theoretical)
    motion_score = min(raw_motion / 10.0, 1.0)

    # Brightness + Contrast: sampled every 5 frames
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

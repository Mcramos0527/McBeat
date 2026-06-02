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

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
    bpm = float(tempo) if np.isscalar(tempo) else float(np.atleast_1d(tempo)[0])
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

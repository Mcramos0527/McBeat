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

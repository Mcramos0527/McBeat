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

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

    high_id = max(scores, key=lambda c: c["motion_score"])["clip_id"]

    beat_map["peaks"] = [beat_map["beats"][0]]
    beat_map["beats"] = beat_map["beats"][:3]
    beat_map["drops"]  = []

    timeline = build_timeline(beat_map, scores)
    first_entry = timeline[0]
    assert first_entry["clip_id"] == high_id, (
        f"Expected high-motion clip '{high_id}' at peak, got '{first_entry['clip_id']}'"
    )

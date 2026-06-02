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

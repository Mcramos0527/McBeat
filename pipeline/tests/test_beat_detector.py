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

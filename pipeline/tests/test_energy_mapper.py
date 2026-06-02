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

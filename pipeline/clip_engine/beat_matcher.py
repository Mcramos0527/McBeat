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

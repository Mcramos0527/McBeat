from __future__ import annotations

import os
import subprocess
import tempfile

EXPORT_PROFILES: dict[str, dict] = {
    "tiktok": {
        "width": 1080, "height": 1920,
        "fps": 30, "bitrate": "8M",
        "max_duration": 180,
    },
    "reels": {
        "width": 1080, "height": 1920,
        "fps": 30, "bitrate": "8M",
        "max_duration": 90,
    },
    "shorts": {
        "width": 1080, "height": 1920,
        "fps": 60, "bitrate": "12M",
        "max_duration": 60,
    },
    "youtube": {
        "width": 1920, "height": 1080,
        "fps": 30, "bitrate": "15M",
        "max_duration": None,
    },
}

_REQUIRED_KEYS = {"clip_id", "clip_path", "start_sec", "end_sec", "beat_time"}


def validate_timeline(timeline: list[dict]) -> None:
    """
    Raise ValueError if the timeline is empty or malformed.
    Does NOT check that clip_path files exist — that is the caller's responsibility.
    """
    if not timeline:
        raise ValueError("Timeline is empty")
    for i, entry in enumerate(timeline):
        missing = _REQUIRED_KEYS - set(entry.keys())
        if missing:
            raise ValueError(f"Entry {i} missing keys: {missing}")
        if entry["start_sec"] >= entry["end_sec"]:
            raise ValueError(
                f"Entry {i}: start_sec ({entry['start_sec']}) >= end_sec ({entry['end_sec']})"
            )


def render(
    timeline: list[dict],
    audio_path: str,
    output_path: str,
    export_format: str = "tiktok",
) -> str:
    """
    Cut clips per timeline, mix audio, encode to output_path.

    Steps:
    1. Trim each clip segment to a temp file (no audio, scaled to target resolution)
    2. Write an ffmpeg concat list
    3. Concatenate all segments (stream copy — fast)
    4. Mux with audio track, final encode with target profile settings

    Args:
        timeline:       list from beat_matcher.build_timeline()
        audio_path:     music file (.mp3 or .wav)
        output_path:    destination .mp4 path
        export_format:  "tiktok" | "reels" | "shorts" | "youtube"

    Returns:
        output_path
    """
    profile = EXPORT_PROFILES.get(export_format)
    if profile is None:
        raise ValueError(
            f"Unknown export format '{export_format}'. "
            f"Valid options: {list(EXPORT_PROFILES)}"
        )

    if not timeline:
        raise ValueError("Timeline is empty")

    w, h = profile["width"], profile["height"]
    fps = profile["fps"]
    bitrate = profile["bitrate"]
    max_dur = profile["max_duration"]

    with tempfile.TemporaryDirectory() as tmp:
        # Step 1: Cut + scale each segment
        seg_paths: list[str] = []
        for idx, entry in enumerate(timeline):
            seg = os.path.join(tmp, f"seg_{idx:05d}.mp4")
            dur = entry["end_sec"] - entry["start_sec"]
            _run([
                "ffmpeg", "-y",
                "-ss", str(entry["start_sec"]),
                "-i", entry["clip_path"],
                "-t", str(dur),
                "-vf",
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1",
                "-r", str(fps),
                "-an",
                "-c:v", "libx264", "-preset", "ultrafast",
                seg,
            ])
            seg_paths.append(seg)

        # Step 2: Write concat list (use forward slashes for ffmpeg on Windows)
        concat_list = os.path.join(tmp, "concat.txt")
        with open(concat_list, "w") as f:
            for seg in seg_paths:
                f.write(f"file '{seg.replace(chr(92), '/')}'\n")

        # Step 3: Concatenate (stream copy)
        concat_out = os.path.join(tmp, "merged.mp4")
        _run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            concat_out,
        ])

        # Step 4: Mux audio + final encode
        video_dur = sum(e["end_sec"] - e["start_sec"] for e in timeline)
        if max_dur is not None:
            video_dur = min(video_dur, float(max_dur))

        _run([
            "ffmpeg", "-y",
            "-i", concat_out,
            "-i", audio_path,
            "-t", str(video_dur),
            "-c:v", "libx264", "-preset", "fast",
            "-b:v", bitrate,
            "-r", str(fps),
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path,
        ])

    return output_path


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg error:\n  cmd: {' '.join(cmd)}\n  stderr: {result.stderr[-800:]}"
        )

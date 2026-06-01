# Pipeline Deep Dive

## Overview

The McBeat pipeline is the core of the product. It transforms two inputs — a music track and a set of raw video clips — into a beat-synchronized video.

```
Music Track + Raw Clips
         │
         ▼
  [1] Audio Analysis
         │
         ▼
  [2] Clip Scoring
         │
         ▼
  [3] Beat Matching
         │
         ▼
  [4] Caption Generation (parallel)
         │
         ▼
  [5] Render & Export
         │
         ▼
   Final Video Output
```

---

## Stage 1: Audio Analysis

**Tools:** Librosa, Essentia  
**Input:** MP3 or WAV file  
**Output:** Structured beat map

What we extract:
- **BPM** — the tempo of the track
- **Beat timestamps** — every beat position in seconds
- **Energy peaks** — moments of high RMS energy (choruses, drops)
- **Structural boundaries** — where verse/chorus/bridge transitions happen
- **Key & mode** — for future color grading matching (minor = cooler tones, major = warmer)

The output beat map looks like:
```json
{
  "bpm": 128,
  "beats": [0.47, 0.94, 1.41, 1.88, ...],
  "peaks": [16.5, 32.0, 48.5, ...],
  "segments": [
    { "start": 0.0, "end": 16.5, "label": "intro" },
    { "start": 16.5, "end": 48.0, "label": "verse" },
    { "start": 48.0, "end": 80.0, "label": "chorus" }
  ]
}
```

---

## Stage 2: Clip Scoring

**Tools:** OpenCV  
**Input:** Each video clip  
**Output:** Score object per clip

Every clip gets scored on:
- **Motion score** — how much movement is in the clip (optical flow analysis)
- **Brightness** — average luminance
- **Contrast** — visual dynamic range
- **Duration** — total clip length in ms

High motion + high contrast clips → paired with beat peaks  
Low motion + wide shots → paired with breakdowns/intros

```python
# Simplified scoring
{
  "clip_id": "uuid",
  "motion_score": 0.82,    # 0-1, higher = more movement
  "brightness": 0.61,
  "contrast": 0.74,
  "duration_ms": 4200
}
```

---

## Stage 3: Beat Matching

**Input:** Beat map + scored clips  
**Output:** Ordered timeline (which clip plays at which beat)

The matching algorithm:
1. Groups beats into "slots" based on clip duration
2. Sorts clips by energy score (high → low)
3. Assigns high-energy clips to peak moments
4. Fills remaining slots with medium/low energy clips
5. Avoids repeating the same clip consecutively
6. Ensures total duration matches the music track

This is the most creative part of the pipeline and the one that will evolve most over time.

---

## Stage 4: Caption Generation

**Tools:** Whisper, Claude API  
**Runs in parallel with Stage 2-3**  
**Only activated if the music track contains vocals**

Steps:
1. Whisper transcribes vocals → raw transcript with timestamps
2. Claude receives transcript + user's brand voice config
3. Claude outputs styled captions with tone, emojis (optional), and formatting
4. Caption objects are timed to Whisper's word-level timestamps

Caption styles:
- **Bold** — large text, centered, high contrast (TikTok native)
- **Minimal** — small lower-third, subtle (YouTube/cinematic)
- **Lyric** — word-by-word reveal, karaoke style

---

## Stage 5: Render & Export

**Tools:** FFmpeg, Remotion (for caption overlay)  
**Input:** Timeline + audio + captions  
**Output:** Final MP4

Process:
1. FFmpeg builds concat filter from timeline
2. Each clip is trimmed to its assigned duration
3. All clips are scaled to target resolution (1080x1920 for vertical)
4. Audio track is mixed in
5. Remotion bakes caption overlays as a second pass
6. Final encode with platform-optimized settings

Export profiles:
| Platform | Resolution | FPS | Bitrate | Max Duration |
|---|---|---|---|---|
| TikTok | 1080x1920 | 30 | 8Mbps | 3 min |
| Instagram Reels | 1080x1920 | 30 | 8Mbps | 90s |
| YouTube Shorts | 1080x1920 | 60 | 12Mbps | 60s |
| YouTube | 1920x1080 | 30 | 15Mbps | Unlimited |

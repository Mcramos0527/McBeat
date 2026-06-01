# Architecture Deep Dive

## System Design Principles

1. **Async-first** — Video rendering is heavy. Every job goes through a queue. No blocking the API.
2. **Music-native** — The audio analysis is the source of truth. Everything downstream (clip selection, timing, transitions) derives from it.
3. **Stateless workers** — Each Celery worker is stateless. State lives in PostgreSQL and Redis. Workers can scale horizontally.
4. **Storage-cheap** — We use Cloudflare R2 to eliminate egress fees. Creators upload large files constantly.

---

## Data Flow

```
User uploads assets
        │
        ▼
R2 Storage (raw assets)
        │
        ▼
Job created in PostgreSQL (status: QUEUED)
        │
        ▼
Celery picks up job
        │
        ├──► Audio Analysis Worker
        │         Librosa → beats[], peaks[], drops[], bpm
        │         Essentia → key, mode, energy_curve
        │
        ├──► Caption Worker (if vocal track)
        │         Whisper → raw_transcript[]
        │         Claude API → styled_captions[]
        │
        ├──► Clip Scoring Worker
        │         OpenCV → motion_score, brightness, contrast per clip
        │
        ├──► Beat Matcher
        │         Combines: beats + clip scores → ordered timeline
        │
        └──► Render Worker
                  FFmpeg → assembles final video
                  Remotion → bakes in captions
                  Output → R2 (output folder)
                  Job status → COMPLETE
                  User notified via WebSocket
```

---

## Database Schema (simplified)

```sql
-- Users
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  plan TEXT DEFAULT 'starter',
  created_at TIMESTAMP DEFAULT NOW()
);

-- Projects
CREATE TABLE projects (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  title TEXT,
  status TEXT DEFAULT 'draft',
  music_file_key TEXT,       -- R2 key
  output_file_key TEXT,      -- R2 key
  config JSONB,              -- export format, caption style, etc.
  created_at TIMESTAMP DEFAULT NOW()
);

-- Clips
CREATE TABLE clips (
  id UUID PRIMARY KEY,
  project_id UUID REFERENCES projects(id),
  file_key TEXT,             -- R2 key
  duration_ms INTEGER,
  energy_score FLOAT,
  motion_score FLOAT,
  order_index INTEGER        -- final position in timeline
);

-- Jobs
CREATE TABLE jobs (
  id UUID PRIMARY KEY,
  project_id UUID REFERENCES projects(id),
  type TEXT,                 -- 'full_render', 'preview', 'caption_only'
  status TEXT DEFAULT 'queued',
  worker_id TEXT,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  error_message TEXT
);
```

---

## Audio Analysis — Technical Detail

The beat detection module uses a multi-pass approach:

```python
import librosa
import numpy as np

class AudioAnalyzer:
    def __init__(self, audio_path: str):
        self.y, self.sr = librosa.load(audio_path)
    
    def get_beats(self) -> list[float]:
        """Returns timestamps (seconds) of every beat"""
        tempo, mcbeats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        return librosa.frames_to_time(mcbeats, sr=self.sr).tolist()
    
    def get_energy_peaks(self, threshold: float = 0.85) -> list[float]:
        """Returns timestamps of high-energy moments (drops, climaxes)"""
        rms = librosa.feature.rms(y=self.y)[0]
        peak_frames = np.where(rms > np.percentile(rms, threshold * 100))[0]
        return librosa.frames_to_time(peak_frames, sr=self.sr).tolist()
    
    def get_structural_segments(self) -> list[dict]:
        """Detects verses, choruses, bridges using spectral analysis"""
        # Uses librosa.segment for structural boundary detection
        boundaries = librosa.segment.agglomerative(
            librosa.feature.mfcc(y=self.y, sr=self.sr), 
            k=8
        )
        return [{"start": s, "label": f"segment_{i}"} 
                for i, s in enumerate(librosa.frames_to_time(boundaries))]
```

---

## Render Pipeline — FFmpeg Command Generation

```python
class FFmpegBuilder:
    def build_command(self, timeline: list[ClipSegment], audio_path: str, output_path: str) -> str:
        """
        Builds an FFmpeg concat command from the beat-matched timeline.
        Each ClipSegment has: clip_path, start_ms, end_ms
        """
        inputs = []
        filter_parts = []
        
        for i, segment in enumerate(timeline):
            inputs.append(f"-ss {segment.start_ms/1000} -t {segment.duration_ms/1000} -i {segment.clip_path}")
            filter_parts.append(f"[{i}:v]scale=1080:1920,setsar=1[v{i}]")
        
        concat = "".join([f"[v{i}]" for i in range(len(timeline))])
        concat += f"concat=n={len(timeline)}:v=1:a=0[vout]"
        
        filter_complex = ";".join(filter_parts) + ";" + concat
        
        return f"""
        ffmpeg {" ".join(inputs)} -i {audio_path}
        -filter_complex "{filter_complex}"
        -map "[vout]" -map {len(timeline)}:a
        -c:v libx264 -preset fast -crf 23
        -c:a aac -b:a 192k
        {output_path}
        """
```

---

## Scaling Strategy

### Current (MVP)
- Single Railway instance
- 2 Celery workers
- ~50 renders/day capacity

### Growth (500+ users)
- Workers scale horizontally via Railway autoscale
- R2 handles storage without limits
- PostgreSQL → managed (Railway Postgres or Supabase)
- Redis → Upstash (serverless Redis)

### Scale (10k+ users)
- GCP Cloud Run for workers (pay per render, not idle)
- CDN for output video delivery (Cloudflare)
- Separate queues per job type (priority queue for Pro tier)
- GPU workers for faster FFmpeg encoding (optional)

---

## BYOK — Bring Your Own Keys

McBeat never pays for AI. The user provides their own API keys, stored encrypted.

### User Settings Flow
```
User visits Settings → API Keys
├── Pastes ANTHROPIC_API_KEY
└── Pastes OPENAI_API_KEY

Backend:
├── Encrypts key with AES-256 (per-user salt)
├── Stores encrypted blob in PostgreSQL
└── Never logs or exposes raw keys
```

### How Keys Are Used at Render Time
```python
class KeyVault:
    def get_key(self, user_id: str, provider: str) -> str:
        encrypted = db.query(ApiKey).filter_by(
            user_id=user_id, provider=provider
        ).first()
        return decrypt_aes256(encrypted.blob, key=user_id)

# In the caption worker:
vault = KeyVault()
anthropic_key = vault.get_key(job.user_id, "anthropic")
client = anthropic.Anthropic(api_key=anthropic_key)
```

### Cost Per Video (User Pays Directly)
| Operation | Model | Est. Cost |
|---|---|---|
| Caption generation | Claude Haiku | ~$0.01-0.03 |
| Transcription (2 min audio) | Whisper | ~$0.02 |
| **Total per video** | | **~$0.03-0.05** |

McBeat's own compute cost per render: ~$0.05 (Render.com worker time)

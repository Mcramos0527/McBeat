# 🎬 McBeat

> **Music-driven video editing powered by AI.**  
> Upload your music. Upload your clips. We sync every cut to the beat — automatically.

![McBeat Banner](docs/assets/banner.png)

---

## The Problem

Every major video editing tool — Adobe Premiere, CapCut, DaVinci Resolve — treats video as the primary element and music as an afterthought.

But the creators who get millions of views know the truth:

> **The music drives the emotion. The video just has to keep up.**

Beat-synced editing — where every cut, transition, and visual moment aligns with the peaks, drops, and rhythm of the music — is what separates a viral travel reel from a forgettable slideshow. Today, doing this right requires either an expensive professional editor or hours of manual work.

**McBeat changes that.**

---

## What McBeat Does

1. **You upload** your music track (MP3/WAV) and raw video clips (MP4/MOV)
2. **Our AI analyzes** the audio — BPM, beat peaks, energy drops, transitions
3. **The engine selects and arranges** your clips, cutting exactly on the beat
4. **Captions are generated** with personality — not generic subtitles, but styled text that matches your brand voice
5. **You preview, adjust, and export** — optimized for TikTok, Instagram Reels, YouTube Shorts, or full-length

---

## Target Users

| Creator Type | Pain Point | McBeat Solution |
|---|---|---|
| 🎵 Independent musicians | Need high-quality music videos without an editor | Beat-native pipeline, music is always the hero |
| ✈️ Travel & lifestyle creators | Hours of raw footage, no time to edit | AI sorts, syncs, and styles the whole reel |
| 🎧 DJs & event producers | Highlight reels need to feel the energy | Drop detection + energy-based clip selection |
| 📱 Content agencies | High volume, consistent quality needed | White-label tier with API access |

---

## Core Features

### 🎵 Beat-Sync Engine
- Automatic BPM detection and beat grid mapping
- Peak energy detection for cut points
- Drop/transition detection for scene changes
- Supports any genre — hip-hop, electronic, cinematic, acoustic

### 🤖 AI Caption System (Claude-powered)
- Transcribes spoken audio with Whisper
- Rewrites captions in your brand voice via Claude API
- Multiple caption styles: bold overlay, minimal lower-third, lyric-style
- Multi-language support

### 🎞️ Smart Clip Selection
- Analyzes visual energy of each clip (motion, brightness, contrast)
- Matches high-energy clips to beat peaks
- Matches slow/wide shots to breakdowns
- Color coherence across the timeline

### 📤 Multi-Platform Export
- TikTok (9:16, max 3 min)
- Instagram Reels (9:16, max 90s)
- YouTube Shorts (9:16, max 60s)
- YouTube long-form (16:9)
- Custom dimensions

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                FRONTEND                         │
│           Next.js 14 + Tailwind CSS             │
│     Upload UI · Preview Player · Dashboard      │
└───────────────────┬─────────────────────────────┘
                    │ REST / WebSocket
┌───────────────────▼─────────────────────────────┐
│                API LAYER                        │
│              FastAPI (Python)                   │
│     Auth · Projects · Jobs · Webhooks           │
└──────┬───────────────────────┬──────────────────┘
       │                       │
┌──────▼──────┐       ┌────────▼────────┐
│   STORAGE   │       │   JOB QUEUE     │
│ Cloudflare  │       │  Redis + Celery │
│     R2      │       │  Async Workers  │
└─────────────┘       └────────┬────────┘
                               │
              ┌────────────────▼──────────────────┐
              │           AI PIPELINE             │
              │                                   │
              │  [1] Audio Analysis               │
              │      Librosa → beats/peaks/BPM    │
              │      Essentia → musical features  │
              │                                   │
              │  [2] Caption Generation           │
              │      Whisper → transcription      │
              │      Claude API → styled text     │
              │                                   │
              │  [3] Clip Intelligence            │
              │      OpenCV → visual energy score │
              │      Custom → clip-beat matching  │
              │                                   │
              │  [4] Render Engine                │
              │      FFmpeg → cut & assemble      │
              │      Remotion → caption overlay   │
              └───────────────────────────────────┘
```

---

## Tech Stack

### Frontend
| Technology | Purpose |
|---|---|
| **Next.js 14** | App Router, SSR, API routes |
| **Tailwind CSS + shadcn/ui** | Component library and styling |
| **WaveSurfer.js** | Audio waveform visualization |
| **Uppy** | Large file uploads with progress |
| **Zustand** | Client state management |

### Backend
| Technology | Purpose |
|---|---|
| **FastAPI** | Main API server (Python) |
| **PostgreSQL** | Users, projects, jobs, configs |
| **Redis** | Job queue and caching |
| **Celery** | Async task processing for renders |
| **SQLAlchemy** | ORM |

### Storage
| Technology | Purpose |
|---|---|
| **Cloudflare R2** | Primary asset storage (no egress fees) |
| **Structure** | `/{user_id}/{project_id}/raw/` + `/output/` |

### AI / ML Pipeline
| Technology | Purpose |
|---|---|
| **Librosa** | Beat detection, BPM, onset detection |
| **Essentia** (Mozilla) | Deep musical feature extraction |
| **OpenAI Whisper** | Speech-to-text for captions |
| **Claude API** (Anthropic) | Caption personality & rewriting |
| **OpenCV** | Visual energy analysis per clip |

### Render Engine
| Technology | Purpose |
|---|---|
| **FFmpeg** | Video cutting, encoding, assembly |
| **Remotion** | React-based caption overlays |
| **Sharp** | Thumbnail generation |

### Infrastructure — Zero Cost Stack
| Technology | Purpose | Cost |
|---|---|---|
| **Vercel** | Frontend hosting (Next.js) | Free |
| **Render.com** | Backend API + workers | Free tier |
| **Supabase** | PostgreSQL + file storage | Free up to 500MB |
| **Cloudflare R2** | Video asset storage | Free up to 10GB |
| **Upstash** | Serverless Redis (job queue) | Free tier |
| **GitHub Actions** | CI/CD pipeline | Free |
| **Sentry** | Error tracking | Free tier |

> 💡 This stack costs **$0/month** until you have meaningful traffic. Scale to paid tiers only when revenue justifies it.

---

## Repository Structure

```
mcbeat/
├── README.md
├── docker-compose.yml
├── .env.example
│
├── frontend/                  # Next.js app
│   ├── app/
│   │   ├── (auth)/
│   │   ├── dashboard/
│   │   ├── projects/
│   │   └── api/
│   ├── components/
│   │   ├── upload/
│   │   ├── waveform/
│   │   ├── preview/
│   │   └── captions/
│   └── package.json
│
├── backend/                   # FastAPI app
│   ├── app/
│   │   ├── api/
│   │   │   ├── auth.py
│   │   │   ├── projects.py
│   │   │   └── jobs.py
│   │   ├── models/
│   │   ├── services/
│   │   └── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── pipeline/                  # AI processing workers
│   ├── audio_analysis/
│   │   ├── beat_detector.py
│   │   └── energy_mapper.py
│   ├── captions/
│   │   ├── transcriber.py
│   │   └── styler.py
│   ├── clip_engine/
│   │   ├── visual_scorer.py
│   │   └── beat_matcher.py
│   ├── renderer/
│   │   ├── ffmpeg_builder.py
│   │   └── caption_overlay.py
│   └── worker.py
│
├── infra/                     # Infrastructure config
│   ├── nginx/
│   ├── docker/
│   └── github-actions/
│
└── docs/
    ├── architecture.md
    ├── api-reference.md
    ├── pipeline-deep-dive.md
    └── roadmap.md
```

---

## The Beat-Sync Pipeline — How It Works

```python
# Simplified pipeline flow

# 1. Analyze the music
beats, peaks, drops = AudioAnalyzer(track).analyze()
# beats   → [0.42s, 0.84s, 1.26s, ...]  (every beat)
# peaks   → [8.4s, 16.8s, ...]           (high energy moments)
# drops   → [32.1s, 64.2s, ...]          (transitions/breakdowns)

# 2. Score the clips
scored_clips = VisualScorer(clips).score()
# Returns each clip with: energy_score, motion_score, brightness

# 3. Match clips to beats
timeline = BeatMatcher(beats, peaks, drops, scored_clips).build()
# High-energy clips → peaks
# Wide/slow clips   → drops/breakdowns
# Standard clips    → regular beats

# 4. Render
FFmpegBuilder(timeline, track).render(output_format="tiktok")
```

---

## Roadmap

### Phase 1 — MVP (Q3 2026)
- [ ] User auth & project management
- [ ] File upload (music + video clips)
- [ ] Beat detection pipeline
- [ ] Basic beat-sync rendering
- [ ] Caption generation (Whisper + Claude)
- [ ] Export: TikTok, Reels, Shorts

### Phase 2 — Growth (Q4 2026)
- [ ] Direct publish to social platforms
- [ ] Licensed music library integration
- [ ] Genre-based templates (Hip-Hop, Electronic, Cinematic, Acoustic)
- [ ] Caption style editor (font, color, animation)
- [ ] Project history & versioning

### Phase 3 — Scale (Q1 2027)
- [ ] White-label / API tier for agencies
- [ ] Team collaboration features
- [ ] Advanced AI: learn user's editing style
- [ ] Mobile app (React Native)
- [ ] Analytics dashboard (what cut style gets more engagement)

---

## Business Model

### BYOK — Bring Your Own Keys

McBeat operates on a **zero AI cost** model. Users connect their own API keys in their account settings:

- `ANTHROPIC_API_KEY` — powers caption generation (Claude)
- `OPENAI_API_KEY` — powers transcription (Whisper)

This means **you pay Anthropic/OpenAI directly** for what you use. McBeat only charges for platform access. Keys are AES-256 encrypted at rest and never logged.

---

### Pricing

| Plan | Price | What You Get |
|---|---|---|
| 🎁 **Free Trial** | $0 — 1 month | 3 videos, all features, no watermark. No credit card required. |
| 🎬 **Single Video** | **$3** | Pay as you go. 1 export, all platforms, full quality. |
| 📦 **Pack 10** | **$25** | $2.50/video. Best for regular creators. |
| ♾️ **Unlimited** | **$49/mo** | Unlimited exports. Priority render queue. |

> 💡 Your AI usage (captions, transcription) is billed separately and directly by Anthropic/OpenAI to your account. Typical cost per video: **$0.02–$0.10** depending on length.

---

### Unit Economics

| Metric | Value |
|---|---|
| McBeat cost per video render | ~$0.05 (compute only) |
| McBeat revenue per video (single) | $3.00 |
| **Gross margin per video** | **~98%** |
| Break-even (monthly infra ~$30) | **10 videos sold** |

---

## Why Now

- Short-form video is the dominant content format globally
- Independent creators are a $250B+ economy
- Beat-synced editing is **the standard** for music/travel content — but no tool makes it accessible
- AI audio analysis (Librosa, Essentia) is mature enough to do this reliably
- The gap between "CapCut generic" and "professional editor" is exactly where McBeat lives

---

## Team

| Role | Status |
|---|---|
| Product / Architecture | ✅ Max — @maxramospe |
| Automation / n8n Backend | 🔄 Ligia |
| Software Engineering | 🔄 Harshil |
| UI/UX | 🔲 Open |
| ML Engineer | 🔲 Open |

---

## Getting Started (Dev)

```bash
# Clone the repo
git clone https://github.com/your-org/mcbeat.git
cd mcbeat

# Copy environment variables
cp .env.example .env

# Start all services
docker-compose up

# Frontend runs on http://localhost:3000
# Backend API on http://localhost:8000
# API docs on http://localhost:8000/docs
```

---

## Contributing

We're in early development. If you're interested in contributing, open an issue or reach out directly.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>McBeat</strong> — Because the music always comes first.
</p>

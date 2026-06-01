# Contributing to McBeat

Thanks for your interest in contributing. Here's how to get involved.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/mcbeat.git`
3. Copy `.env.example` to `.env` and configure your local values
4. Run `docker-compose up` to start all services
5. Frontend: http://localhost:3000 | API: http://localhost:8000

## Branch Naming

- `feature/short-description` — new features
- `fix/short-description` — bug fixes
- `pipeline/short-description` — AI/processing pipeline work
- `infra/short-description` — infrastructure changes

## Commit Style

We use conventional commits:
- `feat: add beat peak detection`
- `fix: correct FFmpeg timestamp offset`
- `docs: update pipeline deep dive`
- `chore: update dependencies`

## Areas Where Help Is Most Needed

- 🤖 **ML/Pipeline** — improving clip-to-beat matching algorithm
- 🎨 **Frontend** — upload UI, waveform visualizer, preview player
- 🔧 **Backend** — job queue optimization, WebSocket implementation
- 📱 **Mobile** — React Native companion app (Phase 3)

## Questions

Open an issue or reach out to the team directly.

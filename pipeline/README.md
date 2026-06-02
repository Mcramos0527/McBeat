---
title: McBeat Pipeline
emoji: 🎵
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
---

# McBeat Pipeline API

Beat-sync video editing engine — music leads, video follows.

**Endpoints:**
- `GET /health` — status check
- `POST /process-job` — trigger render `{ job_id, project_id }`
- `GET /docs` — Swagger UI

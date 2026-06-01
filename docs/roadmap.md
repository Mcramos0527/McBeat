# McBeat Roadmap

## Philosophy
We build the pipeline first. The UI second. The business model third.  
A great beat-sync engine with a basic UI beats a beautiful app with a mediocre engine every time.

---

## Phase 1 — MVP (Q3 2026)
**Goal:** First paying user. One complete flow working end-to-end.

- [ ] User authentication (email + Google OAuth)
- [ ] Project creation and asset upload (music + clips)
- [ ] Beat detection pipeline (Librosa)
- [ ] Basic clip-to-beat matching algorithm
- [ ] FFmpeg render worker
- [ ] Whisper caption transcription
- [ ] Claude caption styling (3 styles: Bold, Minimal, Lyric)
- [ ] Export: TikTok 9:16, Reels 9:16, Shorts 9:16
- [ ] Basic dashboard (projects list, job status)
- [ ] Email notifications (render complete)

**Success metric:** 10 beta users complete at least 1 export

---

## Phase 2 — Product-Market Fit (Q4 2026)
**Goal:** Understand which user type (musicians vs travel creators) converts and retains better.

- [ ] Direct publish to TikTok and Instagram (API integration)
- [ ] Caption style editor (font, color, size, animation)
- [ ] Genre-based presets (Hip-Hop, Electronic, Cinematic, Acoustic, Lo-fi)
- [ ] Preview mode (30s clip before full render)
- [ ] Project history and re-export
- [ ] Referral system
- [ ] Stripe billing integration (Creator + Pro tiers)

**Success metric:** 100 paying users, <10% monthly churn

---

## Phase 3 — Scale (Q1 2027)
**Goal:** Agency and B2B revenue stream.

- [ ] White-label API for content agencies
- [ ] Team workspaces (multiple users per account)
- [ ] Custom brand presets (saved caption styles, color grading filters)
- [ ] Advanced AI: learn individual creator's editing style from past exports
- [ ] Licensed music library (partnership with Artlist or Epidemic Sound)
- [ ] Mobile companion app (React Native) — upload clips from phone
- [ ] Engagement analytics (A/B test cut styles, caption placements)

**Success metric:** First agency contract signed

---

## Backlog (Future Consideration)
- AI voiceover generation (ElevenLabs integration)
- Automatic thumbnail generation per platform
- Script-to-video for talking-head content creators
- Community marketplace for caption style templates
- Multi-language caption auto-translation

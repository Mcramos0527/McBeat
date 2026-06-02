-- supabase/migrations/001_initial_schema.sql

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Users ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email               TEXT UNIQUE NOT NULL,
  plan                TEXT NOT NULL DEFAULT 'trial'
    CHECK (plan IN ('trial', 'single', 'pack10', 'unlimited')),
  trial_exports_used  INT NOT NULL DEFAULT 0,
  stripe_customer_id  TEXT,
  created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── Projects ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS projects (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title            TEXT,
  drive_folder_url TEXT NOT NULL,
  export_format    TEXT NOT NULL DEFAULT 'tiktok'
    CHECK (export_format IN ('tiktok', 'reels', 'shorts', 'youtube')),
  status           TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'processing', 'complete', 'failed')),
  created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ── Jobs ──────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS jobs (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id       UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  status           TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN (
      'queued', 'downloading', 'analyzing', 'matching',
      'rendering', 'uploading', 'complete', 'failed'
    )),
  progress         INT NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
  error_message    TEXT,
  output_drive_url TEXT,
  started_at       TIMESTAMPTZ,
  completed_at     TIMESTAMPTZ
);

-- ── API Keys (BYOK — Phase 2 only) ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS api_keys (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  provider      TEXT NOT NULL CHECK (provider IN ('anthropic', 'openai')),
  encrypted_key TEXT NOT NULL,
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (user_id, provider)
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_projects_user_id  ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_project_id   ON jobs(project_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status       ON jobs(status);

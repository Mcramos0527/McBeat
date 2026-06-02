from __future__ import annotations

import os
import tempfile
import threading
from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="McBeat Pipeline", version="1.0.0")

_SUPABASE_URL  = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY  = os.environ.get("SUPABASE_SERVICE_KEY", "")
_GOOGLE_SA_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")


class JobRequest(BaseModel):
    job_id:     Annotated[str, Field(min_length=1)]
    project_id: Annotated[str, Field(min_length=1)]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process-job", status_code=202)
def process_job(req: JobRequest):
    """
    Accepts a job and starts the pipeline in a background thread.
    Returns 202 immediately — clients poll Supabase for status updates.
    """
    t = threading.Thread(
        target=run_pipeline,
        args=(req.job_id, req.project_id),
        daemon=True,
    )
    t.start()
    return {"job_id": req.job_id, "status": "accepted"}


def run_pipeline(job_id: str, project_id: str) -> None:
    """
    Full pipeline:
    downloading → analyzing → matching → rendering → uploading → complete

    Cleans up /tmp on success or failure.
    Updates job.status + job.progress at each stage via Supabase.
    """
    from supabase import create_client  # lazy import — not needed at module load time
    from audio_analysis.beat_detector import analyze as detect_beats
    from clip_engine.beat_matcher import build_timeline
    from clip_engine.visual_scorer import score_clips
    from renderer.drive_handler import (
        build_service,
        download_file,
        extract_folder_id,
        list_drive_assets,
        upload_file,
    )
    from renderer.ffmpeg_builder import render

    db = create_client(_SUPABASE_URL, _SUPABASE_KEY)

    def set_status(status: str, progress: int) -> None:
        db.table("jobs").update(
            {"status": status, "progress": progress}
        ).eq("id", job_id).execute()

    try:
        set_status("downloading", 5)

        proj = (
            db.table("projects")
            .select("*")
            .eq("id", project_id)
            .single()
            .execute()
            .data
        )
        if not proj:
            raise ValueError(f"Project {project_id} not found")

        drive_url     = proj["drive_folder_url"]
        export_format = proj.get("export_format", "tiktok")

        service   = build_service(_GOOGLE_SA_JSON)
        folder_id = extract_folder_id(drive_url)
        assets    = list_drive_assets(service, folder_id)

        with tempfile.TemporaryDirectory() as tmp:
            audio_path = f"{tmp}/{assets['audio']['name']}"
            download_file(service, assets["audio"]["id"], audio_path)

            clip_paths: list[str] = []
            for video in assets["videos"]:
                dest = f"{tmp}/{video['name']}"
                download_file(service, video["id"], dest)
                clip_paths.append(dest)

            set_status("analyzing", 30)
            beat_map = detect_beats(audio_path)

            set_status("matching", 50)
            clip_scores = score_clips(clip_paths)
            timeline    = build_timeline(beat_map, clip_scores)

            set_status("rendering", 65)
            output_path = f"{tmp}/mcbeat_{job_id}.mp4"
            render(timeline, audio_path, output_path, export_format=export_format)

            set_status("uploading", 90)
            output_file_id   = upload_file(service, output_path, folder_id)
            output_drive_url = f"https://drive.google.com/file/d/{output_file_id}/view"

        db.table("jobs").update({
            "status": "complete",
            "progress": 100,
            "output_drive_url": output_drive_url,
        }).eq("id", job_id).execute()

        db.table("projects").update({"status": "complete"}).eq("id", project_id).execute()

    except Exception as exc:
        db.table("jobs").update({
            "status": "failed",
            "error_message": str(exc)[:1000],
        }).eq("id", job_id).execute()
        db.table("projects").update({"status": "failed"}).eq("id", project_id).execute()
        raise

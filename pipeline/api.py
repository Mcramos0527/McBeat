from __future__ import annotations

import os
import tempfile
import threading
import uuid
from typing import Annotated, List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="McBeat Pipeline", version="1.0.0")

# In-memory job store for upload-based jobs (no Supabase needed)
_jobs: dict[str, dict] = {}

_SUPABASE_URL  = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY  = os.environ.get("SUPABASE_SERVICE_KEY", "")
_GOOGLE_SA_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")


class JobRequest(BaseModel):
    job_id:     Annotated[str, Field(min_length=1)]
    project_id: Annotated[str, Field(min_length=1)]


class RenderRequest(BaseModel):
    drive_url:     Annotated[str, Field(min_length=10)]
    export_format: str = "tiktok"
    user_email:    str = "demo@mcbeat.io"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/render-upload", status_code=202)
async def render_upload(
    audio: UploadFile = File(...),
    clips: List[UploadFile] = File(...),
    export_format: str = Form("tiktok"),
):
    """
    Upload-based pipeline — no Google Drive needed.
    Accepts audio file + video clips directly.
    Returns job_id immediately; poll /job/{job_id} for status.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "progress": 0,
                     "output_path": None, "error": None}

    # Read files into memory before the async context closes
    audio_bytes = await audio.read()
    audio_name  = audio.filename or "audio.mp3"
    clips_data  = [(c.filename or f"clip_{i}.mp4", await c.read())
                   for i, c in enumerate(clips)]

    t = threading.Thread(
        target=_run_upload_pipeline,
        args=(job_id, audio_bytes, audio_name, clips_data, export_format),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id, "status": "accepted"}


@app.get("/job/{job_id}")
def get_job(job_id: str):
    """Poll upload-based job status."""
    # Try in-memory store first
    if job_id in _jobs:
        j = _jobs[job_id]
        return {
            "status": j["status"],
            "progress": j["progress"],
            "output_drive_url": j.get("output_url"),
            "error_message": j.get("error"),
        }
    # Fall back to Supabase for Drive-based jobs
    if _SUPABASE_URL and _SUPABASE_KEY:
        from supabase import create_client
        db = create_client(_SUPABASE_URL, _SUPABASE_KEY)
        res = db.table("jobs").select(
            "status,progress,output_drive_url,error_message"
        ).eq("id", job_id).single().execute()
        return res.data or {"status": "not_found"}
    return {"status": "not_found"}


@app.get("/download/{job_id}")
def download_result(job_id: str):
    """Download the rendered video for upload-based jobs."""
    j = _jobs.get(job_id)
    if not j or j["status"] != "complete":
        return {"error": "Job not ready"}
    path = j.get("output_path")
    if not path or not os.path.exists(path):
        return {"error": "File not found"}
    return FileResponse(path, media_type="video/mp4",
                        filename=f"mcbeat_{job_id[:8]}.mp4")


def _run_upload_pipeline(
    job_id: str,
    audio_bytes: bytes,
    audio_name: str,
    clips_data: list[tuple[str, bytes]],
    export_format: str,
) -> None:
    from audio_analysis.beat_detector import analyze as detect_beats
    from clip_engine.beat_matcher import build_timeline
    from clip_engine.visual_scorer import score_clips
    from renderer.ffmpeg_builder import render

    def set_status(status: str, progress: int) -> None:
        _jobs[job_id]["status"]   = status
        _jobs[job_id]["progress"] = progress

    try:
        set_status("downloading", 10)
        with tempfile.TemporaryDirectory() as tmp:
            # Write audio
            audio_path = os.path.join(tmp, audio_name)
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            # Write clips
            clip_paths = []
            for name, data in clips_data:
                p = os.path.join(tmp, name)
                with open(p, "wb") as f:
                    f.write(data)
                clip_paths.append(p)

            set_status("analyzing", 30)
            beat_map = detect_beats(audio_path)

            set_status("matching", 50)
            clip_scores = score_clips(clip_paths)
            timeline    = build_timeline(beat_map, clip_scores)

            set_status("rendering", 65)
            # Output to a persistent temp file (survives TemporaryDirectory)
            out_dir  = tempfile.mkdtemp(prefix="mcbeat_out_")
            out_path = os.path.join(out_dir, f"mcbeat_{job_id[:8]}.mp4")
            render(timeline, audio_path, out_path, export_format=export_format)

        set_status("complete", 100)
        _jobs[job_id]["output_path"] = out_path
        _jobs[job_id]["output_url"]  = f"/download/{job_id}"

    except Exception as exc:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"]  = str(exc)[:500]


@app.post("/render", status_code=202)
def render_direct(req: RenderRequest):
    """
    All-in-one endpoint for the demo.
    Creates user/project/job in Supabase (server-side with service key),
    then triggers pipeline. Returns job_id for polling.
    """
    import uuid
    from supabase import create_client
    db = create_client(_SUPABASE_URL, _SUPABASE_KEY)

    # Upsert demo user
    user_res = db.table("users").upsert(
        {"email": req.user_email, "plan": "trial"},
        on_conflict="email"
    ).execute()
    user_id = user_res.data[0]["id"] if user_res.data else str(uuid.uuid4())

    # Create project
    proj_res = db.table("projects").insert({
        "user_id": user_id,
        "title": "Demo Render",
        "drive_folder_url": req.drive_url,
        "export_format": req.export_format,
    }).execute()
    project_id = proj_res.data[0]["id"]

    # Create job
    job_res = db.table("jobs").insert({
        "project_id": project_id,
        "status": "queued",
    }).execute()
    job_id = job_res.data[0]["id"]

    t = threading.Thread(
        target=run_pipeline,
        args=(job_id, project_id),
        daemon=True,
    )
    t.start()
    return {"job_id": job_id, "status": "accepted"}


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
    from renderer.drive_handler import download_folder
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

        with tempfile.TemporaryDirectory() as tmp:
            # gdown downloads entire folder — no API key needed
            assets = download_folder(drive_url, tmp)
            audio_path = assets["audio"]
            clip_paths = assets["videos"]

            set_status("analyzing", 30)
            beat_map = detect_beats(audio_path)

            set_status("matching", 50)
            clip_scores = score_clips(clip_paths)
            timeline    = build_timeline(beat_map, clip_scores)

            set_status("rendering", 65)
            output_path = f"{tmp}/mcbeat_{job_id}.mp4"
            render(timeline, audio_path, output_path, export_format=export_format)

            set_status("uploading", 90)
            # Output file URL — job result stored in Supabase
            output_drive_url = drive_url  # folder link (upload not implemented yet)

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

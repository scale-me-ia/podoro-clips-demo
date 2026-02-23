#!/usr/bin/env python3
"""
clip_worker.py — Podoro Clips Mac Mini Worker
================================================
Polls Supabase for pending clip pipeline runs and processes them.

Usage:
    python3 clip_worker.py              # Daemon mode (poll every 60s)
    python3 clip_worker.py --once       # Process one pending run and exit
    python3 clip_worker.py --run-id ID  # Process a specific run by ID
    python3 clip_worker.py --interval N # Poll interval in seconds (default: 60)

Environment:
    SUPABASE_URL            - Supabase project URL
    SUPABASE_SERVICE_ROLE_KEY - Service role key
    OPENAI_API_KEY          - For Whisper transcription
    ANTHROPIC_API_KEY       - For Claude highlight detection
    PIPELINE_SCRIPT         - Path to pipeline_v2.py (optional, auto-detected)
"""

import argparse
import json
import os
import subprocess
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests  # pip install requests

# ─── Config ───────────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://agypzrkevayucfvmawee.supabase.co")
SUPABASE_SERVICE_ROLE = os.environ.get(
    "SUPABASE_SERVICE_ROLE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFneXB6cmtldmF5dWNmdm1hd2VlIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTI0ODU5NSwiZXhwIjoyMDg2ODI0NTk1fQ.4RrzdjMAfwbef2dBT1U0P5Ny2soaLNtwG67sBxUeV9o"
)
SCRIPT_DIR = Path(__file__).parent
PIPELINE_SCRIPT = os.environ.get("PIPELINE_SCRIPT", str(SCRIPT_DIR / "pipeline_v2.py"))
OUTPUT_DIR = SCRIPT_DIR / "output_worker"
STORAGE_BUCKET = "clips"
POLL_INTERVAL = 60  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("clip_worker")

# ─── Supabase helpers ─────────────────────────────────────────────────────────

def supa_headers():
    return {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE}",
        "apikey": SUPABASE_SERVICE_ROLE,
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

def supa_get(table, params=None):
    r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=supa_headers(), params=params)
    r.raise_for_status()
    return r.json()

def supa_post(table, data):
    r = requests.post(f"{SUPABASE_URL}/rest/v1/{table}", headers=supa_headers(), json=data)
    r.raise_for_status()
    return r.json()

def supa_patch(table, row_id, data):
    headers = {**supa_headers(), "Prefer": "return=representation"}
    r = requests.patch(
        f"{SUPABASE_URL}/rest/v1/{table}?id=eq.{row_id}",
        headers=headers, json=data
    )
    r.raise_for_status()
    return r.json()

def upload_to_storage(bucket, path, file_path, content_type="video/mp4"):
    """Upload a file to Supabase Storage. Returns the public URL."""
    with open(file_path, "rb") as f:
        data = f.read()
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE}",
        "Content-Type": content_type,
    }
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}"
    r = requests.post(url, headers=headers, data=data)
    if r.status_code == 409:
        # Already exists — try upsert
        r = requests.put(url, headers=headers, data=data)
    r.raise_for_status()
    # Return public URL
    return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"

# ─── Worker ───────────────────────────────────────────────────────────────────

def log_step(run_id, clip_id, step, status, details=None, error=None, duration_ms=None):
    """Insert a log entry in clip_pipeline_logs."""
    try:
        supa_post("clip_pipeline_logs", {
            "run_id": run_id,
            "clip_id": clip_id,
            "step": step,
            "status": status,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "details": details or {},
            "error": error,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        log.warning(f"Failed to log step: {e}")

def update_run(run_id, status, error=None):
    """Update pipeline run status."""
    data = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if status in ("completed", "failed"):
        data["completed_at"] = datetime.now(timezone.utc).isoformat()
    if error:
        data["error"] = error
    supa_patch("clip_pipeline_runs", run_id, data)

def slugify(text):
    """Basic slug from title."""
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:50]

def process_run(run):
    """Process a single pipeline run."""
    run_id = run["id"]
    episode_id = run["episode_id"]

    log.info(f"Processing run {run_id} for episode {episode_id}")

    # Mark as transcribing (first processing state; valid states: pending/transcribing/scoring/extracting/reframing/subtitling/completed/failed)
    update_run(run_id, "transcribing")
    supa_patch("clip_pipeline_runs", run_id, {"started_at": datetime.now(timezone.utc).isoformat()})

    try:
        # Fetch episode details
        episodes = supa_get("episodes", {"id": f"eq.{episode_id}", "select": "*"})
        if not episodes:
            raise RuntimeError(f"Episode {episode_id} not found")
        episode = episodes[0]

        log.info(f"Episode: {episode['title']}")
        log_step(run_id, None, "fetch_episode", "completed", details={"title": episode["title"]})

        # Prepare output dir
        out_dir = OUTPUT_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build pipeline command
        cmd = [
            sys.executable, PIPELINE_SCRIPT,
            "--out", str(out_dir),
            "--max-clips", "3",
            "--language", "fr",
        ]

        # Determine source: YouTube URL or audio URL
        config = run.get("config") or {}
        youtube_url = config.get("youtube_url") or episode.get("youtube_url")
        audio_url = episode.get("audio_url")

        if youtube_url:
            cmd += ["--url", youtube_url]
        elif audio_url:
            # Download audio first, use --video (audio-only fallback)
            audio_path = str(out_dir / "source_audio.mp3")
            log.info(f"Downloading audio from {audio_url}")
            r = requests.get(audio_url, stream=True, timeout=300)
            r.raise_for_status()
            with open(audio_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            cmd += ["--video", audio_path, "--no-reframe"]
        else:
            raise RuntimeError("No source URL (youtube_url or audio_url) for episode")

        log_step(run_id, None, "start_pipeline", "running")

        # Run pipeline
        log.info(f"Running: {' '.join(cmd)}")
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        duration = int((time.time() - t0) * 1000)

        if result.returncode != 0:
            raise RuntimeError(f"Pipeline failed:\n{result.stderr[-500:]}")

        log.info(f"Pipeline completed in {duration/1000:.1f}s")
        log_step(run_id, None, "pipeline", "completed", duration_ms=duration)

        # Find output clips
        results_file = out_dir / "results.json"
        clips_data = []

        if results_file.exists():
            with open(results_file) as f:
                pipeline_results = json.load(f)
            clips_data = pipeline_results.get("clips", [])
        else:
            # Fallback: scan for mp4 files
            mp4_files = sorted(out_dir.glob("*.mp4"))
            for i, mp4 in enumerate(mp4_files[:5]):
                clips_data.append({
                    "index": i,
                    "title": f"Clip {i+1}",
                    "video_path": str(mp4),
                    "score": 70,
                    "start_time": 0,
                    "end_time": 60,
                })

        if not clips_data:
            raise RuntimeError("No clips generated")

        log.info(f"Generated {len(clips_data)} clips")

        # Upload clips to Storage + insert into DB
        inserted_clips = []
        for clip_info in clips_data:
            clip_index = clip_info.get("index", 0)
            video_path = clip_info.get("video_path") or clip_info.get("output_path")

            if not video_path or not Path(video_path).exists():
                log.warning(f"Clip {clip_index} video file not found: {video_path}")
                continue

            title = clip_info.get("title", f"Clip {clip_index + 1}")
            slug = slugify(title)
            storage_path = f"{episode_id}/{clip_index}_{slug}.mp4"

            # Upload to Storage
            log.info(f"Uploading clip {clip_index}: {storage_path}")
            t1 = time.time()
            video_url = upload_to_storage(STORAGE_BUCKET, storage_path, video_path)
            upload_duration = int((time.time() - t1) * 1000)

            # Insert clip row
            clip_row = {
                "episode_id": episode_id,
                "title": title,
                "video_url": video_url,
                "duration_seconds": int(clip_info.get("duration", 0) or
                                       (clip_info.get("end_time", 0) - clip_info.get("start_time", 0))),
                "score": int(clip_info.get("score", 0)),
                "clip_index": clip_index,
                "format": "9:16",
                "hook": clip_info.get("hook") or clip_info.get("title", ""),
                "transcript": clip_info.get("transcript"),
                "status": "ready",
                "start_time": int(clip_info.get("start_time", 0)),
                "end_time": int(clip_info.get("end_time", 0)),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            result_rows = supa_post("clips", clip_row)
            if isinstance(result_rows, list) and result_rows:
                clip_db = result_rows[0]
                inserted_clips.append(clip_db)
                log_step(run_id, clip_db["id"], "upload_storage", "completed",
                         details={"url": video_url}, duration_ms=upload_duration)
                log.info(f"Clip {clip_index} inserted: {clip_db['id']}")

        # Mark run completed
        update_run(run_id, "completed")
        log.info(f"Run {run_id} completed. {len(inserted_clips)} clips ready.")
        return True

    except Exception as e:
        log.error(f"Run {run_id} failed: {e}")
        update_run(run_id, "failed", error=str(e))
        log_step(run_id, None, "worker", "failed", error=str(e))
        return False

def fetch_pending_runs(run_id=None):
    """Fetch pending pipeline runs."""
    if run_id:
        return supa_get("clip_pipeline_runs", {"id": f"eq.{run_id}"})
    return supa_get("clip_pipeline_runs", {
        "status": "eq.pending",
        "order": "created_at.asc",
        "limit": "5"
    })

def main():
    parser = argparse.ArgumentParser(description="Podoro Clips Worker")
    parser.add_argument("--once", action="store_true", help="Process one run and exit")
    parser.add_argument("--run-id", help="Process a specific run by ID")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL, help="Poll interval (seconds)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Clip Worker starting. Pipeline: {PIPELINE_SCRIPT}")
    log.info(f"Supabase: {SUPABASE_URL}")

    while True:
        try:
            runs = fetch_pending_runs(args.run_id)
            if runs:
                log.info(f"Found {len(runs)} pending run(s)")
                for run in runs:
                    process_run(run)
                    if args.once or args.run_id:
                        return
            else:
                log.debug("No pending runs")
        except Exception as e:
            log.error(f"Worker error: {e}")

        if args.once or args.run_id:
            break

        log.info(f"Sleeping {args.interval}s...")
        time.sleep(args.interval)

if __name__ == "__main__":
    main()

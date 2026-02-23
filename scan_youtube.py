#!/usr/bin/env python3
"""
scan_youtube.py — Podoro YouTube Video Scanner
================================================
Scans YouTube channels for podcast episodes and matches them to Supabase episodes.
When a match is found, updates episodes.youtube_url and triggers clip generation.

Usage:
    python3 scan_youtube.py                    # Scan all podcasts with youtube_channel_id
    python3 scan_youtube.py --podcast-id UUID  # Scan a specific podcast
    python3 scan_youtube.py --days 7           # Look back N days (default: 7)
    python3 scan_youtube.py --dry-run          # Don't update DB, just print matches
    python3 scan_youtube.py --trigger-clips    # Auto-trigger clip_worker for matches
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher

import requests

# ─── Config ────────────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://agypzrkevayucfvmawee.supabase.co")
SUPABASE_SERVICE_ROLE = os.environ.get(
    "SUPABASE_SERVICE_ROLE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFneXB6cmtldmF5dWNmdm1hd2VlIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTI0ODU5NSwiZXhwIjoyMDg2ODI0NTk1fQ.4RrzdjMAfwbef2dBT1U0P5Ny2soaLNtwG67sBxUeV9o"
)
MATCH_THRESHOLD = 0.65    # fuzzy match threshold (0-1)
MAX_VIDEOS_PER_SCAN = 50  # max videos to fetch per channel per scan
LOOKBACK_DAYS = 7         # how many days back to scan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("scan_youtube")

# ─── Supabase helpers ──────────────────────────────────────────────────────────

def supa_headers():
    return {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE}",
        "apikey": SUPABASE_SERVICE_ROLE,
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

def supa_get(table, params=None):
    r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=supa_headers(), params=params or {})
    r.raise_for_status()
    return r.json()

def supa_patch(table, id_value, data):
    r = requests.patch(
        f"{SUPABASE_URL}/rest/v1/{table}?id=eq.{id_value}",
        headers=supa_headers(), json=data
    )
    r.raise_for_status()
    return r.json()

def supa_post(table, data):
    r = requests.post(f"{SUPABASE_URL}/rest/v1/{table}", headers=supa_headers(), json=data)
    r.raise_for_status()
    return r.json()

# ─── YouTube helpers ────────────────────────────────────────────────────────────

def normalize_title(title: str) -> str:
    """Normalize title for comparison: lowercase, remove special chars."""
    title = title.lower()
    title = re.sub(r'[#\[\](){}|/\\:;.,!?\'"`]', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    # Remove common podcast-specific noise
    for noise in ['podcast', 'épisode', 'episode', 'ep.', 'ep ', 'avec ', 'feat.', 'ft.', '#']:
        title = title.replace(noise, ' ')
    return re.sub(r'\s+', ' ', title).strip()

def title_similarity(t1: str, t2: str) -> float:
    """Compute fuzzy similarity between two titles (0-1)."""
    n1 = normalize_title(t1)
    n2 = normalize_title(t2)
    # SequenceMatcher ratio
    base_score = SequenceMatcher(None, n1, n2).ratio()
    # Bonus: token overlap
    tokens1 = set(n1.split())
    tokens2 = set(n2.split())
    if tokens1 and tokens2:
        overlap = len(tokens1 & tokens2) / max(len(tokens1), len(tokens2))
        # Weighted: 60% sequence ratio + 40% token overlap
        return 0.6 * base_score + 0.4 * overlap
    return base_score

def get_channel_videos(channel_url: str, max_videos: int = MAX_VIDEOS_PER_SCAN, days_back: int = LOOKBACK_DAYS) -> list:
    """Fetch recent videos from a YouTube channel using yt-dlp."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y%m%d")

    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--playlist-items", f"1:{max_videos}",
        "--print", "%(id)s\t%(title)s\t%(upload_date)s\t%(url)s",
        "--no-warnings",
        "--quiet",
        f"{channel_url}/videos"
    ]

    log.info(f"Fetching videos from {channel_url} (last {days_back} days)...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            log.warning(f"yt-dlp error: {result.stderr[:200]}")
            return []
    except subprocess.TimeoutExpired:
        log.error(f"yt-dlp timeout for {channel_url}")
        return []

    videos = []
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        vid_id, title, upload_date, url = parts[0], parts[1], parts[2], parts[3]
        # Filter by date
        if upload_date and upload_date < cutoff:
            continue
        videos.append({
            "id": vid_id,
            "title": title,
            "upload_date": upload_date,
            "url": url if url.startswith("http") else f"https://www.youtube.com/watch?v={vid_id}"
        })

    log.info(f"  Found {len(videos)} recent videos")
    return videos

def find_channel_id(channel_url: str):
    """Get the YouTube channel ID from a channel URL."""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--playlist-items", "1",
        "--print", "channel_id",
        "--quiet",
        channel_url
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        channel_id = result.stdout.strip().split('\n')[0]
        return channel_id if channel_id and channel_id.startswith('UC') else None
    except Exception as e:
        log.error(f"Could not get channel ID for {channel_url}: {e}")
        return None

# ─── Core scanner ──────────────────────────────────────────────────────────────

def scan_podcast(podcast: dict, days_back: int, dry_run: bool, trigger_clips: bool) -> dict:
    """Scan a podcast's YouTube channel and match videos to episodes."""
    podcast_id = podcast["id"]
    podcast_name = podcast["name"]
    channel_url = podcast.get("youtube_channel_url") or podcast.get("youtube_channel_id")

    if not channel_url:
        log.info(f"Skipping {podcast_name}: no youtube_channel_url")
        return {"podcast": podcast_name, "skipped": True}

    log.info(f"\n{'='*60}")
    log.info(f"Scanning: {podcast_name} ({channel_url})")

    # Fetch YouTube videos
    videos = get_channel_videos(channel_url, days_back=days_back)
    if not videos:
        return {"podcast": podcast_name, "videos_found": 0, "matches": 0}

    # Fetch episodes from DB (last N days)
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
    episodes = supa_get("episodes", {
        "podcast_id": f"eq.{podcast_id}",
        "published_at": f"gte.{cutoff_date}",
        "select": "id,title,status,youtube_url,has_video",
        "limit": "100"
    })

    if not episodes:
        log.info(f"  No recent episodes in DB for {podcast_name}")
        return {"podcast": podcast_name, "videos_found": len(videos), "matches": 0}

    log.info(f"  DB episodes to match: {len(episodes)}")

    matches = []
    for episode in episodes:
        best_video = None
        best_score = 0.0

        for video in videos:
            score = title_similarity(episode["title"], video["title"])
            if score > best_score:
                best_score = score
                best_video = video

        if best_score >= MATCH_THRESHOLD and best_video:
            log.info(f"  ✅ MATCH ({best_score:.2f}): '{episode['title'][:50]}' ↔ '{best_video['title'][:50]}'")
            matches.append({
                "episode_id": episode["id"],
                "episode_title": episode["title"],
                "video_url": best_video["url"],
                "video_title": best_video["title"],
                "score": best_score,
                "episode_status": episode["status"],
                "already_has_video": episode.get("has_video", False)
            })
        else:
            log.debug(f"  ✗ No match ({best_score:.2f}): '{episode['title'][:50]}'")

    log.info(f"  Total matches: {len(matches)}/{len(episodes)}")

    if not dry_run:
        triggered = 0
        for match in matches:
            # Update episode with youtube_url
            supa_patch("episodes", match["episode_id"], {
                "youtube_url": match["video_url"],
                "has_video": True
            })

            # Auto-trigger clip_worker if conditions met
            if (
                trigger_clips
                and match["episode_status"] == "published"
                and not match["already_has_video"]
            ):
                # Check no existing clips for this episode
                existing_runs = supa_get("clip_pipeline_runs", {
                    "episode_id": f"eq.{match['episode_id']}",
                    "status": "not.eq.failed",
                    "limit": "1"
                })
                if not existing_runs:
                    supa_post("clip_pipeline_runs", {
                        "episode_id": match["episode_id"],
                        "status": "pending",
                        "triggered_by": "youtube_scanner",
                        "config": {"auto": True, "youtube_url": match["video_url"]},
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    })
                    triggered += 1
                    log.info(f"  🎬 Triggered clip_worker for: {match['episode_title'][:50]}")

        log.info(f"  DB updated. Clip runs triggered: {triggered}")

    return {
        "podcast": podcast_name,
        "videos_found": len(videos),
        "episodes_checked": len(episodes),
        "matches": len(matches),
        "details": matches
    }

def process_scan_requests(dry_run: bool = False) -> list:
    """
    Check for pending scan_requests from the Edge Function and process them.
    Returns list of processed request IDs.
    """
    try:
        pending = supa_get("scan_requests", {
            "status": "eq.pending",
            "order": "created_at.asc",
            "limit": "10",
            "select": "id,podcast_id,days_back,trigger_clips,triggered_by"
        })
    except Exception as e:
        log.warning(f"Could not fetch scan_requests: {e}")
        return []

    if not pending:
        return []

    log.info(f"Found {len(pending)} pending scan request(s) from Edge Function")
    processed = []

    for req in pending:
        req_id = req["id"]
        log.info(f"Processing scan_request {req_id} (triggered_by={req.get('triggered_by')})")

        # Mark as processing
        if not dry_run:
            try:
                supa_patch("scan_requests", req_id, {
                    "status": "processing",
                    "processed_at": datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                log.warning(f"Could not update scan_request status: {e}")

        # Run the scan
        params = {"is_active": "eq.true", "select": "id,name,youtube_channel_id,youtube_channel_url"}
        if req.get("podcast_id"):
            params["id"] = f"eq.{req['podcast_id']}"

        try:
            podcasts = supa_get("podcasts", params)
            podcasts_with_yt = [p for p in podcasts if p.get("youtube_channel_url") or p.get("youtube_channel_id")]
            results = []
            for podcast in podcasts_with_yt:
                result = scan_podcast(podcast, req.get("days_back", LOOKBACK_DAYS), dry_run, req.get("trigger_clips", False))
                results.append(result)

            total_matches = sum(r.get("matches", 0) for r in results)

            if not dry_run:
                supa_patch("scan_requests", req_id, {
                    "status": "done",
                    "result": {"results": results, "total_matches": total_matches},
                    "processed_at": datetime.now(timezone.utc).isoformat()
                })
        except Exception as e:
            log.error(f"Error processing scan_request {req_id}: {e}")
            if not dry_run:
                try:
                    supa_patch("scan_requests", req_id, {"status": "failed"})
                except Exception:
                    pass

        processed.append(req_id)

    return processed


def main():
    parser = argparse.ArgumentParser(description="Podoro YouTube Scanner")
    parser.add_argument("--podcast-id", help="Scan a specific podcast by ID")
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS, help="Look back N days")
    parser.add_argument("--dry-run", action="store_true", help="Don't update DB")
    parser.add_argument("--trigger-clips", action="store_true", help="Auto-trigger clip_worker for matches")
    args = parser.parse_args()

    log.info(f"YouTube Scanner starting (last {args.days} days, trigger-clips={args.trigger_clips})")

    # First: process any pending scan_requests from the Edge Function
    processed_requests = process_scan_requests(dry_run=args.dry_run)
    if processed_requests:
        log.info(f"Processed {len(processed_requests)} scan_request(s) from Edge Function")

    # Fetch podcasts
    params = {"is_active": "eq.true", "select": "id,name,youtube_channel_id,youtube_channel_url"}
    if args.podcast_id:
        params["id"] = f"eq.{args.podcast_id}"

    podcasts = supa_get("podcasts", params)
    podcasts_with_yt = [p for p in podcasts if p.get("youtube_channel_url") or p.get("youtube_channel_id")]

    log.info(f"Active podcasts with YouTube: {len(podcasts_with_yt)}/{len(podcasts)}")

    results = []
    for podcast in podcasts_with_yt:
        result = scan_podcast(podcast, args.days, args.dry_run, args.trigger_clips)
        results.append(result)

    # Summary
    total_matches = sum(r.get("matches", 0) for r in results)
    total_videos = sum(r.get("videos_found", 0) for r in results)
    log.info(f"\n{'='*60}")
    log.info(f"SCAN COMPLETE: {total_matches} matches found across {total_videos} videos")
    if args.dry_run:
        log.info("DRY RUN — no DB changes made")

    # Output JSON summary
    print(json.dumps({
        "scan_date": datetime.now(timezone.utc).isoformat(),
        "days_back": args.days,
        "dry_run": args.dry_run,
        "trigger_clips": args.trigger_clips,
        "results": results,
        "total_matches": total_matches,
        "total_videos": total_videos
    }, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

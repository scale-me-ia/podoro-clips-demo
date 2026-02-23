#!/usr/bin/env python3
"""
seed_youtube_channels.py — Seed YouTube channel IDs into podcasts table
=======================================================================
Verifies each YouTube channel via yt-dlp and updates Supabase podcasts table.

Usage:
    python3 seed_youtube_channels.py           # Seed all known channels
    python3 seed_youtube_channels.py --dry-run # Print what would be updated
    python3 seed_youtube_channels.py --verify  # Re-verify all channel IDs
"""

import argparse
import json
import logging
import subprocess
import sys

import requests

# ─── Config ────────────────────────────────────────────────────────────────────

SUPABASE_URL = "https://agypzrkevayucfvmawee.supabase.co"
SUPABASE_SERVICE_ROLE = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFneXB6cmtldmF5dWNmdm1hd2VlIiwicm9sZSI6"
    "InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTI0ODU5NSwiZXhwIjoyMDg2ODI0NTk1fQ"
    ".4RrzdjMAfwbef2dBT1U0P5Ny2soaLNtwG67sBxUeV9o"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("seed_youtube")

# ─── Channel mapping ──────────────────────────────────────────────────────────
# Format: (podcast_id, podcast_name, youtube_channel_url, verified_channel_id)
# channel_id verified via yt-dlp; None = channel not found on YouTube
CHANNEL_MAP = [
    (
        "05dc632a-ba1c-4551-824b-67ecb01949fe",
        "Le Gratin",
        None,  # No YouTube channel found
        None,
    ),
    (
        "a2b74f12-9efb-491b-86d8-b48d4b395749",
        "La Martingale",
        "https://www.youtube.com/@LaMartingale",
        "UCnmnyIcDZfWwzQQhABB700w",
    ),
    (
        "3c5acae8-d94b-4865-aebf-fcd68305d65c",
        "Tribu Indé",
        "https://www.youtube.com/@AlexisdeTribuIndé",  # Real handle (not @TribuInde)
        "UCkpP1hV5qR0RkpfkO9KsPVw",
    ),
    (
        "61b097b0-e8d9-45ad-b2b9-5da7f8dab2f8",
        "Marketing Mania",
        "https://www.youtube.com/@MarketingMania",
        "UCSmUdD2Dd_v5uqBuRwtEZug",
    ),
    (
        "96cd1eb6-87de-4039-95d5-a406fd295eb1",
        "Génération Do It Yourself",
        "https://www.youtube.com/@GenerationDoItYourself",
        "UCQ1ffXBtmuSbSxmaM363-eA",
    ),
    (
        "99bac2ab-9108-47d7-b2c2-2e581263b56e",
        "Sans Permission",
        "https://www.youtube.com/@SansPermission",
        "UCr9axFlY2sW2RbUfQ7v7qEw",
    ),
    (
        "bd815d4c-fe88-4802-9a56-84998d084d59",
        "2 Heures de Perdues",
        "https://www.youtube.com/@2heuresdeperdues",
        "UCZAP6dC-Rmgtp7BEhUa4ZjQ",
    ),
    (
        "b4b18f61-6bfc-480b-97ff-51f11503c408",
        "L'Envolée",
        None,  # No YouTube channel found
        None,
    ),
    (
        "f48b7abb-f122-40bf-a2fe-ef6691c9a412",
        "Little Big Things",
        "https://www.youtube.com/@littlebigthings_",
        "UCuRlsHM2bTRMCUqhWv1K4Gw",
    ),
    (
        "32ae169e-a73c-423b-9144-160ed6b66361",
        "Le Panier",
        "https://www.youtube.com/@LePanier",
        "UCRXKISlvSrOVT2vHenAYRag",
    ),
    (
        "d7cd564c-57ec-4aaa-8a79-ea8bee43cf81",
        "TheBBoost",
        "https://www.youtube.com/channel/UC5Yn4FOHIRuezaC55Dn1yTQ",  # Aline de TheBBoost (real channel)
        "UC5Yn4FOHIRuezaC55Dn1yTQ",
    ),
    (
        "bb747291-6a72-455e-b626-ba518ceae22b",
        "Serial Entrepreneurs",
        "https://www.youtube.com/@serialentrepreneurs",
        "UCAWEcoIqloo4rlwT8lQQwvQ",
    ),
]

# ─── Supabase helpers ──────────────────────────────────────────────────────────

def supa_headers():
    return {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE}",
        "apikey": SUPABASE_SERVICE_ROLE,
        "Content-Type": "application/json",
    }

def supa_patch(podcast_id: str, data: dict) -> bool:
    r = requests.patch(
        f"{SUPABASE_URL}/rest/v1/podcasts?id=eq.{podcast_id}",
        headers=supa_headers(),
        json=data
    )
    if r.status_code in (200, 204):
        return True
    log.error(f"Supabase PATCH failed: {r.status_code} — {r.text[:200]}")
    return False

# ─── yt-dlp helpers ───────────────────────────────────────────────────────────

def verify_channel(channel_url: str) -> tuple:
    """
    Verify a YouTube channel exists and get its channel_id.
    Returns (channel_id, channel_name) or (None, None).
    """
    cmd = [
        "yt-dlp",
        f"{channel_url}/videos",
        "--playlist-items", "1",
        "--dump-json",
        "--quiet",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            import json as _json
            d = _json.loads(line)
            cid = d.get("channel_id")
            cname = d.get("channel") or d.get("uploader")
            return cid, cname
    except Exception as e:
        log.warning(f"verify_channel error for {channel_url}: {e}")
    return None, None

def search_channel(query: str) -> tuple:
    """
    Search YouTube for a channel and return (channel_id, channel_name, channel_url).
    """
    cmd = [
        "yt-dlp",
        f"ytsearch3:{query}",
        "--dump-json",
        "--flat-playlist",
        "--quiet",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        import json as _json
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            d = _json.loads(line)
            cid = d.get("channel_id")
            cname = d.get("channel")
            curl = d.get("channel_url")
            if cid:
                return cid, cname, curl
    except Exception as e:
        log.warning(f"search_channel error for '{query}': {e}")
    return None, None, None

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seed YouTube channels into Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Don't update DB")
    parser.add_argument("--verify", action="store_true", help="Re-verify all channels via yt-dlp")
    args = parser.parse_args()

    log.info("=== Podoro YouTube Channel Seeder ===")
    if args.dry_run:
        log.info("DRY RUN mode — no DB changes")

    seeded = 0
    skipped = 0
    results = []

    for podcast_id, name, channel_url, channel_id in CHANNEL_MAP:
        if not channel_url:
            log.info(f"⚪ {name}: no YouTube channel mapped")
            skipped += 1
            results.append({"podcast": name, "status": "no_channel"})
            continue

        verified_id = channel_id

        if args.verify or not channel_id:
            log.info(f"🔍 Verifying {name} → {channel_url}")
            vid, vname = verify_channel(channel_url)
            if vid:
                verified_id = vid
                log.info(f"  ✅ Verified: {vname} ({vid})")
            else:
                log.warning(f"  ⚠️  Could not verify {channel_url}")

        if not verified_id:
            log.warning(f"⚠️  {name}: channel_id unknown, skipping DB update")
            results.append({"podcast": name, "status": "unverified", "url": channel_url})
            skipped += 1
            continue

        data = {
            "youtube_channel_id": verified_id,
            "youtube_channel_url": channel_url,
        }
        log.info(f"{'[DRY RUN] ' if args.dry_run else ''}✅ {name}: {channel_url} ({verified_id})")

        if not args.dry_run:
            ok = supa_patch(podcast_id, data)
            if ok:
                seeded += 1
                results.append({"podcast": name, "status": "seeded", "channel_id": verified_id, "url": channel_url})
            else:
                results.append({"podcast": name, "status": "error", "url": channel_url})
        else:
            seeded += 1
            results.append({"podcast": name, "status": "would_seed", "channel_id": verified_id, "url": channel_url})

    log.info(f"\n=== DONE: {seeded} seeded, {skipped} skipped ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import json
    main()

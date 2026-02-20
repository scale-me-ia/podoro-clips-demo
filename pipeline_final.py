#!/usr/bin/env python3
"""
Podoro Clips Pipeline — Phase 4 Final
======================================
Usage:
    python3 pipeline_final.py --video VIDEO.mp4 --vtt SUBS.vtt --out OUTPUT_DIR

Steps:
    1. Extract audio (mp3) from video
    2. Run Gemini Flash highlight detection (audio-based)
    3. Run Claude highlight detection (VTT text-based)
    4. Merge + deduplicate highlights (top 5)
    5. For each highlight: extract → reframe B2 (MediaPipe) → subtitles C1 (PIL karaoke)
    6. Return list of clips with scores

Requirements:
    pip install anthropic google-generativeai opencv-python mediapipe pillow pydub ffmpeg-python
    ffmpeg installed on PATH
    ANTHROPIC_API_KEY env var or --anthropic-key arg
    GOOGLE_API_KEY env var or --gemini-key arg

Cost estimate: ~$0.17/episode (Gemini $0.02 + Claude $0.15)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

REFRAME_SCRIPT = os.path.join(os.path.dirname(__file__), "reframe_v2.py")
SUBTITLES_SCRIPT = os.path.join(os.path.dirname(__file__), "subtitles.py")

# Fallback script paths (from podoro-clips-v3)
REFRAME_FALLBACK = "/tmp/podoro-clips-v3/agent-cadrage/reframe_v2.py"
SUBTITLES_FALLBACK = "/tmp/podoro-clips-v3/agent-subtitles/subtitles.py"

FACE_MODEL = "/tmp/blaze_face_short_range.tflite"
FONT_PATH = "/tmp/fonts/TypoldExtended-ExtraBold.otf"

MIN_CLIP_DURATION = 20  # seconds — skip clips shorter than this
MAX_CLIPS = 5           # top N clips to process


# ──────────────────────────────────────────────────────────────────────────────
# VTT Parser
# ──────────────────────────────────────────────────────────────────────────────

def ts2s(ts_str: str) -> float:
    """Convert VTT timestamp to seconds."""
    ts_str = ts_str.strip().split()[0]
    m = re.match(r'(\d{2}):(\d{2}):(\d{2})\.(\d+)', ts_str)
    if m:
        return int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3)) + int(m.group(4))/1000
    return 0.0


def extract_vtt_text(vtt_path: str, start_s: float, end_s: float) -> str:
    """Extract clean deduplicated text from VTT for a time range."""
    with open(vtt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    texts = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '-->' in line:
            parts = line.split('-->')
            if len(parts) == 2:
                ts_start = ts2s(parts[0])
                ts_end = ts2s(parts[1])

                if ts_start < start_s - 1 or ts_start > end_s + 1:
                    i += 1
                    continue

                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                    text_lines.append(lines[i].rstrip())
                    i += 1

                # "Separator" entries = clean single line, no timing tags, tiny span
                is_separator = (
                    ts_end - ts_start < 0.02
                    and len(text_lines) == 1
                    and '<' not in text_lines[0]
                )
                if is_separator:
                    t = text_lines[0].strip()
                    if t and t != ' ':
                        texts.append(t)
                continue
        i += 1

    return ' '.join(texts)


def parse_full_vtt(vtt_path: str):
    """Parse full VTT and return list of (start_s, text) tuples."""
    with open(vtt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '-->' in line:
            parts = line.split('-->')
            if len(parts) == 2:
                ts_start = ts2s(parts[0])
                ts_end = ts2s(parts[1])

                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                    text_lines.append(lines[i].rstrip())
                    i += 1

                is_separator = (
                    ts_end - ts_start < 0.02
                    and len(text_lines) == 1
                    and '<' not in text_lines[0]
                )
                if is_separator:
                    t = text_lines[0].strip()
                    if t and t != ' ':
                        entries.append((ts_start, t))
                continue
        i += 1
    return entries


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Extract Audio
# ──────────────────────────────────────────────────────────────────────────────

def extract_audio(video_path: str, out_dir: str) -> str:
    """Extract MP3 audio from video for Gemini processing."""
    audio_path = os.path.join(out_dir, "podcast_audio.mp3")
    if os.path.exists(audio_path):
        print(f"  [audio] Already exists: {audio_path}")
        return audio_path

    print(f"  [audio] Extracting audio from {video_path}...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "libmp3lame", "-ab", "128k",
        audio_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    size = os.path.getsize(audio_path) / 1024 / 1024
    print(f"  [audio] Done: {audio_path} ({size:.1f} MB)")
    return audio_path


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Gemini Highlight Detection
# ──────────────────────────────────────────────────────────────────────────────

GEMINI_PROMPT = """Tu es un expert en contenu viral pour les réseaux sociaux (TikTok/Instagram Reels).
Analyse ce podcast en français et trouve les 5 meilleurs passages pour créer des clips courts viraux.

Pour chaque passage, retourne un JSON avec :
- start_time: "HH:MM:SS" (début)
- end_time: "HH:MM:SS" (fin)
- duration_seconds: durée en secondes (20-90s max)
- hook_text: la phrase d'accroche
- punchline: la fin percutante
- why_viral: pourquoi ça va buzzer
- viral_score: score /100

Retourne UNIQUEMENT un JSON valide : {"highlights": [...]}"""


def detect_highlights_gemini(audio_path: str, gemini_key: str) -> list:
    """Run Gemini Flash highlight detection on audio."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("  [gemini] ⚠️  google-generativeai not installed, skipping")
        return []

    print("  [gemini] Uploading audio and running detection...")
    genai.configure(api_key=gemini_key)

    # Upload audio file
    audio_file = genai.upload_file(audio_path, mime_type="audio/mpeg")

    # Wait for processing
    while audio_file.state.name == "PROCESSING":
        time.sleep(5)
        audio_file = genai.get_file(audio_file.name)

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content([GEMINI_PROMPT, audio_file])

    raw = response.text.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        data = json.loads(raw)
        highlights = data.get("highlights", [])
        print(f"  [gemini] Found {len(highlights)} highlights")
        return highlights
    except json.JSONDecodeError as e:
        print(f"  [gemini] ⚠️  Parse error: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Claude Highlight Detection (VTT-based)
# ──────────────────────────────────────────────────────────────────────────────

CLAUDE_HIGHLIGHT_PROMPT = """Tu es un expert en contenu viral pour les réseaux sociaux.
Voici la transcription d'un podcast en français (avec timestamps en secondes).

Identifie les 5 meilleurs passages pour des clips courts viraux (20-90 secondes).
Critères : hook fort, arc narratif, émotion, punchline mémorable, autonomie.

Transcription (format "[Xs] texte") :
{transcript}

Retourne UNIQUEMENT un JSON valide :
{{"highlights": [
  {{
    "start_s": <float>,
    "end_s": <float>,
    "duration_seconds": <int>,
    "hook_text": "<accroche>",
    "punchline": "<fin percutante>",
    "why_viral": "<raison>",
    "viral_score": <int 0-100>
  }}
]}}"""


def detect_highlights_claude(vtt_path: str, anthropic_key: str) -> list:
    """Run Claude highlight detection on VTT transcription."""
    try:
        import anthropic
    except ImportError:
        print("  [claude] ⚠️  anthropic not installed, skipping")
        return []

    print("  [claude] Parsing VTT and running detection...")
    entries = parse_full_vtt(vtt_path)

    # Build transcript with timestamps (sample every 5th entry to stay in token budget)
    transcript_lines = []
    for i, (ts, text) in enumerate(entries):
        if i % 5 == 0:  # Sample to reduce tokens
            mins = int(ts // 60)
            secs = int(ts % 60)
            transcript_lines.append(f"[{int(ts)}s / {mins}:{secs:02d}] {text}")

    transcript = '\n'.join(transcript_lines[:500])  # Max 500 entries
    prompt = CLAUDE_HIGHLIGHT_PROMPT.format(transcript=transcript)

    client = anthropic.Anthropic(api_key=anthropic_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        data = json.loads(raw)
        highlights = data.get("highlights", [])
        print(f"  [claude] Found {len(highlights)} highlights")
        return highlights
    except json.JSONDecodeError as e:
        print(f"  [claude] ⚠️  Parse error: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Score & Merge Highlights
# ──────────────────────────────────────────────────────────────────────────────

SCORE_PROMPT = """Analyse ce passage de podcast en français et score-le pour le potentiel viral.

Passage ({duration}s) :
{text}

Retourne UNIQUEMENT un JSON valide :
{{
  "hook": <0-30>,
  "arc": <0-25>,
  "emotion": <0-25>,
  "punchline": <0-20>,
  "autonomie": <0 ou 5>,
  "total": <somme>,
  "verdict": "<1 phrase>"
}}"""


def score_passage(text: str, duration: int, client) -> dict:
    """Score a passage with Claude."""
    prompt = SCORE_PROMPT.format(text=text[:1500], duration=duration)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    try:
        return json.loads(raw)
    except Exception:
        return {"total": 0, "verdict": "parse_error"}


def normalize_highlight(h: dict, source: str) -> dict:
    """Normalize a highlight to a standard format."""
    # Handle both Gemini (start_time: "HH:MM:SS") and Claude (start_s: float) formats
    if "start_time" in h:
        # Gemini format
        def hhmmss2s(t):
            parts = t.split(":")
            return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
        start_s = hhmmss2s(h["start_time"])
        end_s = hhmmss2s(h["end_time"])
    else:
        start_s = float(h.get("start_s", 0))
        end_s = float(h.get("end_s", 0))

    return {
        "start_s": start_s,
        "end_s": end_s,
        "duration": int(end_s - start_s),
        "hook_text": h.get("hook_text", ""),
        "punchline": h.get("punchline", ""),
        "why_viral": h.get("why_viral", ""),
        "source_score": h.get("viral_score", 0),
        "source": source,
    }


def merge_highlights(gemini_hits: list, claude_hits: list, vtt_path: str, anthropic_key: str) -> list:
    """Merge, deduplicate, score, and rank all highlights."""
    import anthropic
    client = anthropic.Anthropic(api_key=anthropic_key)

    all_hits = []
    for h in gemini_hits:
        all_hits.append(normalize_highlight(h, "gemini"))
    for h in claude_hits:
        all_hits.append(normalize_highlight(h, "claude"))

    # Deduplicate: remove passages within 30s of each other
    deduped = []
    for h in sorted(all_hits, key=lambda x: x["start_s"]):
        overlap = False
        for existing in deduped:
            if abs(h["start_s"] - existing["start_s"]) < 30:
                overlap = True
                # Keep the one with higher source score
                if h["source_score"] > existing["source_score"]:
                    deduped.remove(existing)
                    deduped.append(h)
                break
        if not overlap:
            deduped.append(h)

    # Score each passage with Claude
    print(f"  [merge] Scoring {len(deduped)} deduplicated passages...")
    for h in deduped:
        text = extract_vtt_text(vtt_path, h["start_s"], h["end_s"])
        h["text"] = text
        if len(text) > 50:
            scores = score_passage(text, h["duration"], client)
            h["claude_score"] = scores.get("total", 0)
            h["verdict"] = scores.get("verdict", "")
        else:
            h["claude_score"] = 0
            h["verdict"] = "text_too_short"

    # Sort by Claude score
    deduped.sort(key=lambda x: x["claude_score"], reverse=True)
    print(f"  [merge] Top 5 passages:")
    for i, h in enumerate(deduped[:5]):
        mins = int(h["start_s"] // 60)
        secs = int(h["start_s"] % 60)
        print(f"    {i+1}. {mins}:{secs:02d} ({h['duration']}s) — Claude: {h['claude_score']}/100 [{h['source']}]")

    return deduped[:MAX_CLIPS]


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Produce Clips
# ──────────────────────────────────────────────────────────────────────────────

def find_script(name: str, fallback: str) -> str:
    """Find a script by name, falling back to known path."""
    local = os.path.join(os.path.dirname(__file__), name)
    if os.path.exists(local):
        return local
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Script not found: {name} (tried {local} and {fallback})")


def produce_clip(highlight: dict, video_path: str, vtt_path: str, out_dir: str, clip_num: int) -> str | None:
    """Extract → reframe B2 → subtitles C1 for one highlight."""
    start_s = highlight["start_s"]
    duration = highlight["duration"]

    if duration < MIN_CLIP_DURATION:
        print(f"  [clip {clip_num}] ⚠️  Too short ({duration}s < {MIN_CLIP_DURATION}s), skipping")
        return None

    tmp_dir = os.path.join(out_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    raw_path = os.path.join(tmp_dir, f"clip{clip_num}_raw.mp4")
    b2_path = os.path.join(tmp_dir, f"clip{clip_num}_b2.mp4")
    final_path = os.path.join(out_dir, f"final_{clip_num}.mp4")

    mins = int(start_s // 60)
    secs = int(start_s % 60)
    print(f"\n  [clip {clip_num}] {mins}:{secs:02d} ({duration}s) — {highlight.get('hook_text', '')[:50]}...")

    # 1. Extract raw clip
    print(f"  [clip {clip_num}] Extracting...")
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start_s), "-i", video_path,
        "-t", str(duration), "-c", "copy", raw_path
    ], check=True, capture_output=True)

    # 2. Reframe B2
    print(f"  [clip {clip_num}] Reframing (MediaPipe B2)...")
    reframe_script = find_script("reframe_v2.py", REFRAME_FALLBACK)
    result = subprocess.run(
        [sys.executable, reframe_script, raw_path, b2_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [clip {clip_num}] ⚠️  Reframe failed: {result.stderr[-200:]}")
        return None

    # 3. Subtitles C1 (PIL karaoke)
    print(f"  [clip {clip_num}] Adding subtitles (PIL karaoke C1)...")
    subs_script = find_script("subtitles.py", SUBTITLES_FALLBACK)
    offset = -start_s
    result = subprocess.run([
        sys.executable, subs_script,
        "--video", b2_path,
        "--vtt", vtt_path,
        "--output", final_path,
        "--offset", str(offset),
        "--crop", "1080x1920"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [clip {clip_num}] ⚠️  Subtitles failed: {result.stderr[-200:]}")
        return None

    # Validate
    probe = subprocess.run([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", final_path
    ], capture_output=True, text=True)
    actual_dur = float(probe.stdout.strip() or "0")

    if actual_dur < MIN_CLIP_DURATION:
        print(f"  [clip {clip_num}] ⚠️  Output too short ({actual_dur:.1f}s), skipping")
        return None

    size = os.path.getsize(final_path) / 1024 / 1024
    print(f"  [clip {clip_num}] ✅  {final_path} ({actual_dur:.1f}s, {size:.1f}MB)")
    return final_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Podoro Clips Pipeline — Phase 4 Final",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--video", required=True, help="Source video file (MP4)")
    parser.add_argument("--vtt", required=True, help="VTT transcription file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--anthropic-key", default=os.environ.get("ANTHROPIC_API_KEY", ""),
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--gemini-key", default=os.environ.get("GOOGLE_API_KEY", ""),
                        help="Google API key for Gemini (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--skip-gemini", action="store_true", help="Skip Gemini detection")
    parser.add_argument("--skip-claude-detect", action="store_true",
                        help="Skip Claude detection (use only Gemini)")
    parser.add_argument("--max-clips", type=int, default=MAX_CLIPS, help="Max clips to generate")

    args = parser.parse_args()

    if not args.anthropic_key:
        print("❌  ANTHROPIC_API_KEY not set. Use --anthropic-key or set env var.")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Podoro Clips Pipeline v1")
    print(f"  Video:  {args.video}")
    print(f"  VTT:    {args.vtt}")
    print(f"  Output: {args.out}")
    print(f"{'='*60}\n")

    # Step 1: Extract audio
    print("Step 1: Extracting audio...")
    audio_path = extract_audio(args.video, args.out)

    # Step 2+3: Detect highlights in parallel
    gemini_hits = []
    claude_hits = []

    print("\nStep 2+3: Detecting highlights (parallel)...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}

        if not args.skip_gemini and args.gemini_key:
            futures["gemini"] = executor.submit(
                detect_highlights_gemini, audio_path, args.gemini_key
            )
        else:
            print("  [gemini] Skipped (no key or --skip-gemini)")

        if not args.skip_claude_detect:
            futures["claude"] = executor.submit(
                detect_highlights_claude, args.vtt, args.anthropic_key
            )
        else:
            print("  [claude] Skipped (--skip-claude-detect)")

        for name, future in futures.items():
            try:
                hits = future.result()
                if name == "gemini":
                    gemini_hits = hits
                else:
                    claude_hits = hits
            except Exception as e:
                print(f"  [{name}] ❌  Error: {e}")

    # Step 4: Merge, deduplicate, score
    print("\nStep 4: Merging and scoring highlights...")
    top_highlights = merge_highlights(
        gemini_hits, claude_hits, args.vtt, args.anthropic_key
    )

    if not top_highlights:
        print("❌  No highlights found! Check your API keys.")
        sys.exit(1)

    # Step 5: Produce clips
    print(f"\nStep 5: Producing top {min(args.max_clips, len(top_highlights))} clips...")
    output_clips = []
    results_data = []

    for i, highlight in enumerate(top_highlights[:args.max_clips], 1):
        clip_path = produce_clip(highlight, args.video, args.vtt, args.out, i)
        if clip_path:
            output_clips.append(clip_path)
            results_data.append({
                "clip": clip_path,
                "rank": i,
                "start_s": highlight["start_s"],
                "duration": highlight["duration"],
                "claude_score": highlight.get("claude_score", 0),
                "source": highlight.get("source", "unknown"),
                "hook_text": highlight.get("hook_text", ""),
                "verdict": highlight.get("verdict", ""),
            })

    # Save results
    results_path = os.path.join(args.out, "pipeline_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅  Pipeline complete!")
    print(f"  Generated {len(output_clips)}/{min(args.max_clips, len(top_highlights))} clips")
    for clip in output_clips:
        print(f"  📹  {clip}")
    print(f"  📊  {results_path}")
    print(f"{'='*60}\n")

    return output_clips


if __name__ == "__main__":
    main()

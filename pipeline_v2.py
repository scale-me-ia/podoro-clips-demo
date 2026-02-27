#!/usr/bin/env python3
"""
pipeline_v2.py — Podoro Clips Full E2E Pipeline v2 (Rush 4)
=============================================================
Usage:
    python3 pipeline_v2.py --url "https://youtube.com/watch?v=..."
    python3 pipeline_v2.py --video ./podcast.mp4 --out ./output/
    python3 pipeline_v2.py --video ./podcast.mp4 --max-clips 5 --min-score 60
    python3 pipeline_v2.py --video ./podcast.mp4 --skip-subtitles --no-diarization

Options:
    --url URL              YouTube URL to download (requires yt-dlp)
    --video PATH           Local video file (mp4/mkv/webm)
    --whisper-json PATH    Pre-existing Whisper JSON (skip transcription step)
    --out DIR              Output directory (default: ./output)
    --max-clips N          Maximum clips to produce (default: 3)
    --min-score N          Minimum score 1-10 to include clip (default: 0)
    --language LANG        Whisper language code (default: fr)
    --dry-run              Detect highlights only, skip video processing
    --no-reframe           Skip reframing step (use raw clip)
    --no-subs              Skip subtitles step
    --skip-subtitles       Alias for --no-subs
    --no-diarization       Disable speaker diarization (default: already off; for compatibility)
    --anthropic-key KEY    Anthropic API key
    --openai-key KEY       OpenAI API key (Whisper)

Pipeline:
    Step 1 — Download (yt-dlp) or use local file
    Step 2 — Whisper API transcription (word-level timestamps)
    Step 3 — Claude highlight detection (word-level precise cuts)
    Step 4 — Dynamic window expansion (min 45s, max 90s, complete arc)
    Step 5 — For each top clip: extract → reframe_v3.py → subtitles_v3.py
    Step 6 — Output JSON + summary

Output: output/{podcast_name}/clip_{N}_{score}.mp4 + results.json

Environment variables:
    ANTHROPIC_API_KEY      Claude API key
    OPENAI_API_KEY         OpenAI Whisper API key
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    class tqdm:  # noqa: F811
        """Minimal tqdm stub when package is not installed."""
        def __init__(self, iterable=None, **kwargs):
            self._it = iterable
            desc = kwargs.get("desc", "")
            total = kwargs.get("total", "?")
            if desc:
                print(f"  [{desc}] (0/{total})")
        def __iter__(self):
            return iter(self._it or [])
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): pass
        def set_postfix_str(self, s=""): pass
        def set_description(self, s=""): pass
        @staticmethod
        def write(s): print(s)


# ──────────────────────────────────────────────────────────────────────────────
# Retry helper
# ──────────────────────────────────────────────────────────────────────────────

def with_retry(fn, *args, retries: int = 3, base_delay: float = 2.0, label: str = "", **kwargs):
    """Call fn(*args, **kwargs) up to `retries` times with exponential backoff."""
    last_exc = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                wait = base_delay * (2 ** attempt)
                tag = f"[{label}] " if label else ""
                print(f"  ⚠️  {tag}Attempt {attempt+1}/{retries} failed: {e}. Retrying in {wait:.0f}s...")
                time.sleep(wait)
    raise last_exc

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFRAME_SCRIPT = os.path.join(SCRIPT_DIR, "scripts", "reframe_v3.py")
SUBTITLES_SCRIPT = os.path.join(SCRIPT_DIR, "scripts", "subtitles_v3.py")

MIN_CLIP_DURATION = 45   # seconds
MAX_CLIP_DURATION = 90   # seconds
TARGET_CLIP_DURATION = 65  # ideal clip length
DEFAULT_MAX_CLIPS = 3
DEFAULT_MIN_SCORE = 0

# Cost tracking (approximate)
COST_TRACKER = {
    "whisper_minutes": 0.0,
    "claude_input_tokens": 0,
    "claude_output_tokens": 0,
}

def estimate_cost() -> float:
    """Estimate total API cost in USD."""
    # Whisper: $0.006/minute
    whisper_cost = COST_TRACKER["whisper_minutes"] * 0.006
    # Claude Sonnet: $3/M input, $15/M output
    claude_cost = (
        COST_TRACKER["claude_input_tokens"] / 1_000_000 * 3.0
        + COST_TRACKER["claude_output_tokens"] / 1_000_000 * 15.0
    )
    return round(whisper_cost + claude_cost, 4)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Download or validate local file
# ──────────────────────────────────────────────────────────────────────────────

def download_youtube(url: str, out_dir: str) -> str:
    """Download YouTube video using yt-dlp. Returns path to downloaded file."""
    print(f"\n▶ Step 1/6: Downloading YouTube video...")
    print(f"  URL: {url}")

    os.makedirs(out_dir, exist_ok=True)
    output_template = os.path.join(out_dir, "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ yt-dlp failed:\n{result.stderr[-500:]}")
        raise RuntimeError(f"yt-dlp failed: {result.stderr[-200:]}")

    # Find downloaded file (last modified .mp4 in out_dir)
    mp4_files = sorted(
        Path(out_dir).glob("*.mp4"),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )
    if not mp4_files:
        raise RuntimeError("yt-dlp succeeded but no .mp4 found in output dir")

    video_path = str(mp4_files[0])
    duration = get_duration(video_path)
    print(f"  ✅ Downloaded: {video_path} ({duration:.0f}s)")
    return video_path


def validate_local_file(path: str) -> str:
    """Validate local video file exists and is readable."""
    print(f"\n▶ Step 1/6: Using local video file...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    duration = get_duration(path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  ✅ {path} ({duration:.0f}s, {size_mb:.1f} MB)")
    return path


def get_duration(path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Whisper transcription
# ──────────────────────────────────────────────────────────────────────────────

def transcribe(video_path: str, out_dir: str, language: str = "fr") -> dict:
    """Transcribe video using Whisper API. Returns parsed JSON."""
    print(f"\n▶ Step 2/6: Whisper transcription...")

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip3 install openai")

    # Extract audio
    with tempfile.NamedTemporaryFile(suffix=".mp3", prefix="whisper_", delete=False) as f:
        tmp_audio = f.name

    try:
        print("  Extracting audio...")
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "libmp3lame", "-q:a", "4",
            "-ar", "16000", "-ac", "1",
            tmp_audio
        ], capture_output=True, check=True)

        size_mb = os.path.getsize(tmp_audio) / (1024 * 1024)
        print(f"  Audio: {size_mb:.1f} MB")

        if size_mb > 24.9:
            raise ValueError(
                f"Audio too large for Whisper API: {size_mb:.1f} MB (limit ~25 MB). "
                "Consider splitting the video into segments."
            )

        client = OpenAI()
        print(f"  Calling Whisper API (language={language})...")

        def _call_whisper():
            with open(tmp_audio, "rb") as f:
                return client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                )

        response = with_retry(_call_whisper, retries=3, base_delay=2.0, label="whisper")
        # Track cost: audio length in minutes
        COST_TRACKER["whisper_minutes"] += size_mb / 1.5  # rough estimate: ~1.5 MB/min

        result = {
            "text": response.text,
            "language": getattr(response, "language", language),
            "duration": getattr(response, "duration", None),
            "segments": [],
            "words": [],
        }

        if hasattr(response, "segments") and response.segments:
            for seg in response.segments:
                result["segments"].append({
                    "start": seg.start if hasattr(seg, "start") else seg["start"],
                    "end": seg.end if hasattr(seg, "end") else seg["end"],
                    "text": (seg.text if hasattr(seg, "text") else seg["text"]).strip(),
                })

        if hasattr(response, "words") and response.words:
            for w in response.words:
                result["words"].append({
                    "start": w.start if hasattr(w, "start") else w["start"],
                    "end": w.end if hasattr(w, "end") else w["end"],
                    "word": (w.word if hasattr(w, "word") else w["word"]).strip(),
                })

        # Save JSON
        json_path = os.path.join(out_dir, "transcription.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        n_words = len(result["words"])
        n_segs = len(result["segments"])
        dur = result.get("duration", "?")
        print(f"  ✅ {n_words} words, {n_segs} segments, duration={dur}s")
        print(f"     Saved: {json_path}")

        return result

    finally:
        if os.path.exists(tmp_audio):
            os.remove(tmp_audio)


def load_whisper_json(path: str) -> dict:
    """Load existing Whisper JSON file."""
    print(f"\n▶ Step 2/6: Loading existing Whisper JSON...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    words = data.get("words", [])
    if not words:
        for seg in data.get("segments", []):
            words.extend(seg.get("words", []))
        data["words"] = words

    n_words = len(words)
    n_segs = len(data.get("segments", []))
    print(f"  ✅ {n_words} words, {n_segs} segments — {path}")
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Claude highlight detection
# ──────────────────────────────────────────────────────────────────────────────

def build_text_with_timestamps(whisper_data: dict, chunk_size: int = 500) -> list[dict]:
    """Build text chunks with start/end timestamps from word-level data.

    Returns list of {start, end, text} chunks of ~chunk_size words.
    """
    words = whisper_data.get("words", [])
    if not words:
        # Fall back to segments
        segs = whisper_data.get("segments", [])
        return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segs]

    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        text = " ".join(w["word"] for w in chunk_words)
        chunks.append({
            "start": chunk_words[0]["start"],
            "end": chunk_words[-1]["end"],
            "text": text,
        })
        i += chunk_size

    return chunks


def get_full_transcript_text(whisper_data: dict) -> str:
    """Get the full transcript as plain text with timestamps."""
    words = whisper_data.get("words", [])
    if not words:
        segs = whisper_data.get("segments", [])
        lines = []
        for seg in segs:
            ts = f"[{seg['start']:.1f}s-{seg['end']:.1f}s]"
            lines.append(f"{ts} {seg['text']}")
        return "\n".join(lines)

    # Build readable lines grouping words into ~10-word sentences
    lines = []
    chunk = []
    chunk_start = words[0]["start"]
    for w in words:
        chunk.append(w["word"])
        if len(chunk) >= 10 or w["word"].rstrip().endswith((".", "!", "?")):
            chunk_end = w["end"]
            ts = f"[{chunk_start:.1f}s]"
            lines.append(f"{ts} {' '.join(chunk)}")
            chunk = []
            chunk_start = chunk_end
    if chunk:
        chunk_end = words[-1]["end"]
        ts = f"[{chunk_start:.1f}s]"
        lines.append(f"{ts} {' '.join(chunk)}")

    return "\n".join(lines)


def detect_highlights_claude(whisper_data: dict, api_key: str, max_candidates: int = 10) -> list[dict]:
    """Use Claude to detect highlight moments from transcript.

    Returns list of {start, end, hook_text, score, explanation}.
    """
    print(f"\n▶ Step 3/6: Claude highlight detection...")

    import anthropic

    words = whisper_data.get("words", [])
    total_duration = words[-1]["end"] if words else 0
    full_text = get_full_transcript_text(whisper_data)

    print(f"  Transcript: {len(words)} words, {total_duration:.0f}s total")

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Tu es un expert en contenu viral pour TikTok/Reels/YouTube Shorts.

Analyse cette transcription de podcast (avec timestamps) et identifie les meilleurs moments viraux.

TRANSCRIPTION:
{full_text}

CRITÈRES DE SÉLECTION:
1. **HOOK fort** — Les premières secondes doivent captiver immédiatement (question choc, déclaration forte, réaction émotionnelle)
2. **Arc narratif complet** — Le moment doit avoir un début, un développement et une conclusion claire
3. **Potentiel viral** — Opinion tranchée, émotion forte, controverse, moment mémorable
4. **Autonomie** — Le clip doit se comprendre sans contexte supplémentaire

RÈGLES:
- Chaque clip doit faire entre 45s et 90s (idéalement 60-75s)
- Commence toujours au début d'une phrase complète (jamais en milieu de phrase)
- Termine toujours à la fin d'une phrase/pensée complète
- Le timestamp de début doit être exact (utilisé pour couper la vidéo)

Identifie les top {max_candidates} moments. Pour chaque moment, donne le timestamp de début et de fin (en secondes, précis à 0.5s près) basé sur les timestamps de la transcription.

Réponds UNIQUEMENT avec un JSON array:
[
  {{
    "start": <float — timestamp début exact en secondes>,
    "end": <float — timestamp fin exact en secondes>,
    "score": <int 1-10 — score viral>,
    "hook_text": "<les premières paroles du clip>",
    "arc_summary": "<résumé: hook + développement + conclusion>",
    "explanation": "<2-3 phrases expliquant pourquoi ce moment est viral>"
  }},
  ...
]

Important: utilise les timestamps exacts de la transcription (marqués [Xs]). Le start/end doivent correspondre à des mots réels dans la transcription."""

    try:
        print("  Calling Claude API...")
        response = with_retry(
            client.messages.create,
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
            retries=3, base_delay=2.0, label="claude-highlights"
        )
        COST_TRACKER["claude_input_tokens"] += getattr(response.usage, "input_tokens", 0)
        COST_TRACKER["claude_output_tokens"] += getattr(response.usage, "output_tokens", 0)

        response_text = response.content[0].text.strip()

        # Extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        candidates = json.loads(response_text)

        # Validate and clean
        valid = []
        for c in candidates:
            start = float(c.get("start", 0))
            end = float(c.get("end", 0))
            if end - start < 10:  # sanity check: must be at least 10s
                continue
            valid.append({
                "start": start,
                "end": end,
                "score": int(c.get("score", 5)),
                "hook_text": c.get("hook_text", ""),
                "arc_summary": c.get("arc_summary", ""),
                "explanation": c.get("explanation", ""),
                "raw_duration": end - start,
            })

        # Sort by score descending
        valid.sort(key=lambda x: x["score"], reverse=True)

        print(f"  ✅ {len(valid)} highlight candidates detected")
        for i, c in enumerate(valid[:5]):
            print(f"     #{i+1} [{c['start']:.0f}s-{c['end']:.0f}s] score={c['score']} — {c['hook_text'][:60]}")

        return valid

    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parse error: {e}")
        print(f"     Raw response: {response_text[:300]}")
        # Emergency fallback: return evenly-spaced windows
        return _fallback_highlights(whisper_data, max_candidates)
    except Exception as e:
        print(f"  ❌ Claude API error: {e}")
        return _fallback_highlights(whisper_data, max_candidates)


def _fallback_highlights(whisper_data: dict, n: int = 5) -> list[dict]:
    """Emergency fallback: evenly-spaced windows across transcript."""
    words = whisper_data.get("words", [])
    if not words:
        return []
    total = words[-1]["end"]
    step = total / (n + 1)
    results = []
    for i in range(n):
        start = step * (i + 1) - 30
        end = start + 60
        start = max(0, start)
        end = min(total, end)
        results.append({
            "start": start,
            "end": end,
            "score": 5,
            "hook_text": "(fallback)",
            "arc_summary": "Auto-generated fallback window",
            "explanation": "Claude detection failed, using evenly-spaced windows",
            "raw_duration": end - start,
        })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Dynamic window expansion
# ──────────────────────────────────────────────────────────────────────────────

def snap_to_word_boundary(t: float, words: list[dict], direction: str = "nearest") -> float:
    """Snap timestamp to nearest word boundary."""
    if not words:
        return t

    if direction == "start":
        # Find word that starts at or after t
        for w in words:
            if w["start"] >= t - 0.5:
                return w["start"]
        return words[-1]["start"]
    elif direction == "end":
        # Find word that ends at or before t
        for w in reversed(words):
            if w["end"] <= t + 0.5:
                return w["end"]
        return words[0]["end"]
    else:  # nearest
        best = min(words, key=lambda w: min(abs(w["start"] - t), abs(w["end"] - t)))
        if abs(best["start"] - t) <= abs(best["end"] - t):
            return best["start"]
        return best["end"]


def find_sentence_boundary(t: float, words: list[dict], direction: str = "forward",
                            lookahead: float = 15.0) -> float:
    """Find nearest sentence boundary (word ending in . ! ?) near t."""
    if direction == "forward":
        candidates = [w for w in words if w["end"] > t and w["end"] < t + lookahead]
    else:
        candidates = [w for w in words if w["end"] < t and w["end"] > t - lookahead]

    sentence_enders = {".", "!", "?", "…"}

    for w in (candidates if direction == "forward" else reversed(candidates)):
        word_text = w["word"].rstrip()
        if word_text and word_text[-1] in sentence_enders:
            return w["end"]

    # No sentence boundary found → snap to word boundary
    return snap_to_word_boundary(t, words, "end" if direction != "forward" else "start")


def expand_window(candidate: dict, whisper_data: dict,
                  min_dur: float = MIN_CLIP_DURATION,
                  max_dur: float = MAX_CLIP_DURATION,
                  target_dur: float = TARGET_CLIP_DURATION) -> dict:
    """Expand candidate clip to complete narrative arc.

    Algorithm:
    1. Start with raw Claude candidate (may be too short or too long)
    2. Snap start to previous sentence boundary (avoid mid-sentence cuts)
    3. Expand end until: duration >= min_dur AND ends at sentence boundary
    4. Cap at max_dur
    5. Trim start if needed to stay within max_dur while keeping minimum

    Returns updated candidate with adjusted start/end.
    """
    words = whisper_data.get("words", [])
    if not words:
        return candidate

    total_duration = words[-1]["end"]
    raw_start = candidate["start"]
    raw_end = candidate["end"]

    # Step A: Snap start to a sentence boundary BEFORE raw_start (don't cut mid-sentence)
    # Look back up to 5s for a sentence end
    start_adjusted = find_sentence_boundary(raw_start, words, direction="backward", lookahead=5.0)
    if abs(start_adjusted - raw_start) > 5.0:
        # Too far back — just snap to word boundary
        start_adjusted = snap_to_word_boundary(raw_start, words, direction="start")

    # Step B: Determine target end
    raw_duration = raw_end - start_adjusted
    if raw_duration < min_dur:
        # Need to extend
        target_end = start_adjusted + target_dur
    elif raw_duration > max_dur:
        # Need to trim
        target_end = start_adjusted + max_dur
    else:
        target_end = raw_end

    target_end = min(target_end, total_duration)

    # Step C: Find sentence boundary near target_end
    end_adjusted = find_sentence_boundary(target_end, words, direction="forward", lookahead=10.0)
    if end_adjusted > total_duration:
        end_adjusted = total_duration

    # Step D: Enforce min/max duration
    duration = end_adjusted - start_adjusted
    if duration < min_dur:
        # Extend end further
        end_adjusted = min(start_adjusted + min_dur, total_duration)
        # Snap to word boundary
        end_adjusted = snap_to_word_boundary(end_adjusted, words, direction="end")

    if duration > max_dur:
        # Trim from end (find sentence boundary closer)
        end_adjusted = find_sentence_boundary(start_adjusted + max_dur, words,
                                               direction="backward", lookahead=15.0)
        if end_adjusted <= start_adjusted:
            end_adjusted = start_adjusted + max_dur

    # Final clamping
    start_adjusted = max(0.0, start_adjusted)
    end_adjusted = min(total_duration, end_adjusted)
    duration = end_adjusted - start_adjusted

    # Safety check
    if duration < 5:
        print(f"  ⚠️  Window too short after expansion ({duration:.1f}s) — using raw candidate")
        start_adjusted = raw_start
        end_adjusted = raw_end

    updated = dict(candidate)
    updated["start"] = round(start_adjusted, 2)
    updated["end"] = round(end_adjusted, 2)
    updated["duration"] = round(end_adjusted - start_adjusted, 2)
    updated["start_raw"] = raw_start
    updated["end_raw"] = raw_end

    return updated


def expand_all_windows(candidates: list[dict], whisper_data: dict,
                       min_dur: float = MIN_CLIP_DURATION,
                       max_dur: float = MAX_CLIP_DURATION) -> list[dict]:
    """Expand all candidate windows and deduplicate overlapping clips."""
    print(f"\n▶ Step 4/6: Dynamic window expansion (min={min_dur}s, max={max_dur}s)...")

    expanded = []
    for i, c in enumerate(candidates):
        e = expand_window(c, whisper_data, min_dur, max_dur)
        print(f"  #{i+1}: [{c['start']:.0f}s-{c['end']:.0f}s] "
              f"→ [{e['start']:.0f}s-{e['end']:.0f}s] ({e['duration']:.0f}s)")
        expanded.append(e)

    # Deduplicate: remove clips that overlap >50% with a higher-scoring clip
    deduped = []
    for i, clip in enumerate(expanded):
        dominated = False
        for j, other in enumerate(deduped):
            overlap_start = max(clip["start"], other["start"])
            overlap_end = min(clip["end"], other["end"])
            overlap = max(0, overlap_end - overlap_start)
            clip_dur = clip["end"] - clip["start"]
            if clip_dur > 0 and overlap / clip_dur > 0.5:
                dominated = True
                break
        if not dominated:
            deduped.append(clip)

    print(f"  ✅ {len(deduped)} unique clips after deduplication (from {len(expanded)})")
    return deduped


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Extract + Reframe + Subtitles
# ──────────────────────────────────────────────────────────────────────────────

def extract_clip(video_path: str, start: float, end: float, out_path: str) -> str:
    """Extract clip using FFmpeg (accurate cut with keyframe seek)."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg extract failed: {result.stderr[-300:]}")
    return out_path


def reframe_clip(clip_path: str, out_path: str) -> str:
    """Run reframe_v3.py (YOLO + ByteTrack) on clip."""
    cmd = ["python3", REFRAME_SCRIPT, clip_path, out_path]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"  ⚠️  reframe_v3 stderr:\n{result.stderr[-400:]}")
        raise RuntimeError(f"reframe_v3 failed (exit {result.returncode})")
    return out_path


def add_subtitles(clip_path: str, whisper_data: dict, clip_start: float,
                  out_path: str, tmp_dir: str) -> str:
    """Run subtitles_v3.py with offset-adjusted Whisper JSON."""
    # Create offset-adjusted whisper JSON for this clip
    words = whisper_data.get("words", [])
    clip_words = [
        {
            "start": w["start"] - clip_start,
            "end": w["end"] - clip_start,
            "word": w["word"],
        }
        for w in words
        if w["start"] >= clip_start - 0.5 and w["end"] <= clip_start + 9999
    ]

    # Also adjust segments
    segments = whisper_data.get("segments", [])
    clip_segs = [
        {
            "start": s["start"] - clip_start,
            "end": s["end"] - clip_start,
            "text": s["text"],
        }
        for s in segments
        if s["start"] >= clip_start - 0.5
    ]

    clip_whisper = {
        "text": " ".join(w["word"] for w in clip_words),
        "language": whisper_data.get("language", "fr"),
        "duration": None,
        "segments": clip_segs,
        "words": clip_words,
    }

    whisper_tmp = os.path.join(tmp_dir, "clip_whisper.json")
    with open(whisper_tmp, "w", encoding="utf-8") as f:
        json.dump(clip_whisper, f, ensure_ascii=False)

    cmd = [
        "python3", SUBTITLES_SCRIPT,
        "--video", clip_path,
        "--whisper", whisper_tmp,
        "--output", out_path,
        "--style", "stroke",
        "--highlight", "box",
        "--max-words", "3",
        "--font-size", "68",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"  ⚠️  subtitles_v3 stderr:\n{result.stderr[-400:]}")
        raise RuntimeError(f"subtitles_v3 failed (exit {result.returncode})")
    return out_path


def process_clip(clip_idx: int, clip: dict, video_path: str, whisper_data: dict,
                 out_dir: str, podcast_name: str = "podcast",
                 no_reframe: bool = False, no_subs: bool = False) -> dict:
    """Full clip processing: extract → reframe → subtitles."""
    score = clip.get("score", 0)
    clip_dir = os.path.join(out_dir, f"clip_{clip_idx+1:02d}")
    os.makedirs(clip_dir, exist_ok=True)

    start = clip["start"]
    end = clip["end"]
    duration = end - start

    print(f"\n  ── Clip #{clip_idx+1}: [{start:.0f}s-{end:.0f}s] ({duration:.0f}s) ──")

    result = dict(clip)
    result["clip_dir"] = clip_dir
    result["clip_index"] = clip_idx + 1

    # 1. Extract raw clip
    raw_path = os.path.join(clip_dir, "raw.mp4")
    print(f"    [extract] → {raw_path}")
    extract_clip(video_path, start, end, raw_path)
    result["raw_path"] = raw_path

    if no_reframe and no_subs:
        result["final_path"] = raw_path
        return result

    # 2. Reframe 9:16
    if no_reframe:
        reframed_path = raw_path
    else:
        reframed_path = os.path.join(clip_dir, "reframed_9x16.mp4")
        print(f"    [reframe] → {reframed_path}")
        try:
            reframe_clip(raw_path, reframed_path)
        except RuntimeError as e:
            print(f"    ⚠️  Reframe failed: {e} — using raw clip")
            reframed_path = raw_path
    result["reframed_path"] = reframed_path

    if no_subs:
        result["final_path"] = reframed_path
        return result

    # 3. Add subtitles
    # Final output: {out_dir}/{podcast_name}/clip_{N}_{score}.mp4
    podcast_out_dir = os.path.join(out_dir, podcast_name)
    os.makedirs(podcast_out_dir, exist_ok=True)
    final_path = os.path.join(podcast_out_dir, f"clip_{clip_idx+1}_{score}.mp4")
    print(f"    [subtitles] → {final_path}")
    with tempfile.TemporaryDirectory(prefix="subs_v2_") as tmp_dir:
        try:
            add_subtitles(reframed_path, whisper_data, start, final_path, tmp_dir)
        except RuntimeError as e:
            print(f"    ⚠️  Subtitles failed: {e} — using reframed clip")
            final_path = reframed_path

    result["final_path"] = final_path

    # Validate output
    if os.path.exists(final_path):
        actual_dur = get_duration(final_path)
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        print(f"    ✅ {final_path} ({actual_dur:.0f}s, {size_mb:.1f} MB)")
        result["actual_duration"] = actual_dur
        result["size_mb"] = size_mb
    else:
        print(f"    ❌ Final clip not found: {final_path}")
        result["error"] = "Final clip missing"

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Output
# ──────────────────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def print_summary(clips: list[dict], dry_run: bool = False):
    """Print final summary."""
    print("\n" + "=" * 70)
    if dry_run:
        print("  🔍 DRY-RUN COMPLETE — Highlight Detection Results")
    else:
        print("  ✅ PIPELINE v2 COMPLETE")
    print("=" * 70)

    for i, clip in enumerate(clips):
        start = clip["start"]
        end = clip["end"]
        duration = clip.get("duration", end - start)
        score = clip.get("score", "?")
        hook = clip.get("hook_text", "")[:70]

        print(f"\n  #{i+1} — Score: {score}/10 — {format_time(start)}-{format_time(end)} ({duration:.0f}s)")
        print(f"       Hook: {hook}")
        print(f"       {clip.get('explanation', '')[:100]}")

        if not dry_run:
            final = clip.get("final_path", "N/A")
            actual_dur = clip.get("actual_duration", "?")
            size = clip.get("size_mb", "?")
            print(f"       📹 {final} ({actual_dur}s, {size:.1f}MB)" if isinstance(size, float) else f"       📹 {final}")

    print("\n" + "=" * 70)


def save_results(clips: list[dict], out_dir: str, dry_run: bool = False):
    """Save results to JSON."""
    results_path = os.path.join(out_dir, "results.json")

    output = {
        "pipeline_version": "v2",
        "dry_run": dry_run,
        "clips": clips,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  💾 Results saved: {results_path}")
    return results_path


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Podoro Clips Pipeline v2 — Full E2E: YouTube → Clips viraux"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="YouTube URL to download")
    src.add_argument("--video", help="Local video file path")

    p.add_argument("--whisper-json", help="Pre-existing Whisper JSON (skip Step 2)")
    p.add_argument("--out", default="./output", help="Output directory (default: ./output)")
    p.add_argument("--max-clips", type=int, default=DEFAULT_MAX_CLIPS,
                   help=f"Max clips to produce (default: {DEFAULT_MAX_CLIPS})")
    p.add_argument("--min-score", type=int, default=DEFAULT_MIN_SCORE,
                   help=f"Minimum viral score 1-10 (default: {DEFAULT_MIN_SCORE}, no filter)")
    p.add_argument("--language", default="fr", help="Whisper language (default: fr)")
    p.add_argument("--dry-run", action="store_true",
                   help="Detect highlights only, skip video processing")
    p.add_argument("--no-reframe", action="store_true", help="Skip reframing step")
    p.add_argument("--no-subs", action="store_true", help="Skip subtitles step")
    p.add_argument("--skip-subtitles", action="store_true", help="Alias for --no-subs")
    p.add_argument("--no-diarization", action="store_true",
                   help="Disable speaker diarization (not used in this pipeline; for compatibility)")
    p.add_argument("--anthropic-key", help="Anthropic API key (default: ANTHROPIC_API_KEY env)")
    p.add_argument("--openai-key", help="OpenAI API key (default: OPENAI_API_KEY env)")
    return p.parse_args()


def load_api_keys(args):
    """Load API keys from args, env, or .env file."""
    # Anthropic
    anthropic_key = (
        args.anthropic_key
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not anthropic_key:
        env_file = "/Users/OpenClaw/.openclaw/workspace-anthropic/.env.anthropic"
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    if line.startswith("ANTHROPIC_API_KEY="):
                        anthropic_key = line.strip().split("=", 1)[1]
                        break
    if not anthropic_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. Set env var or use --anthropic-key"
        )

    # OpenAI (Whisper) — only needed if transcription required
    openai_key = (
        args.openai_key
        or os.environ.get("OPENAI_API_KEY")
    )
    if not openai_key:
        env_file = "/Users/OpenClaw/.openclaw/workspace-anthropic/.env.anthropic"
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    if line.startswith("OPENAI_API_KEY="):
                        openai_key = line.strip().split("=", 1)[1]
                        break
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    return anthropic_key, openai_key


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 70)
    print("  🎙️  Podoro Clips Pipeline v2")
    print("=" * 70)

    t_start = time.time()

    # Merge --skip-subtitles into --no-subs
    if args.skip_subtitles:
        args.no_subs = True

    if args.no_diarization:
        print("  ℹ️  --no-diarization: diarization is not used in this pipeline (no-op)")

    # Load API keys
    try:
        anthropic_key, openai_key = load_api_keys(args)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # ── Step 1: Source video ──────────────────────────────────────────────────
    if args.url:
        video_path = download_youtube(args.url, args.out)
    else:
        video_path = validate_local_file(args.video)

    # Derive podcast_name from video filename
    podcast_name = Path(video_path).stem
    # Sanitize: keep only alphanumeric, dashes, underscores
    import re as _re
    podcast_name = _re.sub(r'[^\w\-]', '_', podcast_name)[:50].strip('_') or "podcast"

    # ── Step 2: Transcription ─────────────────────────────────────────────────
    if args.whisper_json:
        whisper_data = load_whisper_json(args.whisper_json)
        # Save a copy to output dir
        json_copy = os.path.join(args.out, "transcription.json")
        if not os.path.exists(json_copy):
            import shutil
            shutil.copy2(args.whisper_json, json_copy)
    else:
        if not openai_key:
            print("❌ OPENAI_API_KEY required for Whisper transcription")
            print("   Use --whisper-json to skip, or set OPENAI_API_KEY")
            sys.exit(1)
        whisper_data = transcribe(video_path, args.out, args.language)

    # ── Step 3: Highlight detection ───────────────────────────────────────────
    candidates = detect_highlights_claude(whisper_data, anthropic_key,
                                          max_candidates=args.max_clips * 3)

    if not candidates:
        print("❌ No highlight candidates detected")
        sys.exit(1)

    # ── Step 4: Window expansion ──────────────────────────────────────────────
    expanded = expand_all_windows(candidates, whisper_data)

    # Filter by min-score
    if args.min_score > 0:
        before = len(expanded)
        expanded = [c for c in expanded if c.get("score", 0) >= args.min_score]
        print(f"  🔍 Min-score filter ({args.min_score}): {len(expanded)}/{before} clips kept")
        if not expanded:
            print(f"❌ No clips with score >= {args.min_score}")
            sys.exit(1)

    # Take top N by score
    top_clips = expanded[:args.max_clips]

    if args.dry_run:
        print(f"\n▶ Steps 5-6: SKIPPED (--dry-run)")
        print_summary(top_clips, dry_run=True)
        results_path = save_results(top_clips, args.out, dry_run=True)
        elapsed = time.time() - t_start
        print(f"\n  ⏱️  Total time: {elapsed:.0f}s")
        return top_clips

    # ── Step 5: Process clips ─────────────────────────────────────────────────
    print(f"\n▶ Step 5/6: Processing {len(top_clips)} clip(s)...")
    final_clips = []
    clips_ok = []
    clips_failed = []

    with tqdm(total=len(top_clips), desc="clips", unit="clip") as pbar:
        for i, clip in enumerate(top_clips):
            pbar.set_description(f"Clip {i+1}/{len(top_clips)} [{clip['start']:.0f}s-{clip['end']:.0f}s]")
            try:
                result = process_clip(
                    i, clip, video_path, whisper_data, args.out,
                    podcast_name=podcast_name,
                    no_reframe=args.no_reframe,
                    no_subs=args.no_subs,
                )
                final_clips.append(result)
                if result.get("final_path") and os.path.exists(result.get("final_path", "")):
                    clips_ok.append(result)
                else:
                    clips_failed.append(result)
            except Exception as e:
                print(f"\n  ❌ Clip #{i+1} failed (continuing): {e}")
                clip["error"] = str(e)
                final_clips.append(clip)
                clips_failed.append(clip)
            pbar.update(1)

    # ── Step 6: Output ────────────────────────────────────────────────────────
    print(f"\n▶ Step 6/6: Results...")
    print_summary(final_clips)
    results_path = save_results(final_clips, args.out)

    elapsed = time.time() - t_start
    estimated_cost = estimate_cost()

    print("\n" + "=" * 70)
    print("  📊 FINAL SUMMARY")
    print("=" * 70)
    print(f"  ✅ Clips generated  : {len(clips_ok)}")
    print(f"  ❌ Clips failed     : {len(clips_failed)}")
    print(f"  ⏱️  Total time       : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  💰 Estimated cost   : ~${estimated_cost:.4f} USD")
    print(f"  📁 Output dir       : {args.out}/{podcast_name}/")
    if clips_ok:
        for c in clips_ok:
            fp = c.get("final_path", "")
            size = c.get("size_mb", 0)
            dur = c.get("actual_duration", 0)
            print(f"     📹 {os.path.basename(fp)} ({dur:.0f}s, {size:.1f} MB)")
    if clips_failed:
        print(f"\n  Failed clips:")
        for c in clips_failed:
            print(f"     ❌ Clip #{c.get('clip_index','?')}: {c.get('error', 'unknown error')}")
    print("=" * 70)

    return final_clips


if __name__ == "__main__":
    main()

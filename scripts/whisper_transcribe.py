#!/usr/bin/env python3
"""
Whisper API transcription with word-level timestamps.
Outputs JSON with segments and word timestamps — single source of truth
for subtitles AND diarization input.

Usage:
    python whisper_transcribe.py <input_audio_or_video> [output.json]

Requires: OPENAI_API_KEY env var or .env file.
"""

import sys
import os
import json
import subprocess
import tempfile
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("❌ openai package not installed. Run: pip3 install openai")
    sys.exit(1)


def extract_audio(video_path: str, audio_path: str) -> str:
    """Extract audio from video as mp3 (Whisper API accepts mp3/mp4/wav/etc)."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "libmp3lame", "-q:a", "4",
        "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return audio_path


def transcribe_whisper(audio_path: str, language: str = "fr") -> dict:
    """Call OpenAI Whisper API with word-level timestamps."""
    client = OpenAI()

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

    # Convert to our standard format
    result = {
        "text": response.text,
        "language": response.language,
        "duration": response.duration,
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

    return result


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_audio_or_video> [output.json] [--language fr]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    
    language = "fr"
    for i, arg in enumerate(sys.argv):
        if arg == "--language" and i + 1 < len(sys.argv):
            language = sys.argv[i + 1]

    if not output_path:
        output_path = str(Path(input_path).with_suffix(".whisper.json"))

    print(f"=== Whisper Transcription: {input_path} ===")

    # Extract audio if input is video
    ext = Path(input_path).suffix.lower()
    audio_path = input_path
    tmp_audio = None

    if ext in (".mp4", ".mkv", ".webm", ".mov", ".avi"):
        tmp_audio = tempfile.mktemp(suffix=".mp3", prefix="whisper_")
        print("  Extracting audio...")
        audio_path = extract_audio(input_path, tmp_audio)

    try:
        print(f"  Transcribing with Whisper API (language={language})...")
        result = transcribe_whisper(audio_path, language=language)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        n_words = len(result.get("words", []))
        n_segs = len(result.get("segments", []))
        print(f"  ✅ {n_words} words, {n_segs} segments")
        print(f"  Output: {output_path}")

    finally:
        if tmp_audio and os.path.exists(tmp_audio):
            os.remove(tmp_audio)

    return result


if __name__ == "__main__":
    main()

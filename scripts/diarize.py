#!/usr/bin/env python3
"""
Speaker diarization using pyannote.audio.
Outputs a JSON file with speaker segments compatible with reframe_v3.

Usage:
    python diarize.py <input_audio_or_video> [output.json] [--hf-token TOKEN]

Output format:
    [
        {"start": 0.0, "end": 3.5, "speaker": "SPEAKER_00"},
        {"start": 3.5, "end": 7.2, "speaker": "SPEAKER_01"},
        ...
    ]

Requires:
    - pyannote.audio (pip install pyannote.audio)
    - HuggingFace token with pyannote/speaker-diarization-3.1 access
      Set via --hf-token, HF_TOKEN env var, or ~/.huggingface/token
"""

import sys
import os
import json
import subprocess
import tempfile
from pathlib import Path


def extract_wav(video_path: str, wav_path: str) -> str:
    """Extract audio as 16kHz mono WAV for pyannote."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        wav_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return wav_path


def run_diarization(audio_path: str, hf_token: str = None, num_speakers: int = None) -> list:
    """Run pyannote speaker diarization."""
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("❌ pyannote.audio not installed. Run: pip3 install pyannote.audio")
        sys.exit(1)

    token = hf_token or os.environ.get("HF_TOKEN") or None

    if not token:
        # Try reading from huggingface cache
        hf_token_path = Path.home() / ".huggingface" / "token"
        if hf_token_path.exists():
            token = hf_token_path.read_text().strip()

    if not token:
        print("❌ HALT: pyannote requires a HuggingFace token.")
        print("  Get one at https://huggingface.co/settings/tokens")
        print("  Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("  Then: export HF_TOKEN=hf_... or pass --hf-token")
        sys.exit(2)

    print("  Loading pyannote pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,  # FIX #5: use_auth_token deprecated since transformers>=4.34
    )

    print("  Running diarization...")
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(audio_path, **kwargs)

    # Convert to list of segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        })

    # Merge consecutive same-speaker segments
    merged = []
    for seg in segments:
        if merged and merged[-1]["speaker"] == seg["speaker"] and seg["start"] - merged[-1]["end"] < 0.5:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)

    return merged


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input> [output.json] [--hf-token TOKEN] [--num-speakers N]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = None
    hf_token = None
    num_speakers = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--hf-token" and i + 1 < len(sys.argv):
            hf_token = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--num-speakers" and i + 1 < len(sys.argv):
            num_speakers = int(sys.argv[i + 1])
            i += 2
        elif not output_path and not sys.argv[i].startswith("--"):
            output_path = sys.argv[i]
            i += 1
        else:
            i += 1

    if not output_path:
        output_path = str(Path(input_path).with_suffix(".diarization.json"))

    print(f"=== Speaker Diarization: {input_path} ===")

    # Extract audio if needed
    ext = Path(input_path).suffix.lower()
    audio_path = input_path
    tmp_wav = None

    if ext not in (".wav",):
        # MEDIUM FIX #4: NamedTemporaryFile replaces deprecated mktemp() (TOCTOU race condition)
        with tempfile.NamedTemporaryFile(suffix=".wav", prefix="diarize_", delete=False) as _f:
            tmp_wav = _f.name
        print("  Extracting audio...")
        audio_path = extract_wav(input_path, tmp_wav)

    try:
        segments = run_diarization(audio_path, hf_token=hf_token, num_speakers=num_speakers)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        speakers = set(s["speaker"] for s in segments)
        total_dur = max((s["end"] for s in segments), default=0)
        print(f"  ✅ {len(segments)} segments, {len(speakers)} speakers, {total_dur:.1f}s")
        print(f"  Speakers: {', '.join(sorted(speakers))}")
        print(f"  Output: {output_path}")

    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)

    return segments


if __name__ == "__main__":
    main()

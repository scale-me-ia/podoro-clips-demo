#!/usr/bin/env python3
"""Cut silences from video while preserving punchline pauses."""

import sys
import subprocess
import json
import tempfile
import os

def detect_silences(audio_path, threshold_db=-35, min_duration=0.4):
    """Detect silent segments using ffmpeg silencedetect."""
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_duration}",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    silences = []
    lines = result.stderr.split("\n")
    current_start = None
    
    for line in lines:
        if "silence_start:" in line:
            try:
                current_start = float(line.split("silence_start:")[1].strip().split()[0])
            except:
                pass
        elif "silence_end:" in line and current_start is not None:
            try:
                parts = line.split("silence_end:")[1].strip().split()
                end = float(parts[0])
                duration = float(parts[2]) if len(parts) > 2 else end - current_start
                silences.append({"start": current_start, "end": end, "duration": duration})
                current_start = None
            except:
                pass
    
    return silences

def cut_silences(input_path, output_path, min_silence=0.4, keep_duration=0.15):
    """Remove silences, keeping a small gap."""
    print(f"Detecting silences (>{min_silence}s)...")
    silences = detect_silences(input_path, min_duration=min_silence)
    print(f"Found {len(silences)} silences")
    
    if not silences:
        # No silences to cut, just copy
        subprocess.run(["cp", input_path, output_path])
        return
    
    # Get video duration
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
         "-of", "default=nk=1:nw=1", input_path],
        capture_output=True, text=True
    )
    total_duration = float(probe.stdout.strip())
    
    # Build segments to keep
    segments = []
    prev_end = 0.0
    
    for s in silences:
        # Keep audio before silence
        if s["start"] > prev_end:
            segments.append({"start": prev_end, "end": s["start"] + keep_duration / 2})
        # Skip most of the silence, keep a small gap
        prev_end = s["end"] - keep_duration / 2
    
    # Keep final segment
    if prev_end < total_duration:
        segments.append({"start": prev_end, "end": total_duration})
    
    print(f"Keeping {len(segments)} segments")
    
    # Use ffmpeg concat with trim filters
    tmpdir = tempfile.mkdtemp(prefix="silcut_")
    parts = []
    
    for i, seg in enumerate(segments):
        part_path = os.path.join(tmpdir, f"part_{i:03d}.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ss", str(seg["start"]), "-to", str(seg["end"]),
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k",
            part_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        parts.append(part_path)
    
    # Concat
    listfile = os.path.join(tmpdir, "concat.txt")
    with open(listfile, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
        "-c", "copy", output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    
    # Get output duration
    probe2 = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", output_path],
        capture_output=True, text=True
    )
    out_dur = float(probe2.stdout.strip())
    saved = total_duration - out_dur
    print(f"Output: {output_path} ({out_dur:.1f}s, saved {saved:.1f}s)")
    
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.mp4> <output.mp4> [min_silence] [keep_duration]")
        sys.exit(1)
    
    min_silence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
    keep_duration = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
    cut_silences(sys.argv[1], sys.argv[2], min_silence, keep_duration)

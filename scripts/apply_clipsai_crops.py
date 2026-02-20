#!/usr/bin/env python3
"""Apply ClipsAI crop segments to generate a 9:16 video with smooth transitions."""

import sys
import cv2
import numpy as np
import subprocess

LERP_INERTIA = 0.95  # Smooth transitions between segments

def apply_crops(input_path, output_path, segments, target_w=1080, target_h=1920):
    """
    segments: list of (start_time, end_time, crop_x)
    """
    cap = cv2.VideoCapture(input_path)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    crop_w = int(src_h * 9 / 16)
    crop_h = src_h
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(src_w * 16 / 9)
    
    print(f"Source: {src_w}x{src_h} @ {fps}fps, {total} frames")
    print(f"Crop: {crop_w}x{crop_h}")
    print(f"Segments: {len(segments)}")
    
    # Build per-frame target crop_x
    targets = np.zeros(total)
    for start, end, crop_x in segments:
        # Clamp crop_x
        crop_x = max(0, min(crop_x, src_w - crop_w))
        start_frame = int(start * fps)
        end_frame = min(int(end * fps), total)
        targets[start_frame:end_frame] = crop_x
    
    # Smooth with exponential lerp
    smoothed = np.zeros(total)
    smoothed[0] = targets[0]
    for i in range(1, total):
        smoothed[i] = LERP_INERTIA * smoothed[i-1] + (1 - LERP_INERTIA) * targets[i]
    
    # Apply
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{crop_w}x{crop_h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-i", input_path,
        "-map", "0:v", "-map", "1:a",
        "-vf", f"scale={target_w}:{target_h}",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "copy",
        "-shortest",
        output_path
    ]
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        crop_x = int(smoothed[min(frame_idx, total-1)])
        crop_x = max(0, min(crop_x, src_w - crop_w))
        cropped = frame[0:crop_h, crop_x:crop_x + crop_w]
        
        try:
            proc.stdin.write(cropped.tobytes())
        except BrokenPipeError:
            break
        
        frame_idx += 1
    
    cap.release()
    proc.stdin.close()
    proc.wait()
    
    # Stats
    diffs = [abs(int(smoothed[i]) - int(smoothed[i-1])) for i in range(1, min(frame_idx, total))]
    print(f"Stats: max_jump={max(diffs)}px, avg_move={np.mean(diffs):.1f}px/frame")
    print(f"✅ Output: {output_path} ({frame_idx} frames)")

if __name__ == "__main__":
    # Parse ClipsAI crop segments from command line
    import json
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    crops_json = sys.argv[3]
    
    with open(crops_json) as f:
        raw = json.load(f)
    
    segments = []
    for seg in raw:
        segments.append((seg["start"], seg["end"], seg["crop_x"]))
    
    apply_crops(input_path, output_path, segments)

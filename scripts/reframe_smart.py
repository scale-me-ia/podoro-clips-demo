#!/usr/bin/env python3
"""Smart reframe v3: smooth + always a face + hysteresis.

Fixes:
1. SMOOTH: exponential lerp (0.92/0.08) per frame — no snapping
2. ALWAYS A FACE: last-known fallback + rolling 30-frame face buffer
3. HYSTERESIS: don't switch target unless new face stable for >0.5s
"""

import sys
import cv2
import numpy as np
import subprocess
import json
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import Counter, deque

MODEL_PATH = "/tmp/blaze_face_short_range.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

# === TUNING CONSTANTS ===
LERP_INERTIA = 0.92          # Fix 1: how much to keep old position (0.92 = very smooth)
LERP_PULL = 1.0 - LERP_INERTIA  # 0.08 = gentle pull toward target
HYSTERESIS_FRAMES = 15        # Fix 3: new target must be stable for N frames before switching
FACE_BUFFER_SIZE = 30         # Fix 2: rolling buffer of recent face positions
WIDE_SHOT_GAP = 0.3           # min gap between 2 faces to classify as "wide"
DETECT_CONFIDENCE = 0.25      # low threshold to catch profiles/beards


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face detection model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def analyze_video(video_path, sample_every=2):
    """Detect faces every N frames. Returns per-sample face data."""
    ensure_model()
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=base_options, min_detection_confidence=DETECT_CONFIDENCE
    )
    detector = vision.FaceDetector.create_from_options(options)
    
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    samples = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            
            faces = []
            for det in result.detections:
                bbox = det.bounding_box
                cx = (bbox.origin_x + bbox.width / 2) / w
                face_w = bbox.width / w
                score = det.categories[0].score if det.categories else 0
                faces.append({"cx": cx, "w": face_w, "score": score})
            
            # Sort by x
            faces.sort(key=lambda f: f["cx"])
            
            samples.append({"frame": frame_idx, "faces": faces})
        
        frame_idx += 1
    
    cap.release()
    detector.close()
    
    n_wide = sum(1 for s in samples if len(s["faces"]) >= 2 
                 and s["faces"][-1]["cx"] - s["faces"][0]["cx"] > WIDE_SHOT_GAP)
    n_close = sum(1 for s in samples if len(s["faces"]) == 1)
    n_none = sum(1 for s in samples if len(s["faces"]) == 0)
    print(f"Analyzed {len(samples)} samples (wide: {n_wide}, close: {n_close}, noface: {n_none})")
    
    return samples, w, h, fps, total


def compute_crop_positions(samples, src_w, src_h, fps, total_frames):
    """Core algorithm: compute per-frame crop_x with smooth + hysteresis + fallback.
    
    Returns list of crop_x (int) for every frame.
    """
    crop_w = int(src_h * 9 / 16)
    crop_h = src_h
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(src_w * 16 / 9)
    
    min_cx = crop_w / (2 * src_w)
    max_cx = 1.0 - min_cx
    
    def clamp_cx(cx):
        return max(min_cx, min(max_cx, cx))
    
    # --- Step 1: Build raw target_cx per sample ---
    # With hysteresis + fallback
    
    face_buffer = deque(maxlen=FACE_BUFFER_SIZE)  # recent face cx values
    current_target_cx = 0.5
    candidate_cx = None
    candidate_frames = 0
    
    raw_targets = []  # (frame_idx, target_cx)
    
    for s in samples:
        faces = s["faces"]
        frame = s["frame"]
        
        if len(faces) == 0:
            # FIX 2: No face → use buffer fallback
            if face_buffer:
                fallback_cx = float(np.median(list(face_buffer)))
            else:
                fallback_cx = current_target_cx
            desired_cx = fallback_cx
            # Don't update candidate — we're in fallback mode
        
        elif len(faces) == 1:
            # Single face — straightforward
            desired_cx = faces[0]["cx"]
            face_buffer.append(desired_cx)
            candidate_cx = None
            candidate_frames = 0
        
        elif len(faces) >= 2:
            gap = faces[-1]["cx"] - faces[0]["cx"]
            
            if gap > WIDE_SHOT_GAP:
                # Wide shot: pick largest face (most prominent)
                biggest = max(faces, key=lambda f: f["w"])
                desired_cx = biggest["cx"]
                face_buffer.append(desired_cx)
            else:
                # Close shot with multiple faces: pick best score
                best = max(faces, key=lambda f: f["score"])
                desired_cx = best["cx"]
                face_buffer.append(desired_cx)
                candidate_cx = None
                candidate_frames = 0
        
        # FIX 3: Hysteresis — only switch target if new position is stable
        desired_cx = clamp_cx(desired_cx)
        
        if abs(desired_cx - current_target_cx) > 0.15:
            # Big jump — candidate for switch
            if candidate_cx is not None and abs(desired_cx - candidate_cx) < 0.1:
                candidate_frames += 1
            else:
                candidate_cx = desired_cx
                candidate_frames = 1
            
            if candidate_frames >= HYSTERESIS_FRAMES:
                # Stable enough — switch!
                current_target_cx = candidate_cx
                candidate_cx = None
                candidate_frames = 0
            # else: keep old target (ignore jitter)
        else:
            # Small movement — track smoothly, no hysteresis needed
            current_target_cx = desired_cx
            candidate_cx = None
            candidate_frames = 0
        
        raw_targets.append((frame, current_target_cx))
    
    # --- Step 2: Interpolate to all frames ---
    if not raw_targets:
        return [0] * total_frames, crop_w, crop_h
    
    sample_frames = [t[0] for t in raw_targets]
    sample_cxs = [t[1] for t in raw_targets]
    all_target_cxs = np.interp(range(total_frames), sample_frames, sample_cxs)
    
    # --- Step 3: FIX 1 — Exponential smoothing (lerp) ---
    smoothed_cx = np.zeros(total_frames)
    smoothed_cx[0] = all_target_cxs[0]
    
    for i in range(1, total_frames):
        smoothed_cx[i] = LERP_INERTIA * smoothed_cx[i-1] + LERP_PULL * all_target_cxs[i]
    
    # --- Step 4: Convert to crop_x ---
    crop_positions = []
    for cx in smoothed_cx:
        crop_x = int(cx * src_w - crop_w / 2)
        crop_x = max(0, min(crop_x, src_w - crop_w))
        crop_positions.append(crop_x)
    
    # Stats
    diffs = [abs(crop_positions[i] - crop_positions[i-1]) for i in range(1, len(crop_positions))]
    max_jump = max(diffs) if diffs else 0
    avg_move = np.mean(diffs) if diffs else 0
    print(f"Crop stats: max_jump={max_jump}px, avg_move={avg_move:.1f}px/frame")
    
    return crop_positions, crop_w, crop_h


def apply_dynamic_crop(input_path, output_path, crop_positions, crop_w, crop_h,
                       src_w, src_h, fps, target_w=1080, target_h=1920):
    """Frame-by-frame crop via ffmpeg pipe."""
    
    cap = cv2.VideoCapture(input_path)
    
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
        
        idx = min(frame_idx, len(crop_positions) - 1)
        crop_x = crop_positions[idx]
        cropped = frame[0:crop_h, crop_x:crop_x + crop_w]
        
        try:
            proc.stdin.write(cropped.tobytes())
        except BrokenPipeError:
            break
        
        frame_idx += 1
    
    cap.release()
    proc.stdin.close()
    proc.wait()
    print(f"Output: {output_path} ({frame_idx} frames)")


def reframe_smart(input_path, output_path, whisper_json=None, time_offset=0.0):
    """Full pipeline."""
    print(f"=== Reframe v3: {os.path.basename(input_path)} ===")
    
    samples, src_w, src_h, fps, total = analyze_video(input_path)
    
    if whisper_json and os.path.exists(whisper_json):
        with open(whisper_json) as f:
            data = json.load(f)
        print(f"Whisper: {len(data.get('segments', []))} segments loaded")
    
    crop_positions, crop_w, crop_h = compute_crop_positions(
        samples, src_w, src_h, fps, total
    )
    
    print(f"Applying crop ({crop_w}x{crop_h}, {total} frames)...")
    apply_dynamic_crop(
        input_path, output_path, crop_positions, crop_w, crop_h,
        src_w, src_h, fps
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.mp4> <output.mp4> [whisper.json] [time_offset]")
        sys.exit(1)
    
    whisper_json = sys.argv[3] if len(sys.argv) > 3 else None
    time_offset = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    reframe_smart(sys.argv[1], sys.argv[2], whisper_json, time_offset)

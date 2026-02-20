#!/usr/bin/env python3
"""Smart reframe: detect wide shots with 2 speakers, crop to active speaker.
Uses Whisper word timestamps to determine who's talking per segment."""

import sys
import cv2
import numpy as np
import subprocess
import json
import os
import urllib.request
import tempfile
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "/tmp/blaze_face_short_range.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def analyze_video(video_path, sample_every=3):
    """Detect faces per frame, classify as close-up vs wide shot."""
    ensure_model()
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=base_options, min_detection_confidence=0.25
    )
    detector = vision.FaceDetector.create_from_options(options)
    
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_data = []  # (frame_idx, shot_type, faces_info)
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
                cy = (bbox.origin_y + bbox.height / 2) / h
                face_w = bbox.width / w
                score = det.categories[0].score if det.categories else 0
                faces.append({
                    "cx": cx, "cy": cy, "w": face_w,
                    "score": score, "side": "left" if cx < 0.5 else "right"
                })
            
            if len(faces) >= 2:
                # Sort by x position
                faces.sort(key=lambda f: f["cx"])
                gap = faces[-1]["cx"] - faces[0]["cx"]
                
                if gap > 0.3:  # Wide shot: faces far apart
                    shot_type = "wide"
                else:
                    shot_type = "close"
            elif len(faces) == 1:
                shot_type = "close"
            else:
                shot_type = "noface"
            
            frame_data.append({
                "frame": frame_idx, "time": frame_idx / fps,
                "shot_type": shot_type, "faces": faces
            })
        
        frame_idx += 1
    
    cap.release()
    detector.close()
    
    # Stats
    wide_count = sum(1 for f in frame_data if f["shot_type"] == "wide")
    close_count = sum(1 for f in frame_data if f["shot_type"] == "close")
    noface_count = sum(1 for f in frame_data if f["shot_type"] == "noface")
    print(f"Frames analyzed: {len(frame_data)} (wide: {wide_count}, close: {close_count}, noface: {noface_count})")
    
    return frame_data, w, h, fps, total


def load_whisper_segments(json_path, time_offset=0.0):
    """Load Whisper segments to know when speech happens."""
    with open(json_path) as f:
        data = json.load(f)
    segments = []
    for seg in data.get("segments", []):
        segments.append({
            "start": seg["start"] + time_offset,
            "end": seg["end"] + time_offset,
            "text": seg["text"].strip()
        })
    return segments


def assign_speaker_heuristic(frame_data, whisper_segments=None):
    """For wide shots and noface frames, decide which face to crop to.
    
    Strategy:
    - Wide shots: pick the LARGER face (leaning in = speaking)
    - Noface: use LAST KNOWN position (don't jump to center)
    - With Whisper: alternate sides on segment boundaries (TODO: diarization)
    """
    last_known_cx = None
    last_known_side = None
    
    for fd in frame_data:
        if fd["shot_type"] == "wide" and len(fd["faces"]) >= 2:
            # Pick the face with the largest bounding box (most prominent)
            biggest = max(fd["faces"], key=lambda f: f["w"])
            fd["target_cx"] = biggest["cx"]
            fd["target_side"] = biggest["side"]
            last_known_cx = fd["target_cx"]
            last_known_side = fd["target_side"]
        elif fd["shot_type"] == "close" and fd["faces"]:
            best = max(fd["faces"], key=lambda f: f["score"])
            fd["target_cx"] = best["cx"]
            fd["target_side"] = best["side"]
            last_known_cx = fd["target_cx"]
            last_known_side = fd["target_side"]
        else:
            # NOFACE: use last known position instead of defaulting to center
            if last_known_cx is not None:
                fd["target_cx"] = last_known_cx
                fd["target_side"] = last_known_side
            else:
                fd["target_cx"] = 0.5
                fd["target_side"] = "center"
    
    return frame_data


def generate_crop_timeline(frame_data, src_w, src_h, fps, total_frames, 
                            target_w=1080, target_h=1920, segment_duration=2.0):
    """Generate smooth crop positions with segment-based decisions."""
    
    crop_w = int(src_h * 9 / 16)
    crop_h = src_h
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(src_w * 16 / 9)
    
    # Interpolate target_cx for all frames
    frame_indices = [fd["frame"] for fd in frame_data]
    target_cxs = [fd["target_cx"] for fd in frame_data]
    all_cxs = np.interp(range(total_frames), frame_indices, target_cxs)
    
    # Smooth with large window to avoid jitter, but allow jumps on shot changes
    # Use segment-based averaging: decide per 2-second segment
    seg_frames = int(fps * segment_duration)
    segment_targets = []
    
    for i in range(0, total_frames, seg_frames):
        seg_cxs = all_cxs[i:i + seg_frames]
        # Use median (robust to outliers)
        median_cx = float(np.median(seg_cxs))
        
        # Snap to left or right face if clearly on one side
        if median_cx < 0.35:
            snap_cx = max(median_cx, crop_w / (2 * src_w))
        elif median_cx > 0.65:
            snap_cx = min(median_cx, 1.0 - crop_w / (2 * src_w))
        else:
            snap_cx = median_cx
        
        for j in range(i, min(i + seg_frames, total_frames)):
            segment_targets.append(snap_cx)
    
    # Smooth transitions between segments (ease in/out over 0.3s)
    transition_frames = int(fps * 0.3)
    smoothed = np.array(segment_targets[:total_frames], dtype=float)
    
    for i in range(1, len(smoothed)):
        if abs(smoothed[i] - smoothed[i-1]) > 0.01:
            # Smooth transition
            start = max(0, i - transition_frames)
            for j in range(start, min(i + transition_frames, len(smoothed))):
                alpha = (j - start) / (2 * transition_frames)
                alpha = min(1.0, max(0.0, alpha))
                # Ease in-out
                alpha = alpha * alpha * (3 - 2 * alpha)
                smoothed[j] = smoothed[start] * (1 - alpha) + smoothed[i] * alpha
    
    # Convert to crop_x positions
    crop_positions = []
    for cx in smoothed:
        crop_x = int(cx * src_w - crop_w / 2)
        crop_x = max(0, min(crop_x, src_w - crop_w))
        crop_positions.append(crop_x)
    
    return crop_positions, crop_w, crop_h


def reframe_with_dynamic_crop(input_path, output_path, crop_positions, crop_w, crop_h,
                                src_w, src_h, fps, target_w=1080, target_h=1920):
    """Apply dynamic crop using frame-by-frame processing."""
    
    cap = cv2.VideoCapture(input_path)
    
    # Use ffmpeg pipe for output
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
        
        # Get crop position for this frame
        if frame_idx < len(crop_positions):
            crop_x = crop_positions[frame_idx]
        else:
            crop_x = crop_positions[-1] if crop_positions else 0
        
        # Crop
        cropped = frame[0:crop_h, crop_x:crop_x + crop_w]
        
        try:
            proc.stdin.write(cropped.tobytes())
        except BrokenPipeError:
            break
        
        frame_idx += 1
    
    cap.release()
    proc.stdin.close()
    proc.wait()
    
    print(f"Output: {output_path} ({frame_idx} frames processed)")


def reframe_smart(input_path, output_path, whisper_json=None, time_offset=0.0):
    """Full smart reframe pipeline."""
    print(f"=== Smart Reframe: {os.path.basename(input_path)} ===")
    
    # 1. Analyze video
    frame_data, src_w, src_h, fps, total = analyze_video(input_path)
    
    # 2. Load Whisper if available
    whisper_segments = None
    if whisper_json and os.path.exists(whisper_json):
        whisper_segments = load_whisper_segments(whisper_json, time_offset)
        print(f"Loaded {len(whisper_segments)} Whisper segments")
    
    # 3. Assign speaker targets
    frame_data = assign_speaker_heuristic(frame_data, whisper_segments)
    
    # Log target distribution
    sides = [fd.get("target_side", "?") for fd in frame_data]
    from collections import Counter
    print(f"Target distribution: {dict(Counter(sides))}")
    
    # 4. Generate crop timeline
    crop_positions, crop_w, crop_h = generate_crop_timeline(
        frame_data, src_w, src_h, fps, total
    )
    
    # 5. Apply dynamic crop
    print(f"Applying dynamic crop ({crop_w}x{crop_h})...")
    reframe_with_dynamic_crop(
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

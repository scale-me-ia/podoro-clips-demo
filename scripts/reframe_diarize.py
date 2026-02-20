#!/usr/bin/env python3
"""Reframe v5: Real-time face tracking + diarization.

Strategy:
- Detect faces every frame (sampled)
- When 1 face: always crop to it (it's whoever is on screen)
- When 2+ faces: use diarization to pick the speaker's face
- When 0 faces: hold last known position
- Always smooth with exponential lerp
"""

import sys
import cv2
import numpy as np
import subprocess
import json
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "/tmp/blaze_face_short_range.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

LERP_INERTIA = 0.95  # Higher = smoother, less jitter
DETECT_CONFIDENCE = 0.25


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def load_diarization(json_path):
    with open(json_path) as f:
        return json.load(f)


def get_active_speaker(diarization, time_s):
    for seg in diarization:
        if seg["start"] <= time_s <= seg["end"]:
            return seg["speaker"]
    return None


def build_speaker_face_map(video_path, diarization, fps, sample_every=5):
    """Phase 1: Build a map of speaker→side by analyzing single-face frames only.
    
    Single-face frames are the gold standard: the visible face IS the speaker.
    """
    ensure_model()
    
    detector = vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            min_detection_confidence=DETECT_CONFIDENCE
        )
    )
    
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # speaker → list of cx when they're visible alone
    speaker_solo_cx = {}
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_every == 0:
            t = frame_idx / fps
            speaker = get_active_speaker(diarization, t)
            
            if speaker:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_image)
                
                if len(result.detections) == 1:
                    bbox = result.detections[0].bounding_box
                    cx = (bbox.origin_x + bbox.width / 2) / w
                    if speaker not in speaker_solo_cx:
                        speaker_solo_cx[speaker] = []
                    speaker_solo_cx[speaker].append(cx)
        
        frame_idx += 1
    
    cap.release()
    detector.close()
    
    speaker_cx = {}
    for spk, cxs in speaker_solo_cx.items():
        speaker_cx[spk] = float(np.median(cxs))
        print(f"  Speaker {spk}: solo median_cx={speaker_cx[spk]:.2f} ({len(cxs)} solo frames)")
    
    return speaker_cx


def reframe_with_diarization(input_path, output_path, diarization_path,
                              target_w=1080, target_h=1920):
    print(f"=== Reframe v5 (realtime + diarization): {os.path.basename(input_path)} ===")
    
    diarization = load_diarization(diarization_path)
    print(f"Loaded {len(diarization)} diarization segments")
    
    cap = cv2.VideoCapture(input_path)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    crop_w = int(src_h * 9 / 16)
    crop_h = src_h
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(src_w * 16 / 9)
    
    min_cx = crop_w / (2 * src_w)
    max_cx = 1.0 - min_cx
    
    # Phase 1: Learn speaker→position from solo frames
    print("Phase 1: Mapping speakers from solo frames...")
    speaker_cx = build_speaker_face_map(input_path, diarization, fps)
    
    # Phase 2: Real-time tracking with diarization guidance
    print("Phase 2: Real-time tracking...")
    
    ensure_model()
    detector = vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            min_detection_confidence=DETECT_CONFIDENCE
        )
    )
    
    cap = cv2.VideoCapture(input_path)
    
    # Output via ffmpeg pipe
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
    
    # Pre-scan: find the first face in the video to initialize crop
    print("  Pre-scanning for first face...")
    pre_cap = cv2.VideoCapture(input_path)
    pre_det = vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            min_detection_confidence=DETECT_CONFIDENCE
        )
    )
    start_cx = 0.5
    for _ in range(int(fps * 5)):  # scan first 5 seconds
        ret, frame = pre_cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = pre_det.detect(mp_image)
        if result.detections:
            biggest = max(result.detections, key=lambda d: d.bounding_box.width)
            bbox = biggest.bounding_box
            start_cx = max(min_cx, min(max_cx, (bbox.origin_x + bbox.width/2) / src_w))
            break
    pre_cap.release()
    pre_det.close()
    
    current_cx = start_cx
    last_face_cx = start_cx
    print(f"  Starting cx={start_cx:.2f}")
    sample_every = 2
    last_detection = None  # cache detection for non-sampled frames
    
    frame_idx = 0
    stats = {"jumps": []}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        t = frame_idx / fps
        speaker = get_active_speaker(diarization, t)
        
        # Detect faces (sampled)
        if frame_idx % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            
            faces = []
            for d in result.detections:
                bbox = d.bounding_box
                cx = (bbox.origin_x + bbox.width / 2) / src_w
                fw = bbox.width / src_w
                faces.append({"cx": cx, "w": fw})
            
            last_detection = faces
        else:
            faces = last_detection or []
        
        # Decide target_cx
        if len(faces) == 0:
            # No face: hold last known position
            target_cx = last_face_cx
        
        elif len(faces) == 1:
            # Single face: always follow it (it IS whoever is visible)
            target_cx = faces[0]["cx"]
            last_face_cx = target_cx
        
        else:
            # Multiple faces: use diarization to pick the right one
            if speaker and speaker in speaker_cx:
                # Find the face closest to this speaker's known position
                speaker_pos = speaker_cx[speaker]
                closest = min(faces, key=lambda f: abs(f["cx"] - speaker_pos))
                target_cx = closest["cx"]
            else:
                # Unknown speaker: pick the biggest face
                biggest = max(faces, key=lambda f: f["w"])
                target_cx = biggest["cx"]
            
            last_face_cx = target_cx
        
        # Clamp
        target_cx = max(min_cx, min(max_cx, target_cx))
        
        # Smooth
        old_cx = current_cx
        current_cx = LERP_INERTIA * current_cx + (1 - LERP_INERTIA) * target_cx
        
        # Convert to crop_x
        crop_x = int(current_cx * src_w - crop_w / 2)
        crop_x = max(0, min(crop_x, src_w - crop_w))
        
        # Track stats
        if frame_idx > 0:
            stats["jumps"].append(abs(crop_x - prev_crop_x))
        prev_crop_x = crop_x
        
        # Crop and write
        cropped = frame[0:crop_h, crop_x:crop_x + crop_w]
        try:
            proc.stdin.write(cropped.tobytes())
        except BrokenPipeError:
            break
        
        frame_idx += 1
    
    cap.release()
    detector.close()
    proc.stdin.close()
    proc.wait()
    
    if stats["jumps"]:
        print(f"Crop stats: max_jump={max(stats['jumps'])}px, avg_move={np.mean(stats['jumps']):.1f}px/frame")
    print(f"✅ Output: {output_path} ({frame_idx} frames)")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <input.mp4> <output.mp4> <diarization.json>")
        sys.exit(1)
    reframe_with_diarization(sys.argv[1], sys.argv[2], sys.argv[3])

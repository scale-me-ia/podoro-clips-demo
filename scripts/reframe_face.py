#!/usr/bin/env python3
"""Reframe 16:9 video to 9:16 using MediaPipe face detection (new Tasks API)."""

import sys
import cv2
import numpy as np
import subprocess
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

MODEL_PATH = "/tmp/blaze_face_short_range.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face detection model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def detect_faces_per_frame(video_path, sample_every=3):
    """Detect face positions using MediaPipe Tasks API."""
    ensure_model()
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )
    detector = vision.FaceDetector.create_from_options(options)
    
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    positions = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            
            if result.detections:
                best = result.detections[0]  # first = most confident
                bbox = best.bounding_box
                cx = (bbox.origin_x + bbox.width / 2) / w
                positions.append((frame_idx, cx))
            elif positions:
                positions.append((frame_idx, positions[-1][1]))
            else:
                positions.append((frame_idx, 0.5))
        
        frame_idx += 1
    
    cap.release()
    detector.close()
    return positions, w, h, fps, total

def smooth_positions(positions, window=15):
    xs = [p[1] for p in positions]
    smoothed = []
    for i in range(len(xs)):
        start = max(0, i - window)
        end = min(len(xs), i + window + 1)
        smoothed.append(sum(xs[start:end]) / (end - start))
    return [(positions[i][0], smoothed[i]) for i in range(len(positions))]

def reframe_video(input_path, output_path, target_w=1080, target_h=1920):
    print(f"Detecting faces in {input_path}...")
    positions, src_w, src_h, fps, total = detect_faces_per_frame(input_path)
    
    print(f"Source: {src_w}x{src_h} @ {fps}fps, {total} frames")
    print(f"Detected {len(positions)} face positions")
    
    positions = smooth_positions(positions)
    
    crop_w = int(src_h * 9 / 16)
    crop_h = src_h
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(src_w * 16 / 9)
    
    all_xs = np.interp(range(total), [p[0] for p in positions], [p[1] for p in positions])
    avg_x = float(np.mean(all_xs))
    
    crop_x = int(avg_x * src_w - crop_w / 2)
    crop_x = max(0, min(crop_x, src_w - crop_w))
    
    print(f"Crop: {crop_w}x{crop_h} @ x={crop_x} (face avg: {avg_x:.2f})")
    
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"crop={crop_w}:{crop_h}:{crop_x}:0,scale={target_w}:{target_h}",
        "-c:a", "copy", output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Output: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.mp4> <output.mp4>")
        sys.exit(1)
    reframe_video(sys.argv[1], sys.argv[2])

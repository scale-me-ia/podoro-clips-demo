#!/usr/bin/env python3
"""
Smart Reframing v3: 16:9 → 9:16 with YOLOv8 person detection + ByteTrack.
Fixes the core problem: MediaPipe face detection fails on wide shots.
YOLO detects full persons → works at any distance.
"""

import subprocess
import json
import sys
import os
import tempfile
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
OUT_W, OUT_H = 1080, 1920  # 9:16
SAMPLE_EVERY = 2           # run detection every N frames (YOLO is fast)

# Silence cutting
SILENCE_THRESHOLD = 0.4    # seconds — silences longer than this get trimmed
SILENCE_KEEP = 0.15        # keep this much silence (seconds)

# Encoding
CRF = "22"
AUDIO_BITRATE = "128k"

# Subject selection weights
W_SIZE = 0.4
W_CENTER = 0.2
W_CONTINUITY = 0.3
W_UNUSED = 0.1  # reserved for future speaker score

# Smooth crop
LERP_ALPHA = 0.06          # exponential smoothing factor
LERP_ALPHA_SWITCH = 0.12   # faster alpha when switching subjects
MAX_MOVE_FRAC = 0.05       # max 5% of source width per frame

# Hysteresis
HYSTERESIS_MARGIN = 0.15   # new subject must beat current by this margin
HYSTERESIS_FRAMES_1S = None  # computed from fps


# ──────────────────────────────────────────────────────────────────────────────
# 1. Person Detection + Tracking Pass
# ──────────────────────────────────────────────────────────────────────────────
def detect_persons_pass(video_path: str):
    """Run YOLOv8n person detection with ByteTrack tracking."""
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Source: {src_w}x{src_h}, {fps:.1f}fps, {total_frames} frames ({total_frames/fps:.1f}s)")

    # Per-frame detections: list of dicts {track_id, x1, y1, x2, y2, conf}
    all_detections = []
    frame_idx = 0
    last_dets = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_EVERY == 0:
            results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                  classes=[0], conf=0.3, verbose=False)
            dets = []
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu())
                    track_id = int(boxes.id[i].cpu()) if boxes.id is not None else -1
                    dets.append({
                        'track_id': track_id,
                        'x1': float(x1), 'y1': float(y1),
                        'x2': float(x2), 'y2': float(y2),
                        'conf': conf,
                    })
            last_dets = dets
            all_detections.append(dets)
        else:
            # Interpolate: use last detection
            all_detections.append(last_dets)

        frame_idx += 1

    cap.release()
    return all_detections, fps, total_frames, src_w, src_h


# ──────────────────────────────────────────────────────────────────────────────
# 2. Subject Selection + Smooth Crop
# ──────────────────────────────────────────────────────────────────────────────
def compute_smooth_crop(all_detections, fps, src_w, src_h):
    """Select subject per frame, compute smoothed crop_x."""
    crop_w = int(src_h * (9 / 16))
    crop_w = min(crop_w, src_w)
    crop_h = src_h

    total_frames = len(all_detections)
    hysteresis_frames = int(fps * 1.0)  # 1 second
    max_move_px = int(src_w * MAX_MOVE_FRAC)

    current_subject_id = -1
    candidate_id = -1
    candidate_frames = 0

    # Start at center
    crop_x = float((src_w - crop_w) / 2)
    crop_positions = []
    active_alpha = LERP_ALPHA

    # Stats
    frames_with_person = 0
    moves = []

    for i in range(total_frames):
        dets = all_detections[i]
        frame_area = src_w * src_h

        if not dets:
            # Fallback: hold position, drift to center slowly
            target_x = crop_x  # hold
            crop_positions.append(int(round(crop_x)))
            continue

        frames_with_person += 1

        # Score each detection
        best_id = -1
        best_score = -1
        best_cx = src_w / 2

        for d in dets:
            cx = (d['x1'] + d['x2']) / 2
            cy = (d['y1'] + d['y2']) / 2
            w = d['x2'] - d['x1']
            h = d['y2'] - d['y1']
            area = w * h

            size_score = min(area / frame_area * 10, 1.0)  # normalize, cap at 1
            center_score = 1.0 - abs(cx - src_w / 2) / (src_w / 2)
            continuity_score = 1.0 if d['track_id'] == current_subject_id else 0.0

            total = (size_score * W_SIZE +
                     center_score * W_CENTER +
                     continuity_score * W_CONTINUITY)

            if total > best_score:
                best_score = total
                best_id = d['track_id']
                best_cx = cx

        # Hysteresis: don't switch unless new subject is consistently better
        if best_id != current_subject_id:
            if best_id == candidate_id:
                candidate_frames += 1
            else:
                candidate_id = best_id
                candidate_frames = 1

            if candidate_frames >= hysteresis_frames:
                current_subject_id = best_id
                active_alpha = LERP_ALPHA_SWITCH  # faster transition for subject switch
            # Use current subject's position if not switching yet
            for d in dets:
                if d['track_id'] == current_subject_id:
                    best_cx = (d['x1'] + d['x2']) / 2
                    break
        else:
            candidate_id = -1
            candidate_frames = 0
            active_alpha = LERP_ALPHA

        # Target: center the subject
        target_x = best_cx - crop_w / 2
        target_x = max(0, min(target_x, src_w - crop_w))

        # Smooth: lerp
        prev_crop_x = crop_x
        crop_x = crop_x + active_alpha * (target_x - crop_x)

        # Clamp max movement
        delta = crop_x - prev_crop_x
        if abs(delta) > max_move_px:
            crop_x = prev_crop_x + max_move_px * (1 if delta > 0 else -1)

        crop_x = max(0.0, min(crop_x, float(src_w - crop_w)))
        crop_positions.append(int(round(crop_x)))

        if i > 0:
            move = abs(crop_positions[-1] - crop_positions[-2])
            moves.append(move)

    stats = {
        'frames_with_person': frames_with_person,
        'total_frames': total_frames,
        'pct_person': 100 * frames_with_person / max(total_frames, 1),
        'avg_move': np.mean(moves) if moves else 0,
        'max_jump': max(moves) if moves else 0,
    }

    return crop_positions, crop_w, crop_h, stats


# ──────────────────────────────────────────────────────────────────────────────
# 3. Render cropped video
# ──────────────────────────────────────────────────────────────────────────────
def render_cropped(video_path, crop_positions, crop_w, crop_h, fps, tmp_video):
    """Read source, crop per frame, write to tmp file."""
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (OUT_W, OUT_H))

    frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(crop_positions):
            cx = crop_positions[frame_idx]
        else:
            cx = crop_positions[-1] if crop_positions else 0

        cx = max(0, min(cx, frame.shape[1] - crop_w))
        cropped = frame[0:crop_h, cx:cx + crop_w]
        resized = cv2.resize(cropped, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
        writer.write(resized)
        written += 1
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  Wrote {written} cropped frames")
    return written


# ──────────────────────────────────────────────────────────────────────────────
# 4. Silence detection & cutting (from reframe_v2)
# ──────────────────────────────────────────────────────────────────────────────
def detect_silences(video_path):
    cmd = [
        "ffmpeg", "-i", video_path, "-af",
        f"silencedetect=noise=-30dB:d={SILENCE_THRESHOLD}",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    silences = []
    starts = re.findall(r"silence_start: ([\d.]+)", stderr)
    ends = re.findall(r"silence_end: ([\d.]+)", stderr)
    for s, e in zip(starts, ends):
        silences.append((float(s), float(e)))

    print(f"  Found {len(silences)} silences > {SILENCE_THRESHOLD}s")
    return silences


def build_silence_cut_filter(silences, total_duration):
    if not silences:
        return None

    keep_segments = []
    pos = 0.0
    for s_start, s_end in silences:
        silence_dur = s_end - s_start
        trim_amount = silence_dur - SILENCE_KEEP
        if trim_amount <= 0:
            continue
        keep_end = s_start + SILENCE_KEEP / 2
        if keep_end > pos:
            keep_segments.append((pos, keep_end))
        pos = s_end - SILENCE_KEEP / 2

    if pos < total_duration:
        keep_segments.append((pos, total_duration))

    merged = []
    for seg in keep_segments:
        if merged and seg[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)

    return merged


def get_duration(video_path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def final_encode(tmp_video, audio_source, output_path, keep_segments):
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if keep_segments is None:
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_video, "-i", audio_source,
            "-c:v", "libx264", "-crf", CRF, "-preset", "medium",
            "-c:a", "aac", "-b:a", AUDIO_BITRATE,
            "-map", "0:v:0", "-map", "1:a:0",
            "-movflags", "+faststart",
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return

    n = len(keep_segments)
    filter_parts = []
    concat_inputs = []

    for i, (start, end) in enumerate(keep_segments):
        filter_parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{i}];"
        )
        filter_parts.append(
            f"[1:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}];"
        )
        concat_inputs.append(f"[v{i}][a{i}]")

    filter_complex = "".join(filter_parts)
    filter_complex += "".join(concat_inputs) + f"concat=n={n}:v=1:a=1[outv][outa]"

    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_video, "-i", audio_source,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-crf", CRF, "-preset", "medium",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        "-movflags", "+faststart",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error:\n{result.stderr[-500:]}")
        raise RuntimeError("FFmpeg encoding failed")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.mp4> <output.mp4>")
        sys.exit(1)

    input_video = sys.argv[1]
    output_video = sys.argv[2]

    print(f"=== Smart Reframe v3 (YOLO + ByteTrack): {input_video} → {output_video} ===\n")

    # Step 1: Person detection + tracking
    print("[1/4] Running YOLOv8n person detection + ByteTrack tracking...")
    all_detections, fps, total_frames, src_w, src_h = detect_persons_pass(input_video)

    # Step 2: Subject selection + smooth crop
    print("\n[2/4] Computing subject selection + smooth crop...")
    crop_positions, crop_w, crop_h, stats = compute_smooth_crop(
        all_detections, fps, src_w, src_h
    )
    print(f"  Persons detected in {stats['frames_with_person']}/{stats['total_frames']} frames "
          f"({stats['pct_person']:.1f}%)")
    print(f"  Crop: {crop_w}x{crop_h}")
    print(f"  Avg move/frame: {stats['avg_move']:.1f}px | Max jump: {stats['max_jump']}px")

    # Step 3: Render cropped video
    print("\n[3/4] Rendering cropped video...")
    tmp_dir = tempfile.mkdtemp(prefix="reframe_v3_")
    tmp_video = os.path.join(tmp_dir, "cropped.mp4")
    render_cropped(input_video, crop_positions, crop_w, crop_h, fps, tmp_video)

    # Step 4: Silence detection + final encode
    print("\n[4/4] Detecting silences & final encode...")
    total_duration = get_duration(input_video)
    silences = detect_silences(input_video)
    keep_segments = build_silence_cut_filter(silences, total_duration)

    if keep_segments:
        kept_dur = sum(e - s for s, e in keep_segments)
        print(f"  {total_duration:.1f}s → {kept_dur:.1f}s (cut {total_duration - kept_dur:.1f}s)")
    else:
        print("  No significant silences")

    print("  Encoding final output...")
    final_encode(tmp_video, input_video, output_video, keep_segments)

    # Cleanup
    os.remove(tmp_video)
    os.rmdir(tmp_dir)

    # Report
    out_duration = get_duration(output_video)
    file_size = os.path.getsize(output_video) / (1024 * 1024)

    print(f"\n=== Done ===")
    print(f"Output: {output_video}")
    print(f"Duration: {out_duration:.1f}s | Size: {file_size:.1f} MB")
    print(f"\n--- Stats ---")
    print(f"Person detected: {stats['pct_person']:.1f}% of frames")
    print(f"Avg move/frame: {stats['avg_move']:.1f}px")
    print(f"Max jump: {stats['max_jump']}px")

    return stats


if __name__ == "__main__":
    main()

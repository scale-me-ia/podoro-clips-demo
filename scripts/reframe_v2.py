#!/usr/bin/env python3
"""
Smart Reframing v2: 16:9 → 9:16 with STRICT face centering.
Face MUST dominate the frame — centered and prominent.
"""

import subprocess
import json
import sys
import os
import tempfile
import re
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
INPUT_VIDEO = "/tmp/sp-clips/sp_rank1_raw.mp4"
OUTPUT_VIDEO = "/tmp/podoro-clips-v3/agent-cadrage/output_reframe_v2.mp4"
DEBUG_FRAME_PATH = "/tmp/podoro-clips-v3/agent-cadrage/debug_frame.png"
FACE_MODEL_PATH = "/tmp/blaze_face_short_range.tflite"

OUT_W, OUT_H = 1080, 1920  # 9:16
SAMPLE_EVERY = 3           # sample face detection every 3 frames
HOLD_DURATION_S = 1.5      # hold last position for up to 1.5s when face lost
SILENCE_THRESHOLD = 0.4    # seconds — silences longer than this get trimmed
SILENCE_KEEP = 0.15        # keep this much silence (seconds)

CRF = "22"
AUDIO_BITRATE = "128k"

# Face centering: face center must be within 10% of crop center horizontally
MAX_CENTER_OFFSET_FRAC = 0.10


# ──────────────────────────────────────────────────────────────────────────────
# 1. Face Detection Pass
# ──────────────────────────────────────────────────────────────────────────────
def detect_faces_pass(video_path: str):
    """Run MediaPipe face detection, sampling every N frames."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Source: {src_w}x{src_h}, {fps}fps, {total_frames} frames ({total_frames/fps:.1f}s)")

    options = mp_vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        min_detection_confidence=0.3,
    )
    face_det = mp_vision.FaceDetector.create_from_options(options)

    # per-frame: list of (cx, cy, w, h, score) in pixel coords, or None if not sampled
    raw_detections = [None] * total_frames
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_EVERY == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = face_det.detect(mp_image)
            faces = []
            for det in results.detections:
                bb = det.bounding_box
                cx = bb.origin_x + bb.width / 2
                cy = bb.origin_y + bb.height / 2
                fw = bb.width
                fh = bb.height
                score = det.categories[0].score if det.categories else 0.5
                faces.append((cx, cy, fw, fh, score))
            raw_detections[frame_idx] = faces

        frame_idx += 1

    cap.release()
    face_det.close()

    # Interpolate: fill non-sampled frames with nearest sampled data
    detections = [None] * total_frames
    last_sampled = None
    for i in range(total_frames):
        if raw_detections[i] is not None:
            last_sampled = raw_detections[i]
        detections[i] = last_sampled if last_sampled is not None else []

    return detections, fps, total_frames, src_w, src_h


# ──────────────────────────────────────────────────────────────────────────────
# 2. Pick active speaker per frame — strict filtering
# ──────────────────────────────────────────────────────────────────────────────
# Max plausible face size: 50% of source height (faces at podcast distance)
MAX_FACE_FRAC = 0.50

def pick_active_face(faces, src_w, src_h, prev_face_cx=None, relaxed=False):
    """Pick the most likely speaker face.
    - Filter out unreasonably large bounding boxes (false positives)
    - Prefer high-confidence detections
    - If prev_face_cx given, prefer face near previous position (tracking continuity)
    - If relaxed=True, allow larger bboxes (scene change recovery mode)
    """
    if not faces:
        return None

    max_face_dim = src_h * (0.70 if relaxed else MAX_FACE_FRAC)

    # Filter: reject bounding boxes larger than max plausible face size
    plausible = [(cx, cy, fw, fh, s) for cx, cy, fw, fh, s in faces
                 if fw <= max_face_dim and fh <= max_face_dim]

    if not plausible:
        return None

    # Prefer high-confidence faces (>= 0.5)
    high_conf = [f for f in plausible if f[4] >= 0.5]
    candidates = high_conf if high_conf else plausible

    # Score: face area (biggest = closest to camera = active speaker) * confidence
    # For podcasts, the biggest face is almost always the one talking
    best = None
    best_score = -1
    for cx, cy, fw, fh, conf in candidates:
        face_area = fw * fh
        # Normalize area to 0-1 range (relative to source)
        area_norm = face_area / (src_w * src_h)
        score = area_norm * 0.6 + conf * 0.4
        # Moderate proximity bonus for tracking continuity (avoid rapid switches)
        if prev_face_cx is not None:
            dist = abs(cx - prev_face_cx)
            proximity = max(0, 1.0 - dist / 500.0)
            score *= (0.5 + 0.5 * proximity)
        if score > best_score:
            best_score = score
            best = (cx, cy, fw, fh)

    # Reject if best score is too low (likely a false positive far from tracked face)
    if best is not None and best_score < 0.15:
        return None

    return best


# ──────────────────────────────────────────────────────────────────────────────
# 3. Compute per-frame crop_x with strict face centering
# ──────────────────────────────────────────────────────────────────────────────
def compute_crop_positions(detections, fps, src_w, src_h):
    """Compute crop_x for each frame with strict face centering."""
    crop_w = int(src_h * (9 / 16))  # 607 for 1080p source
    crop_h = src_h
    crop_w = min(crop_w, src_w)

    total_frames = len(detections)
    hold_frames = int(fps * HOLD_DURATION_S)

    # Per-frame: compute ideal crop_x strictly centering on face
    crop_x_per_frame = [None] * total_frames
    face_info_per_frame = [None] * total_frames  # for debug output

    last_valid_crop_x = None
    last_valid_face_cx = None
    frames_since_face = 0

    for i in range(total_frames):
        # Recovery mode: after losing face for much longer than hold time,
        # assume scene change — reset tracking bias and relax size filter.
        # Hold time (1.5s) keeps position stable during brief occlusions.
        # Recovery (3s+) handles actual scene changes.
        recovery_frames = int(fps * 3.0)
        use_prev_cx = last_valid_face_cx
        recovery = frames_since_face > recovery_frames
        if recovery:
            use_prev_cx = None

        face = pick_active_face(detections[i], src_w, src_h,
                                prev_face_cx=use_prev_cx, relaxed=recovery)

        if face is not None:
            cx, cy, fw, fh = face
            face_info_per_frame[i] = face

            # STRICT centering: face center = crop center
            crop_x = int(cx - crop_w / 2)
            # Clamp to valid range
            crop_x = max(0, min(crop_x, src_w - crop_w))

            crop_x_per_frame[i] = crop_x
            last_valid_crop_x = crop_x
            last_valid_face_cx = cx
            frames_since_face = 0
        else:
            frames_since_face += 1
            if last_valid_crop_x is not None:
                # Hold last position (within or beyond hold time — never jump to center)
                crop_x_per_frame[i] = last_valid_crop_x
            else:
                # No face ever seen — default to center
                crop_x_per_frame[i] = (src_w - crop_w) // 2

    return crop_x_per_frame, face_info_per_frame, crop_w, crop_h


# ──────────────────────────────────────────────────────────────────────────────
# 4. Build segments from per-frame positions (group stable positions)
# ──────────────────────────────────────────────────────────────────────────────
def build_segments(crop_x_per_frame, face_info_per_frame, fps, src_w, crop_w):
    """Group consecutive frames with similar crop_x into segments."""
    total_frames = len(crop_x_per_frame)
    min_seg_frames = int(fps * 1.5)
    SHIFT_THRESHOLD = crop_w * 0.12  # 12% of crop width triggers new segment

    segments = []
    seg_start = 0
    seg_xs = [crop_x_per_frame[0]]
    seg_faces = [face_info_per_frame[0]]

    for i in range(1, total_frames):
        seg_median = int(np.median(seg_xs))
        diff = abs(crop_x_per_frame[i] - seg_median)

        if diff > SHIFT_THRESHOLD and (i - seg_start) >= min_seg_frames:
            # Commit current segment — use median crop_x
            crop_x = int(np.median(seg_xs))
            crop_x = max(0, min(crop_x, src_w - crop_w))

            # Compute avg face info for this segment
            valid_faces = [f for f in seg_faces if f is not None]
            avg_face_cx = np.mean([f[0] for f in valid_faces]) if valid_faces else None
            avg_face_w = np.mean([f[2] for f in valid_faces]) if valid_faces else None

            segments.append({
                'start': seg_start, 'end': i, 'crop_x': crop_x,
                'avg_face_cx': avg_face_cx, 'avg_face_w': avg_face_w,
            })
            seg_start = i
            seg_xs = [crop_x_per_frame[i]]
            seg_faces = [face_info_per_frame[i]]
        else:
            seg_xs.append(crop_x_per_frame[i])
            seg_faces.append(face_info_per_frame[i])

    # Final segment
    crop_x = int(np.median(seg_xs))
    crop_x = max(0, min(crop_x, src_w - crop_w))
    valid_faces = [f for f in seg_faces if f is not None]
    avg_face_cx = np.mean([f[0] for f in valid_faces]) if valid_faces else None
    avg_face_w = np.mean([f[2] for f in valid_faces]) if valid_faces else None

    segments.append({
        'start': seg_start, 'end': total_frames, 'crop_x': crop_x,
        'avg_face_cx': avg_face_cx, 'avg_face_w': avg_face_w,
    })

    # Merge short segments
    merged = []
    for seg in segments:
        if merged and (seg['end'] - seg['start']) < min_seg_frames:
            merged[-1]['end'] = seg['end']
        else:
            merged.append(seg)

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 5. Save debug frame at frame 60
# ──────────────────────────────────────────────────────────────────────────────
def save_debug_frame(video_path, segments, crop_w, crop_h, face_info_per_frame):
    """Save frame 60 with crop rectangle and face box drawn on original."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("  WARNING: Could not read frame 60 for debug")
        return

    # Find the segment for frame 60
    seg = None
    for s in segments:
        if s['start'] <= 60 < s['end']:
            seg = s
            break
    if seg is None:
        seg = segments[0]

    crop_x = seg['crop_x']

    # Draw crop rectangle (green)
    cv2.rectangle(frame, (crop_x, 0), (crop_x + crop_w, crop_h), (0, 255, 0), 3)

    # Draw crop center line (green dashed)
    center_x = crop_x + crop_w // 2
    for y in range(0, crop_h, 20):
        cv2.line(frame, (center_x, y), (center_x, min(y + 10, crop_h)), (0, 255, 0), 2)

    # Draw face bounding box if detected at frame 60
    face = face_info_per_frame[60] if 60 < len(face_info_per_frame) else None
    if face is not None:
        cx, cy, fw, fh = face
        x1 = int(cx - fw / 2)
        y1 = int(cy - fh / 2)
        x2 = int(cx + fw / 2)
        y2 = int(cy + fh / 2)
        # Face box (red)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # Face center (red dot)
        cv2.circle(frame, (int(cx), int(cy)), 8, (0, 0, 255), -1)
        # Label
        cv2.putText(frame, f"Face: ({int(cx)}, {int(cy)})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Label crop
    cv2.putText(frame, f"Crop: x={crop_x}, w={crop_w}", (crop_x + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(DEBUG_FRAME_PATH, frame)
    print(f"  Debug frame saved: {DEBUG_FRAME_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Render cropped video (no audio)
# ──────────────────────────────────────────────────────────────────────────────
def render_cropped(video_path, segments, crop_w, crop_h, fps, total_frames, tmp_video, crop_x_per_frame=None):
    """Read source, crop per frame (or per segment), write to tmp file."""
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (OUT_W, OUT_H))

    seg_idx = 0
    frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use per-frame crop_x if available (more accurate), else segment-level
        if crop_x_per_frame is not None and frame_idx < len(crop_x_per_frame):
            crop_x = crop_x_per_frame[frame_idx]
        else:
            while seg_idx < len(segments) - 1 and frame_idx >= segments[seg_idx]['end']:
                seg_idx += 1
            crop_x = segments[seg_idx]['crop_x']

        crop_x = max(0, min(crop_x, frame.shape[1] - crop_w))
        cropped = frame[0:crop_h, crop_x:crop_x + crop_w]

        # Resize to output dimensions
        resized = cv2.resize(cropped, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
        writer.write(resized)
        written += 1
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  Wrote {written} cropped frames to temp file")
    return written


# ──────────────────────────────────────────────────────────────────────────────
# 7. Silence detection & cutting
# ──────────────────────────────────────────────────────────────────────────────
def detect_silences(video_path):
    """Use ffmpeg silencedetect to find silent segments."""
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
    """Build keep-segments list trimming silences to SILENCE_KEEP seconds."""
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

    # Merge overlapping
    merged = []
    for seg in keep_segments:
        if merged and seg[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)

    return merged


def final_encode(tmp_video, audio_source, output_path, keep_segments):
    """Combine cropped video with original audio, apply silence cuts."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
        print(f"FFmpeg error:\n{result.stderr}")
        raise RuntimeError("FFmpeg encoding failed")


def get_duration(video_path):
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    input_video = sys.argv[1] if len(sys.argv) > 1 else INPUT_VIDEO
    output_video = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_VIDEO

    print(f"=== Smart Reframe v2: {input_video} → {output_video} ===\n")

    # Step 1: Face detection
    print("[1/6] Running face detection (sample every 3 frames)...")
    detections, fps, total_frames, src_w, src_h = detect_faces_pass(input_video)

    frames_with_faces = sum(1 for d in detections if d and len(d) > 0)
    print(f"  Faces found in {frames_with_faces}/{total_frames} frames "
          f"({100*frames_with_faces/total_frames:.0f}%)\n")

    # Step 2: Compute per-frame crop positions (strict centering)
    print("[2/6] Computing per-frame crop positions (strict face centering)...")
    crop_x_per_frame, face_info_per_frame, crop_w, crop_h = compute_crop_positions(
        detections, fps, src_w, src_h
    )
    print(f"  Crop window: {crop_w}x{crop_h} from {src_w}x{src_h}\n")

    # Step 3: Build segments
    print("[3/6] Building crop segments...")
    segments = build_segments(crop_x_per_frame, face_info_per_frame, fps, src_w, crop_w)

    print(f"  {len(segments)} segment(s):")
    for i, seg in enumerate(segments):
        dur = (seg['end'] - seg['start']) / fps
        face_cx_str = f"{seg['avg_face_cx']:.0f}" if seg['avg_face_cx'] else "N/A"
        face_w_str = f"{seg['avg_face_w']:.0f}" if seg['avg_face_w'] else "N/A"
        crop_center = seg['crop_x'] + crop_w // 2

        # Check centering quality
        if seg['avg_face_cx'] is not None:
            offset = abs(seg['avg_face_cx'] - crop_center)
            offset_pct = offset / crop_w * 100
            center_ok = "OK" if offset_pct <= MAX_CENTER_OFFSET_FRAC * 100 else "WARN"
            face_ratio = seg['avg_face_w'] / crop_w * 100 if seg['avg_face_w'] else 0
        else:
            offset_pct = 0
            center_ok = "N/A"
            face_ratio = 0

        print(f"    Seg {i+1}: frames {seg['start']}-{seg['end']} ({dur:.1f}s) "
              f"crop_x={seg['crop_x']} | "
              f"face_cx={face_cx_str} offset={offset_pct:.1f}% [{center_ok}] | "
              f"face_w/crop_w={face_ratio:.0f}%")
    print()

    # Step 4: Save debug frame
    print("[4/6] Saving debug frame (frame 60)...")
    save_debug_frame(input_video, segments, crop_w, crop_h, face_info_per_frame)
    print()

    # Step 5: Render cropped video
    print("[5/6] Rendering cropped video...")
    tmp_dir = tempfile.mkdtemp(prefix="reframe_v2_")
    tmp_video = os.path.join(tmp_dir, "cropped.mp4")
    written = render_cropped(input_video, segments, crop_w, crop_h, fps, total_frames, tmp_video, crop_x_per_frame=crop_x_per_frame)
    print()

    # Step 6: Silence detection + final encode
    print("[6/6] Detecting silences & final encode...")
    total_duration = get_duration(input_video)
    silences = detect_silences(input_video)
    keep_segments = build_silence_cut_filter(silences, total_duration)

    if keep_segments:
        kept_dur = sum(e - s for s, e in keep_segments)
        cut_dur = total_duration - kept_dur
        print(f"  Original: {total_duration:.1f}s → After cuts: {kept_dur:.1f}s "
              f"(removed {cut_dur:.1f}s)")
    else:
        print("  No significant silences to cut")

    print("  Encoding final output...")
    final_encode(tmp_video, input_video, output_video, keep_segments)

    # Cleanup
    os.remove(tmp_video)
    os.rmdir(tmp_dir)

    # Report
    out_duration = get_duration(output_video)
    out_cap = cv2.VideoCapture(output_video)
    out_frames = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w = int(out_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(out_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_cap.release()

    file_size = os.path.getsize(output_video) / (1024 * 1024)

    print(f"\n=== Done ===")
    print(f"Output: {output_video}")
    print(f"Resolution: {out_w}x{out_h}")
    print(f"Frames: {out_frames}")
    print(f"Duration: {out_duration:.1f}s")
    print(f"File size: {file_size:.1f} MB")


if __name__ == "__main__":
    main()

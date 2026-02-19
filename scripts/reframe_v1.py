#!/usr/bin/env python3
"""
Smart Reframing: 16:9 → 9:16 with speaker tracking and silence cutting.
Uses MediaPipe Face Detection to track speakers and crop intelligently.
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
OUTPUT_VIDEO = "/tmp/podoro-clips-v3/agent-cadrage/output_reframe.mp4"
FACE_MODEL_PATH = "/tmp/blaze_face_short_range.tflite"

OUT_W, OUT_H = 1080, 1920  # 9:16
SAMPLE_EVERY = 4           # sample face detection every N frames
MIN_SEGMENT_FRAMES = None  # computed from fps (1.5s)
SILENCE_THRESHOLD = 0.4    # seconds — silences longer than this get trimmed
SILENCE_KEEP = 0.15        # keep this much silence (seconds)
FACE_SAFETY_ZONE = 0.60    # face should be within middle 60% of crop

CRF = "22"
AUDIO_BITRATE = "128k"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Face Detection Pass
# ──────────────────────────────────────────────────────────────────────────────
def detect_faces_pass(video_path: str):
    """Run MediaPipe face detection, sampling every N frames. Returns per-frame face data."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Source: {src_w}x{src_h}, {fps}fps, {total_frames} frames ({total_frames/fps:.1f}s)")

    # MediaPipe Tasks API (v0.10+)
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
                # bb has origin_x, origin_y, width, height in pixels
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
# 2. Pick active speaker per frame
# ──────────────────────────────────────────────────────────────────────────────
def pick_active_face(faces, src_w, src_h):
    """From a list of (cx, cy, w, h, score), pick the most likely speaker.
    Heuristic: largest face area * centrality weight."""
    if not faces:
        return None
    best = None
    best_score = -1
    center_x = src_w / 2
    for (cx, cy, fw, fh, conf) in faces:
        area = fw * fh
        # centrality: 1.0 at center, drops toward edges
        centrality = 1.0 - abs(cx - center_x) / center_x
        score = area * (0.5 + 0.5 * centrality) * conf
        if score > best_score:
            best_score = score
            best = (cx, cy, fw, fh)
    return best


# ──────────────────────────────────────────────────────────────────────────────
# 3. Compute crop segments (hard cuts, no smooth panning)
# ──────────────────────────────────────────────────────────────────────────────
def compute_crop_segments(detections, fps, src_w, src_h):
    """Produce a list of (start_frame, end_frame, crop_x) segments."""
    min_seg_frames = int(fps * 1.5)  # 1.5 seconds minimum segment
    crop_aspect = OUT_W / OUT_H      # 9/16 = 0.5625
    crop_w = int(src_h * crop_aspect) # width of crop region in source pixels
    crop_h = src_h                    # use full height

    # Ensure crop_w doesn't exceed source
    crop_w = min(crop_w, src_w)

    total_frames = len(detections)

    # For each frame, compute ideal crop_x centered on active face
    ideal_x = [None] * total_frames
    last_known_x = src_w // 2  # default center

    for i in range(total_frames):
        face = pick_active_face(detections[i], src_w, src_h)
        if face is not None:
            cx, cy, fw, fh = face
            # Center crop on face, but ensure face is within safety zone
            # Safety: face cx should be within middle 60% of crop
            # → face cx in [crop_x + 0.2*crop_w, crop_x + 0.8*crop_w]
            target_x = cx - crop_w / 2
            # Clamp to valid range
            target_x = max(0, min(target_x, src_w - crop_w))

            # Verify safety margin
            face_rel = (cx - target_x) / crop_w
            margin = (1 - FACE_SAFETY_ZONE) / 2  # 0.20
            if face_rel < margin:
                target_x = cx - margin * crop_w
            elif face_rel > 1 - margin:
                target_x = cx - (1 - margin) * crop_w
            target_x = max(0, min(target_x, src_w - crop_w))

            ideal_x[i] = int(target_x)
            last_known_x = int(cx)
        else:
            # No face: use last known position
            target_x = last_known_x - crop_w / 2
            target_x = max(0, min(target_x, src_w - crop_w))
            ideal_x[i] = int(target_x)

    # Build segments: group consecutive frames with similar crop_x
    # A "cut" happens when crop position shifts significantly
    SHIFT_THRESHOLD = crop_w * 0.15  # 15% of crop width to trigger a cut

    segments = []
    seg_start = 0
    seg_xs = [ideal_x[0]]

    for i in range(1, total_frames):
        seg_median = int(np.median(seg_xs))
        diff = abs(ideal_x[i] - seg_median)

        if diff > SHIFT_THRESHOLD and (i - seg_start) >= min_seg_frames:
            # Commit current segment
            crop_x = int(np.median(seg_xs))
            crop_x = max(0, min(crop_x, src_w - crop_w))
            segments.append((seg_start, i, crop_x))
            seg_start = i
            seg_xs = [ideal_x[i]]
        else:
            seg_xs.append(ideal_x[i])

    # Final segment
    crop_x = int(np.median(seg_xs))
    crop_x = max(0, min(crop_x, src_w - crop_w))
    segments.append((seg_start, total_frames, crop_x))

    # Enforce minimum segment duration: merge short segments
    merged = []
    for seg in segments:
        if merged and (seg[1] - seg[0]) < min_seg_frames:
            # Merge with previous
            prev = merged[-1]
            merged[-1] = (prev[0], seg[1], prev[2])
        else:
            merged.append(seg)

    return merged, crop_w, crop_h


# ──────────────────────────────────────────────────────────────────────────────
# 4. Render cropped video (no audio)
# ──────────────────────────────────────────────────────────────────────────────
def render_cropped(video_path: str, segments, crop_w, crop_h, fps, total_frames, tmp_video: str):
    """Read source, crop per segment, write to tmp file."""
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

        # Find current segment
        while seg_idx < len(segments) - 1 and frame_idx >= segments[seg_idx][1]:
            seg_idx += 1

        _, _, crop_x = segments[seg_idx]
        cropped = frame[0:crop_h, crop_x:crop_x + crop_w]

        # Resize to output dimensions
        resized = cv2.resize(cropped, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
        writer.write(resized)
        written += 1
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Wrote {written} cropped frames to temp file")
    return written


# ──────────────────────────────────────────────────────────────────────────────
# 5. Silence detection & cutting
# ──────────────────────────────────────────────────────────────────────────────
def detect_silences(video_path: str):
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

    print(f"Found {len(silences)} silences > {SILENCE_THRESHOLD}s")
    return silences


def build_silence_cut_filter(silences, total_duration):
    """Build ffmpeg filter to trim silences, keeping SILENCE_KEEP seconds of each."""
    if not silences:
        return None, None

    # Build list of segments to KEEP
    keep_segments = []
    pos = 0.0

    for s_start, s_end in silences:
        silence_dur = s_end - s_start
        trim_amount = silence_dur - SILENCE_KEEP
        if trim_amount <= 0:
            continue

        # Keep from pos to silence_start + half of SILENCE_KEEP
        keep_end = s_start + SILENCE_KEEP / 2
        if keep_end > pos:
            keep_segments.append((pos, keep_end))

        # Resume from silence_end - half of SILENCE_KEEP
        pos = s_end - SILENCE_KEEP / 2

    # Keep remainder
    if pos < total_duration:
        keep_segments.append((pos, total_duration))

    # Merge any overlapping segments
    merged = []
    for seg in keep_segments:
        if merged and seg[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], seg[1]))
        else:
            merged.append(seg)

    return merged


def final_encode(tmp_video: str, audio_source: str, output_path: str, keep_segments):
    """Combine cropped video with original audio, apply silence cuts, encode final output."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if keep_segments is None:
        # No silence cutting needed, just mux
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

    # Build concat filter with silence cuts
    # Create a complex filter that selects segments from both video and audio
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


# ──────────────────────────────────────────────────────────────────────────────
# 6. Get video duration
# ──────────────────────────────────────────────────────────────────────────────
def get_duration(video_path: str) -> float:
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

    print(f"=== Smart Reframe: {input_video} → {output_video} ===\n")

    # Step 1: Face detection
    print("[1/5] Running face detection...")
    detections, fps, total_frames, src_w, src_h = detect_faces_pass(input_video)

    frames_with_faces = sum(1 for d in detections if d and len(d) > 0)
    print(f"  Faces detected in {frames_with_faces}/{total_frames} frames "
          f"({100*frames_with_faces/total_frames:.0f}%)\n")

    # Step 2: Compute crop segments
    print("[2/5] Computing crop segments...")
    segments, crop_w, crop_h = compute_crop_segments(detections, fps, src_w, src_h)

    print(f"  Crop region: {crop_w}x{crop_h} from {src_w}x{src_h}")
    print(f"  {len(segments)} segments:")
    for i, (s, e, x) in enumerate(segments):
        dur = (e - s) / fps
        print(f"    Segment {i+1}: frames {s}-{e} ({dur:.1f}s), crop_x={x}")
    print()

    # Step 3: Render cropped video
    print("[3/5] Rendering cropped video...")
    tmp_dir = tempfile.mkdtemp(prefix="reframe_")
    tmp_video = os.path.join(tmp_dir, "cropped.mp4")
    written = render_cropped(input_video, segments, crop_w, crop_h, fps, total_frames, tmp_video)
    print()

    # Step 4: Detect silences
    print("[4/5] Detecting silences...")
    total_duration = get_duration(input_video)
    silences = detect_silences(input_video)
    keep_segments = build_silence_cut_filter(silences, total_duration)

    if keep_segments:
        kept_dur = sum(e - s for s, e in keep_segments)
        cut_dur = total_duration - kept_dur
        print(f"  Original: {total_duration:.1f}s → After cuts: {kept_dur:.1f}s "
              f"(removed {cut_dur:.1f}s)\n")
    else:
        print("  No significant silences to cut\n")

    # Step 5: Final encode
    print("[5/5] Final encoding...")
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

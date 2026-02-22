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
W_SPEAKER = 0.4  # Rush 2: bonus for active speaker (added on top, not normalized)

# Smooth crop
LERP_ALPHA = 0.06          # exponential smoothing factor
LERP_ALPHA_SWITCH = 0.12   # faster alpha when switching subjects
MAX_MOVE_FRAC = 0.05       # max 5% of source width per frame

# Hysteresis
HYSTERESIS_MARGIN = 0.15   # new subject must beat current by this margin
HYSTERESIS_SECONDS = 1.0   # minimum duration before switching subject (used inline as fps * 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Person Detection + Tracking Pass
# ──────────────────────────────────────────────────────────────────────────────
def detect_persons_pass(video_path: str):
    """Run YOLOv8n person detection with ByteTrack tracking."""
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)

    # FIX #6 / HIGH #6 — Validate video opened + has frames
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0:
        cap.release()
        raise RuntimeError(f"Video has 0 frames: {video_path}")

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
            # FIX #6 / HIGH #5 — Better bytetrack.yaml error handling
            try:
                results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                      classes=[0], conf=0.3, verbose=False)
            except Exception as e:
                err_str = str(e).lower()
                if "bytetrack" in err_str or "yaml" in err_str:
                    raise RuntimeError(
                        "bytetrack.yaml introuvable. Réinstaller: pip install ultralytics"
                    ) from e
                raise

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
            # Interpolate: use last detection (copy to avoid shared-reference vote bias)
            all_detections.append(list(last_dets))

        frame_idx += 1

    cap.release()
    return all_detections, fps, total_frames, src_w, src_h


# ──────────────────────────────────────────────────────────────────────────────
# 1b. Diarization: load + map track_id ↔ speaker_label
# ──────────────────────────────────────────────────────────────────────────────
def load_diarization(json_path: str) -> list:
    """Load diarization segments: [{"start": float, "end": float, "speaker": str}, ...]"""
    with open(json_path) as f:
        return json.load(f)


def get_active_speaker(diarization: list, time_s: float) -> str | None:
    """Return the speaker label active at time_s, or None."""
    for seg in diarization:
        if seg["start"] <= time_s <= seg["end"]:
            return seg["speaker"]
    return None


def build_track_speaker_map(all_detections, diarization, fps):
    """Phase 1: Map track_id → speaker_label using single-person frames.

    When only 1 person is detected in a frame AND diarization says someone is speaking,
    that person IS the speaker. Accumulate votes and assign by majority.
    """
    # track_id → {speaker_label: count}
    votes = {}

    for frame_idx, dets in enumerate(all_detections):
        if len(dets) != 1:
            continue  # only use single-person frames for mapping

        t = frame_idx / fps
        speaker = get_active_speaker(diarization, t)
        if not speaker:
            continue

        track_id = dets[0]['track_id']
        if track_id < 0:
            continue

        if track_id not in votes:
            votes[track_id] = {}
        votes[track_id][speaker] = votes[track_id].get(speaker, 0) + 1

    # Assign each track_id to its majority speaker
    track_to_speaker = {}
    for tid, speaker_counts in votes.items():
        best_speaker = max(speaker_counts, key=speaker_counts.get)
        total_votes = sum(speaker_counts.values())
        confidence = speaker_counts[best_speaker] / total_votes
        track_to_speaker[tid] = best_speaker
        print(f"  Track {tid} → {best_speaker} (confidence {confidence:.0%}, {total_votes} votes)")

    # Also build reverse map: speaker → list of track_ids
    speaker_to_tracks = {}
    for tid, spk in track_to_speaker.items():
        speaker_to_tracks.setdefault(spk, []).append(tid)

    return track_to_speaker, speaker_to_tracks


# ──────────────────────────────────────────────────────────────────────────────
# 2. Subject Selection + Smooth Crop
# ──────────────────────────────────────────────────────────────────────────────
def compute_smooth_crop(all_detections, fps, src_w, src_h,
                        diarization=None, track_to_speaker=None):
    """Select subject per frame, compute smoothed crop_x."""
    crop_w = int(src_h * (9 / 16))
    crop_w = min(crop_w, src_w)
    crop_h = src_h

    total_frames = len(all_detections)
    hysteresis_frames = int(fps * HYSTERESIS_SECONDS)  # 1 second
    max_move_px = int(src_w * MAX_MOVE_FRAC)

    current_subject_id = -1
    # CRITICAL FIX #1 — track base score (WITHOUT continuity) for hysteresis comparison.
    # Using base score prevents the continuity bonus from inflating the threshold,
    # which previously made it impossible for a new speaker to ever take over.
    current_base_score = 0.0
    candidate_id = -1
    candidate_frames = 0

    # Start at center
    crop_x = float((src_w - crop_w) / 2)
    crop_positions = []
    active_alpha = LERP_ALPHA

    # Stats
    frames_with_person = 0
    frames_speaker_framed = 0  # Rush 2: count frames where active speaker is the selected subject
    moves = []

    for i in range(total_frames):
        dets = all_detections[i]
        frame_area = src_w * src_h
        t = i / fps  # current time in seconds

        if not dets:
            # FIX #4 — Drift toward center slowly (don't freeze, don't skip LERP)
            center_x = float((src_w - crop_w) / 2)
            drift_alpha = 0.02  # very slow drift, no visible jump
            prev_crop_x = crop_x
            crop_x = crop_x + drift_alpha * (center_x - crop_x)
            crop_x = max(0.0, min(crop_x, float(src_w - crop_w)))
            crop_positions.append(int(round(crop_x)))
            if i > 0:
                move = abs(crop_positions[-1] - crop_positions[-2])
                moves.append(move)
            continue

        frames_with_person += 1

        # Get active speaker for this frame (if diarization available)
        active_speaker = None
        if diarization:
            active_speaker = get_active_speaker(diarization, t)

        # Score each detection
        best_id = -1
        best_score = -1
        best_base_score = -1   # CRITICAL FIX #1: base score WITHOUT continuity
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

            # Rush 2: Speaker bonus
            speaker_score = 0.0
            if active_speaker and track_to_speaker and d['track_id'] in track_to_speaker:
                if track_to_speaker[d['track_id']] == active_speaker:
                    speaker_score = 1.0

            # Base score = WITHOUT continuity (used for fair hysteresis comparison)
            base_score = (size_score * W_SIZE +
                          center_score * W_CENTER +
                          speaker_score * W_SPEAKER)
            total = base_score + continuity_score * W_CONTINUITY

            if total > best_score:
                best_score = total
                best_base_score = base_score
                best_id = d['track_id']
                best_cx = cx

        # CRITICAL FIX #1 — Hysteresis compares BASE scores (without W_CONTINUITY bonus).
        # Previously, current_subject_score was the TOTAL score (with continuity=1.0),
        # which inflated the threshold so a new speaker could never beat it.
        #
        # Two-layer approach:
        # 1. Compare base scores (no continuity) → fair comparison across all subjects.
        # 2. If diarization already signals speaker changed, drop margin to 0 (trust the signal).
        #
        # Recompute current subject's base score freshly each frame (speaker_score may have changed).
        current_det_base_score = 0.0
        for d in dets:
            if d['track_id'] == current_subject_id:
                cx_c = (d['x1'] + d['x2']) / 2
                area_c = (d['x2'] - d['x1']) * (d['y2'] - d['y1'])
                size_c = min(area_c / frame_area * 10, 1.0)
                center_c = 1.0 - abs(cx_c - src_w / 2) / (src_w / 2)
                spk_c = 0.0
                if active_speaker and track_to_speaker and current_subject_id in track_to_speaker:
                    if track_to_speaker[current_subject_id] == active_speaker:
                        spk_c = 1.0
                current_det_base_score = size_c * W_SIZE + center_c * W_CENTER + spk_c * W_SPEAKER
                break

        # Drop hysteresis margin to 0 when diarization already says this subject is NOT the speaker.
        # Speaker change is a strong exogenous signal — we trust it, so no artificial hold-back.
        speaker_switched = (
            active_speaker and track_to_speaker and
            current_subject_id in track_to_speaker and
            track_to_speaker[current_subject_id] != active_speaker
        )
        effective_margin = 0.0 if speaker_switched else HYSTERESIS_MARGIN

        if best_id != current_subject_id:
            # Compare BASE scores (no continuity) so continuity bonus doesn't block speaker switches
            score_beats_current = best_base_score > current_det_base_score + effective_margin

            if score_beats_current:
                if best_id == candidate_id:
                    candidate_frames += 1
                else:
                    candidate_id = best_id
                    candidate_frames = 1

                if candidate_frames >= hysteresis_frames:
                    current_subject_id = best_id
                    current_base_score = best_base_score
                    active_alpha = LERP_ALPHA_SWITCH  # faster transition for subject switch
            else:
                # New subject doesn't beat current by margin → keep current
                candidate_id = -1
                candidate_frames = 0

            # Use current subject's position if not switching yet
            for d in dets:
                if d['track_id'] == current_subject_id:
                    best_cx = (d['x1'] + d['x2']) / 2
                    break
        else:
            candidate_id = -1
            candidate_frames = 0
            current_base_score = best_base_score  # refresh for next frame's comparison
            active_alpha = LERP_ALPHA

        # Rush 2: Track if active speaker is framed
        if active_speaker and track_to_speaker and current_subject_id in track_to_speaker:
            if track_to_speaker[current_subject_id] == active_speaker:
                frames_speaker_framed += 1

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

    # Count frames where diarization says someone is speaking
    frames_with_speech = 0
    if diarization:
        for fi in range(total_frames):
            if get_active_speaker(diarization, fi / fps):
                frames_with_speech += 1

    stats = {
        'frames_with_person': frames_with_person,
        'total_frames': total_frames,
        'pct_person': 100 * frames_with_person / max(total_frames, 1),
        'avg_move': np.mean(moves) if moves else 0,
        'max_jump': max(moves) if moves else 0,
        'frames_speaker_framed': frames_speaker_framed,
        'frames_with_speech': frames_with_speech,
        'pct_speaker_framed': 100 * frames_speaker_framed / max(frames_with_speech, 1) if diarization else None,
    }

    return crop_positions, crop_w, crop_h, stats


# ──────────────────────────────────────────────────────────────────────────────
# 3. Render cropped video
# ──────────────────────────────────────────────────────────────────────────────
def render_cropped(video_path, crop_positions, crop_w, crop_h, fps, tmp_video):
    """Read source, crop per frame, pipe directly to ffmpeg (H.264 lossless).

    FIX #5 — Replaces OpenCV mp4v writer to avoid double lossy encoding.
    Frames are piped as raw BGR24 → ffmpeg encodes to H.264 CRF 0 (lossless).
    """
    cap = cv2.VideoCapture(video_path)

    # FIX #5 — Pipe raw frames directly to ffmpeg (no intermediate lossy encode)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{OUT_W}x{OUT_H}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        tmp_video
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frame_idx = 0
    written = 0

    try:
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
            proc.stdin.write(resized.tobytes())
            written += 1
            frame_idx += 1
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError("FFmpeg pipe encoding failed in render_cropped")

    print(f"  Wrote {written} cropped frames")
    return written


# ──────────────────────────────────────────────────────────────────────────────
# 4. Silence detection & cutting (from reframe_v2)
# ──────────────────────────────────────────────────────────────────────────────
def detect_silences(video_path, total_duration=None):
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

    # FIX MEDIUM #10 — Handle trailing silence (video ends during silence)
    if len(starts) != len(ends):
        print(f"  ⚠️  silence: {len(starts)} starts, {len(ends)} ends — trailing silence detected")
        if len(starts) > len(ends) and total_duration is not None:
            ends.append(str(total_duration))

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
def parse_args():
    """Parse CLI arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Smart Reframe v3: 16:9 → 9:16 with YOLO + ByteTrack")
    parser.add_argument("input", help="Input video (16:9)")
    parser.add_argument("output", help="Output video (9:16)")
    parser.add_argument("--diarization", help="Diarization JSON file (speaker segments)", default=None)
    parser.add_argument("--no-silence-cut", action="store_true", help="Skip silence cutting")
    return parser.parse_args()


def main():
    args = parse_args()
    input_video = args.input
    output_video = args.output
    diarization_path = args.diarization

    print(f"=== Smart Reframe v3 (YOLO + ByteTrack): {input_video} → {output_video} ===")
    if diarization_path:
        print(f"  Speaker diarization: {diarization_path}")
    print()

    # Get source fps for CFR normalization
    cap_info = cv2.VideoCapture(input_video)
    if not cap_info.isOpened():
        print(f"❌ Cannot open video: {input_video}")
        sys.exit(1)
    fps = cap_info.get(cv2.CAP_PROP_FPS)
    cap_info.release()

    # FIX #1 — Wrap all temp file usage in try/finally to guarantee cleanup
    tmp_dir = tempfile.mkdtemp(prefix="reframe_v3_")
    cfr_video = os.path.join(tmp_dir, "cfr_source.mp4")
    tmp_video = os.path.join(tmp_dir, "cropped.mp4")

    try:
        # FIX #2 — Normalize VFR → CFR before any processing
        # This ensures silence timestamps (computed on cfr_video) match the
        # render output (also at constant fps). Prevents A/V desync on YT videos.
        print("[0/4] Normalizing VFR → CFR source...")
        result = subprocess.run([
            "ffmpeg", "-y", "-i", input_video,
            "-vf", f"fps={fps}", "-vsync", "cfr",
            "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
            "-c:a", "copy",
            cfr_video
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ⚠️  CFR normalization failed, using original (VFR desync risk):\n{result.stderr[-300:]}")
            cfr_video = input_video  # fallback
        else:
            print(f"  CFR source: {cfr_video}")

        # Step 1: Person detection + tracking (on CFR video)
        print("[1/4] Running YOLOv8n person detection + ByteTrack tracking...")
        all_detections, fps, total_frames, src_w, src_h = detect_persons_pass(cfr_video)

        # Step 1b: Load diarization + build track↔speaker map (if provided)
        diarization = None
        track_to_speaker = None
        if diarization_path:
            print("\n[1b/4] Loading diarization & mapping tracks → speakers...")
            diarization = load_diarization(diarization_path)
            speakers = set(s["speaker"] for s in diarization)
            print(f"  {len(diarization)} segments, {len(speakers)} speakers: {', '.join(sorted(speakers))}")
            track_to_speaker, speaker_to_tracks = build_track_speaker_map(
                all_detections, diarization, fps
            )
            if not track_to_speaker:
                print("  ⚠️  No track↔speaker mapping found (not enough single-person frames)")
                print("  Falling back to size+center+continuity scoring")

        # Step 2: Subject selection + smooth crop
        print("\n[2/4] Computing subject selection + smooth crop...")
        crop_positions, crop_w, crop_h, stats = compute_smooth_crop(
            all_detections, fps, src_w, src_h,
            diarization=diarization, track_to_speaker=track_to_speaker,
        )
        print(f"  Persons detected in {stats['frames_with_person']}/{stats['total_frames']} frames "
              f"({stats['pct_person']:.1f}%)")
        print(f"  Crop: {crop_w}x{crop_h}")
        print(f"  Avg move/frame: {stats['avg_move']:.1f}px | Max jump: {stats['max_jump']}px")

        # Step 3: Render cropped video (from CFR source, pipe to lossless H.264)
        print("\n[3/4] Rendering cropped video...")
        render_cropped(cfr_video, crop_positions, crop_w, crop_h, fps, tmp_video)

        # Step 4: Silence detection + final encode (on CFR video for timestamp consistency)
        print("\n[4/4] Detecting silences & final encode...")
        total_duration = get_duration(cfr_video)
        keep_segments = None
        if not args.no_silence_cut:
            silences = detect_silences(cfr_video, total_duration=total_duration)
            keep_segments = build_silence_cut_filter(silences, total_duration)
        else:
            print("  Silence cutting disabled")

        if keep_segments:
            kept_dur = sum(e - s for s, e in keep_segments)
            print(f"  {total_duration:.1f}s → {kept_dur:.1f}s (cut {total_duration - kept_dur:.1f}s)")
        else:
            print("  No significant silences")

        print("  Encoding final output...")
        # Audio source = cfr_video (has audio copy from original at same timestamps)
        final_encode(tmp_video, cfr_video, output_video, keep_segments)

    finally:
        # FIX #1 — Guaranteed cleanup even if ffmpeg / YOLO crash
        for f in [tmp_video, cfr_video]:
            if f != input_video and os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
        if os.path.exists(tmp_dir):
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass  # non-empty dir (extra debug files) — don't crash on cleanup

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
    if stats.get('pct_speaker_framed') is not None:
        print(f"Speaker framed: {stats['pct_speaker_framed']:.1f}% of speech frames")

    return stats


if __name__ == "__main__":
    main()

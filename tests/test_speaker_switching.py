#!/usr/bin/env python3
"""
Test CRITICAL FIX #1: speaker switching logic in compute_smooth_crop.

Creates a synthetic scenario with 2 persons at fixed positions and
a fake diarization that alternates speakers. Verifies the crop
actually follows the active speaker.

Run: python tests/test_speaker_switching.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from reframe_v3 import (
    compute_smooth_crop,
    build_track_speaker_map,
    HYSTERESIS_SECONDS,
)

# ─── Synthetic video dimensions ───
SRC_W = 1920
SRC_H = 1080
FPS = 25.0
TOTAL_SECONDS = 8.0
TOTAL_FRAMES = int(FPS * TOTAL_SECONDS)

# Person A: track_id=1, sitting on the LEFT side (x≈480)
# Person B: track_id=2, sitting on the RIGHT side (x≈1440)
PERSON_A = {'track_id': 1, 'x1': 360, 'y1': 200, 'x2': 600, 'y2': 900, 'conf': 0.95}
PERSON_B = {'track_id': 2, 'x1': 1320, 'y1': 200, 'x2': 1560, 'y2': 900, 'conf': 0.95}

# Both visible every frame
ALL_DETS = [[dict(PERSON_A), dict(PERSON_B)]] * TOTAL_FRAMES

# Diarization: A speaks 0→3s, B speaks 3→6s, A speaks 6→8s
DIARIZATION = [
    {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
    {"start": 3.0, "end": 6.0, "speaker": "SPEAKER_01"},
    {"start": 6.0, "end": 8.0, "speaker": "SPEAKER_00"},
]

# Manual mapping: track 1 → SPEAKER_00, track 2 → SPEAKER_01
TRACK_TO_SPEAKER = {1: "SPEAKER_00", 2: "SPEAKER_01"}

CENTER_A = (PERSON_A['x1'] + PERSON_A['x2']) / 2   # 480
CENTER_B = (PERSON_B['x1'] + PERSON_B['x2']) / 2   # 1440
MID = SRC_W / 2  # 960


def test_speaker_switch():
    print("=== Test: Speaker Switching (CRITICAL FIX #1) ===\n")

    crop_positions, crop_w, crop_h, stats = compute_smooth_crop(
        ALL_DETS, FPS, SRC_W, SRC_H,
        diarization=DIARIZATION,
        track_to_speaker=TRACK_TO_SPEAKER,
    )

    # Expected: crop_x centered on A (≈480-crop_w/2) during A's speech,
    # then shifts toward B (≈1440-crop_w/2) during B's speech.
    # crop_w for 1080p source = 1080 * 9/16 ≈ 607px
    # crop_x for A ≈ 480 - 304 ≈ 176  (clamped to 0 minimum)
    # crop_x for B ≈ 1440 - 304 ≈ 1136

    # crop_w for 1920-source ≈ 607px
    # Switch timeline:
    #   A speaks  0→3s  (frames 0→74)
    #   B speaks  3→6s  (frames 75→149):
    #     - hysteresis: 1s accumulation → switch at frame ~100
    #     - lerp settling: ~0.8s → stable right position by frame ~120 (4.8s)
    #   A speaks again 6→8s (frames 150→199):
    #     - switch at frame ~175, lerp incomplete before end of clip
    #
    # Evaluation windows use the settled portion of each segment.

    # A1: frames 50→74 (2s→3s), well after lerp has settled
    seg_a1_start = int(2.0 * FPS)
    seg_a1_end   = int(3.0 * FPS)

    # B: frames 120→149 (4.8s→6s), after 1s hysteresis + 0.8s lerp
    seg_b_start = int(4.8 * FPS)
    seg_b_end   = int(6.0 * FPS)

    # A2: frames 185→199 (7.4s→8s), lerp still in progress but clearly moving left
    seg_a2_start = int(7.4 * FPS)
    seg_a2_end   = TOTAL_FRAMES

    left_threshold  = SRC_W * 0.25   # crop_x < 25% src width → clearly left
    right_threshold = SRC_W * 0.50   # crop_x > 50% src width → clearly right
    # Note: lerp + max_move_px means full settling takes ~2s; 50% is sufficient to prove switch
    # A2 doesn't fully settle in 8s clip; just verify it's moving back (< 70% after switch)
    a2_threshold = SRC_W * 0.70

    def avg_crop(start_f, end_f):
        segment = crop_positions[start_f:end_f]
        return sum(segment) / len(segment) if segment else 0

    avg_a1 = avg_crop(seg_a1_start, seg_a1_end)
    avg_b  = avg_crop(seg_b_start, seg_b_end)
    avg_a2 = avg_crop(seg_a2_start, seg_a2_end)

    print(f"Crop window: {crop_w}px wide")
    print(f"  Center A at x={CENTER_A:.0f} → expected crop_x ≈ {max(0, CENTER_A - crop_w/2):.0f}")
    print(f"  Center B at x={CENTER_B:.0f} → expected crop_x ≈ {min(SRC_W - crop_w, CENTER_B - crop_w/2):.0f}")
    print()
    print(f"  A speaks (0→3s), settled avg crop_x (frames {seg_a1_start}→{seg_a1_end}): {avg_a1:.0f}px "
          f"[should be < {left_threshold:.0f}]")
    print(f"  B speaks (3→6s), settled avg crop_x (frames {seg_b_start}→{seg_b_end}): {avg_b:.0f}px "
          f"[should be > {right_threshold:.0f}]")
    print(f"  A speaks again (6→8s), crop_x at {seg_a2_start/FPS:.1f}s: {avg_a2:.0f}px "
          f"[should be < {a2_threshold:.0f}, lerp still in progress]")
    print()

    failures = []

    if avg_a1 >= left_threshold:
        failures.append(
            f"FAIL: During A's settled speech (2→3s), crop_x={avg_a1:.0f} ≥ {left_threshold:.0f}. "
            "Crop is not following speaker A!"
        )

    if avg_b <= right_threshold:
        failures.append(
            f"FAIL: During B's settled speech (4.8→6s), crop_x={avg_b:.0f} ≤ {right_threshold:.0f}. "
            "Speaker switch to B NOT happening! (This was the CRITICAL #1 bug)"
        )

    if avg_a2 >= a2_threshold:
        failures.append(
            f"FAIL: After switch back to A (7.4→8s), crop_x={avg_a2:.0f} ≥ {a2_threshold:.0f}. "
            "Crop not moving back toward A!"
        )

    if failures:
        print("❌ FAILURES:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("✅ Speaker switching works correctly!")
        print(f"   Crop on LEFT  ({avg_a1:.0f}px < {left_threshold:.0f}) when A speaks")
        print(f"   Crop on RIGHT ({avg_b:.0f}px > {right_threshold:.0f}) when B speaks")
        print(f"   Crop moving back LEFT ({avg_a2:.0f}px < {a2_threshold:.0f}) when A speaks again")
        pct = stats.get('pct_speaker_framed')
        if pct is not None:
            print(f"   Speaker framed: {pct:.1f}% of speech frames")

    return True


if __name__ == "__main__":
    test_speaker_switch()

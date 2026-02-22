#!/usr/bin/env python3
"""subtitles_v2.py — Next-level karaoke subtitles for Podoro Clips (Rush 3).

Features:
- Whisper JSON word-level timestamps input
- Background pill (semi-transparent rounded rect behind text, CapCut style)
- Sharp drop shadow (no blur)
- Scale-up entrance animation (1.0 → 1.05 → 1.0 over 100ms)
- Color highlight pulse on active word (yellow → light yellow → yellow over 200ms)
- Adaptive Y position based on --subject-y
- Auto font sizing if longest word exceeds 90% width
- Max 2 lines, 3-4 words per line
- --style tiktok (bold colors, pill bg) or clean (no pill, subtler)

CLI:
  python subtitles_v2.py --video INPUT.mp4 --whisper WHISPER.json --output OUTPUT.mp4 \
      [--subject-y 0.7] [--style tiktok|clean] [--font-size 72] [--font PATH]
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from PIL import Image, ImageDraw, ImageFont

# ─── Constants ───────────────────────────────────────────────────────────────

DEFAULT_FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FALLBACK_FONTS = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
MAX_WORDS_PER_GROUP = 4
MAX_LINES = 2
SCALE_ANIM_DURATION = 0.100   # seconds
SCALE_ANIM_PEAK = 1.05
PULSE_DURATION = 0.200        # seconds
GROUP_TAIL_BUFFER = 0.35      # keep group visible after last word ends

# Colors
COLOR_WHITE = (255, 255, 255, 255)
COLOR_YELLOW = (255, 255, 0, 255)
COLOR_YELLOW_LIGHT = (255, 255, 160, 255)
COLOR_SHADOW = (0, 0, 0, 220)
COLOR_PILL_BG = (0, 0, 0, 140)

# ─── Data loading ────────────────────────────────────────────────────────────

def load_whisper_words(json_path, offset=0.0):
    with open(json_path) as f:
        data = json.load(f)
    words = []
    # Try top-level "words" first (Whisper API format)
    raw_words = data.get("words", [])
    if not raw_words:
        # Fallback: words nested inside segments (local Whisper format)
        for seg in data.get("segments", []):
            raw_words.extend(seg.get("words", []))
    for w in raw_words:
        words.append({
            "word": w["word"].strip(),
            "start": w["start"] + offset,
            "end": w["end"] + offset,
        })
    return [w for w in words if w["word"]]  # skip empty


def group_words(words, max_words=MAX_WORDS_PER_GROUP):
    """Group into chunks of max_words, splitting on sentence-end punctuation."""
    groups = []
    cur = []
    for w in words:
        cur.append(w)
        ends_sentence = w["word"].rstrip().endswith((".", "!", "?"))
        if len(cur) >= max_words or ends_sentence:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def split_group_lines(group, max_lines=MAX_LINES):
    """Split a word group into up to max_lines lines for display."""
    n = len(group)
    if n <= max_lines:
        return [[w] for w in group]
    per = math.ceil(n / max_lines)
    lines = []
    for i in range(0, n, per):
        lines.append(group[i:i+per])
    return lines[:max_lines]

# ─── Video probe ─────────────────────────────────────────────────────────────

def probe_video(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", path],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(r.stdout)
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    w, h = int(vs["width"]), int(vs["height"])
    num, den = map(int, vs["r_frame_rate"].split("/"))
    fps = num / den
    duration = float(info["format"]["duration"])
    total_frames = int(round(duration * fps))
    return w, h, fps, total_frames, duration

# ─── Font helpers ────────────────────────────────────────────────────────────

def load_font(font_path, size):
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        for fb in FALLBACK_FONTS:
            try:
                return ImageFont.truetype(fb, size)
            except Exception:
                continue
    return ImageFont.load_default()


def auto_font_size(group, font_path, base_size, max_width):
    """Reduce font size if the longest word exceeds 90% of frame width."""
    threshold = int(max_width * 0.90)
    size = base_size
    while size > 24:
        font = load_font(font_path, size)
        longest = max(group, key=lambda w: len(w["word"]))
        bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), longest["word"].upper(), font=font)
        tw = bbox[2] - bbox[0]
        if tw <= threshold:
            break
        size -= 2
    return size

# ─── Rendering ───────────────────────────────────────────────────────────────

def compute_subtitle_y(height, subject_y):
    """Adaptive Y: avoid overlapping the subject."""
    if subject_y is not None:
        if subject_y > 0.55:
            return int(height * 0.22)   # subject low → subs high
        else:
            return int(height * 0.70)   # subject high/mid → subs low
    return int(height * 0.70)           # default


def pulse_color(t_in_word, duration=PULSE_DURATION):
    """Interpolate yellow → yellow-light → yellow over duration."""
    if duration <= 0:
        return COLOR_YELLOW
    phase = (t_in_word % duration) / duration
    # triangle wave 0→1→0
    mix = 1.0 - abs(2.0 * phase - 1.0)
    r = int(COLOR_YELLOW[0] + (COLOR_YELLOW_LIGHT[0] - COLOR_YELLOW[0]) * mix)
    g = int(COLOR_YELLOW[1] + (COLOR_YELLOW_LIGHT[1] - COLOR_YELLOW[1]) * mix)
    b = int(COLOR_YELLOW[2] + (COLOR_YELLOW_LIGHT[2] - COLOR_YELLOW[2]) * mix)
    return (r, g, b, 255)


def scale_factor(t_since_group_start, duration=SCALE_ANIM_DURATION, peak=SCALE_ANIM_PEAK):
    """Entrance scale animation: 1.0 → peak → 1.0."""
    if t_since_group_start < 0 or t_since_group_start > duration:
        return 1.0
    phase = t_since_group_start / duration
    return 1.0 + (peak - 1.0) * math.sin(phase * math.pi)


def render_frame(width, height, group, active_idx, t, font, sub_y, style="tiktok"):
    """Render subtitle overlay for one frame.  Returns RGBA Image."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    lines = split_group_lines(group)
    line_texts = [" ".join(w["word"].upper() for w in line) for line in lines]

    # Measure each line
    line_metrics = []
    for lt in line_texts:
        bbox = draw.textbbox((0, 0), lt, font=font)
        line_metrics.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))

    max_tw = max(m[0] for m in line_metrics)
    total_th = sum(m[1] for m in line_metrics) + 8 * (len(lines) - 1)  # 8px line gap

    # Scale animation
    group_start = group[0]["start"]
    sf = scale_factor(t - group_start)

    # Pill background (tiktok style)
    pad_x, pad_y = 24, 14
    pill_w = int((max_tw + pad_x * 2) * sf)
    pill_h = int((total_th + pad_y * 2) * sf)
    pill_x = (width - pill_w) // 2
    pill_y = sub_y - pad_y

    if style == "tiktok":
        pill = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        pd = ImageDraw.Draw(pill)
        radius = 16
        pd.rounded_rectangle(
            [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
            radius=radius, fill=COLOR_PILL_BG,
        )
        img = Image.alpha_composite(img, pill)
        draw = ImageDraw.Draw(img)

    # Draw text line by line
    cur_y = sub_y
    word_flat_idx = 0  # flat index across all lines
    shadow_offset = 3

    for li, line_words in enumerate(lines):
        line_text = " ".join(w["word"].upper() for w in line_words)
        ltw = line_metrics[li][0]
        lth = line_metrics[li][1]
        line_x = (width - int(ltw * sf)) // 2

        # Render word by word for highlighting
        cx = line_x
        for w in line_words:
            wt = w["word"].upper()
            spacer = " " if w != line_words[-1] else ""
            token = wt + spacer

            # Determine color
            if word_flat_idx == active_idx:
                t_in_word = t - w["start"]
                color = pulse_color(t_in_word)
            else:
                color = COLOR_WHITE

            # Sharp drop shadow
            draw.text((cx + shadow_offset, cur_y + shadow_offset), token, font=font, fill=COLOR_SHADOW)
            # Main text
            draw.text((cx, cur_y), token, font=font, fill=color)

            token_bbox = draw.textbbox((0, 0), token, font=font)
            cx += int((token_bbox[2] - token_bbox[0]) * sf)
            word_flat_idx += 1

        cur_y += lth + 8

    # Apply scale if != 1.0 (resize around center)
    if abs(sf - 1.0) > 0.001:
        new_w = int(width * sf)
        new_h = int(height * sf)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # Crop/paste back to original size
        result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ox = (new_w - width) // 2
        oy = (new_h - height) // 2
        result.paste(img.crop((ox, oy, ox + width, oy + height)), (0, 0))
        img = result

    return img

# ─── Main pipeline ───────────────────────────────────────────────────────────

def find_active(groups, t):
    """Return (group_index, word_index_in_flat_group) for time t, or (None, -1)."""
    for gi, group in enumerate(groups):
        gs = group[0]["start"]
        ge = group[-1]["end"] + GROUP_TAIL_BUFFER
        if gs <= t <= ge:
            # Find active word
            active = -1
            for wi, w in enumerate(group):
                if w["start"] <= t <= w["end"] + 0.15:
                    active = wi
                    break
            if active == -1:
                # fallback: last word whose start <= t
                for wi, w in enumerate(group):
                    if t >= w["start"]:
                        active = wi
            return gi, active
    return None, -1


def process_video(args):
    width, height, fps, total_frames, duration = probe_video(args.video)
    words = load_whisper_words(args.whisper)
    groups = group_words(words)
    font_path = args.font or DEFAULT_FONT
    base_size = args.font_size
    sub_y = compute_subtitle_y(height, args.subject_y)

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames, {duration:.1f}s")
    print(f"Words: {len(words)}, Groups: {len(groups)}, Sub Y: {sub_y}")
    print(f"Style: {args.style}")

    tmpdir = tempfile.mkdtemp(prefix="subsv2_")
    cache = {}
    frame_list = os.path.join(tmpdir, "frames.txt")
    blank_path = os.path.join(tmpdir, "blank.png")
    Image.new("RGBA", (width, height), (0, 0, 0, 0)).save(blank_path)

    fl = open(frame_list, "w")

    for fi in range(total_frames):
        t = fi / fps
        gi, wi = find_active(groups, t)

        if gi is None:
            fl.write(f"file '{blank_path}'\nduration {1/fps}\n")
            continue

        # Quantize time for caching (round to 33ms ≈ 1 frame at 30fps)
        t_q = round(t * 30) / 30
        cache_key = (gi, wi, t_q)

        if cache_key not in cache:
            group = groups[gi]
            fsize = auto_font_size(group, font_path, base_size, width)
            font = load_font(font_path, fsize)
            img = render_frame(width, height, group, wi, t, font, sub_y, args.style)
            path = os.path.join(tmpdir, f"f_{gi}_{wi}_{fi}.png")
            img.save(path)
            cache[cache_key] = path

        fl.write(f"file '{cache[cache_key]}'\nduration {1/fps}\n")

        if fi % 100 == 0:
            print(f"  Frame {fi}/{total_frames} ({100*fi//total_frames}%)", end="\r")

    fl.close()
    print(f"\nRendered {len(cache)} unique subtitle frames")

    # Build subtitle overlay video
    sub_vid = os.path.join(tmpdir, "subs.mov")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", frame_list,
        "-c:v", "png", "-r", str(fps), sub_vid,
    ], check=True, capture_output=True)

    # Overlay on original
    subprocess.run([
        "ffmpeg", "-y",
        "-i", args.video, "-i", sub_vid,
        "-filter_complex", "[0:v][1:v]overlay=0:0:shortest=1[out]",
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-c:a", "copy",
        args.output,
    ], check=True, capture_output=True)

    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"✅ Output: {args.output}")


def main():
    p = argparse.ArgumentParser(description="Podoro Clips — Next-level karaoke subtitles v2")
    p.add_argument("--video", required=True, help="Input video (.mp4)")
    p.add_argument("--whisper", required=True, help="Whisper JSON with word-level timestamps")
    p.add_argument("--output", required=True, help="Output video (.mp4)")
    p.add_argument("--subject-y", type=float, default=None,
                   help="Normalized Y position of subject (0-1). Subs avoid this zone.")
    p.add_argument("--style", choices=["tiktok", "clean"], default="tiktok",
                   help="Visual style: tiktok (pill bg, bold) or clean (minimal)")
    p.add_argument("--font-size", type=int, default=72)
    p.add_argument("--font", type=str, default=None, help="Path to .ttf/.otf font")
    args = p.parse_args()
    process_video(args)


if __name__ == "__main__":
    main()

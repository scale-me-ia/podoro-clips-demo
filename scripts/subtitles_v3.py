#!/usr/bin/env python3
"""subtitles_v3.py — Pro-level karaoke subtitles (Opus Clip / CapCut quality).

Improvements over v2:
- Thick black text stroke (4px) — main readability driver
- Montserrat Black font (bundled)
- No background pill by default (stroke-only style)
- Soft drop shadow behind stroked text
- Per-word highlight: active word gets colored background box
- Position at ~65% to avoid platform UI
- Word-by-word pop-in animation option
- Multiple highlight styles: box, underline, color-only

CLI:
  python subtitles_v3.py --video INPUT.mp4 --whisper WHISPER.json --output OUTPUT.mp4 \
      [--subject-y 0.7] [--style stroke|pill|boxword] [--highlight box|color|underline] \
      [--font-size 68] [--font PATH] [--highlight-color "#FFD700"] [--max-words 3]
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ─── Constants ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "assets", "fonts")
DEFAULT_FONT = os.path.join(ASSETS_DIR, "Montserrat-Black.ttf")
FALLBACK_FONTS = [
    os.path.join(ASSETS_DIR, "Montserrat-ExtraBold.ttf"),
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
]

MAX_WORDS_PER_GROUP = 3       # fewer words = more impact
MAX_LINES = 2
STROKE_WIDTH = 5              # thick black outline
SHADOW_OFFSET = (5, 5)
SHADOW_BLUR = 7
SCALE_ANIM_DURATION = 0.120   # pop-in duration (seconds)
SCALE_ANIM_PEAK = 1.08
GROUP_TAIL_BUFFER = 0.30

# Colors
COLOR_WHITE = (255, 255, 255, 255)
COLOR_BLACK = (0, 0, 0, 255)
COLOR_SHADOW = (0, 0, 0, 160)
COLOR_HIGHLIGHT_BG = (255, 215, 0, 240)      # gold box behind active word
COLOR_HIGHLIGHT_TEXT = (0, 0, 0, 255)          # black text on gold bg
COLOR_PILL_BG = (0, 0, 0, 140)
LINE_GAP = 12

# ─── Data loading ────────────────────────────────────────────────────────────

def load_whisper_words(json_path, offset=0.0):
    with open(json_path) as f:
        data = json.load(f)
    words = []
    raw_words = data.get("words", [])
    if not raw_words:
        for seg in data.get("segments", []):
            raw_words.extend(seg.get("words", []))
    for w in raw_words:
        words.append({
            "word": w["word"].strip(),
            "start": w["start"] + offset,
            "end": w["end"] + offset,
        })
    return [w for w in words if w["word"]]


def group_words(words, max_words=MAX_WORDS_PER_GROUP):
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
    threshold = int(max_width * 0.88)
    size = base_size
    while size > 28:
        font = load_font(font_path, size)
        # Check full line width (not just longest word)
        lines = split_group_lines(group)
        for line in lines:
            text = " ".join(w["word"].upper() for w in line)
            bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox(
                (0, 0), text, font=font, stroke_width=STROKE_WIDTH
            )
            tw = bbox[2] - bbox[0]
            if tw > threshold:
                size -= 2
                break
        else:
            break
    return size

# ─── Text rendering with stroke ──────────────────────────────────────────────

def draw_stroked_text(draw, pos, text, font, fill, stroke_fill=COLOR_BLACK, stroke_width=STROKE_WIDTH):
    """Draw text with a thick stroke outline."""
    x, y = pos
    draw.text((x, y), text, font=font, fill=fill,
              stroke_width=stroke_width, stroke_fill=stroke_fill)


def measure_text(draw, text, font):
    """Measure text with stroke considered."""
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=STROKE_WIDTH)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

# ─── Rendering ───────────────────────────────────────────────────────────────

def compute_subtitle_y(height, subject_y):
    if subject_y is not None:
        if subject_y > 0.55:
            return int(height * 0.18)
        else:
            return int(height * 0.62)
    return int(height * 0.62)


def parse_color(s):
    """Parse hex color string like '#FFD700' to RGBA tuple."""
    s = s.strip().lstrip("#")
    if len(s) == 6:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), 240)
    return COLOR_HIGHLIGHT_BG


def scale_factor(t_since_start, duration=SCALE_ANIM_DURATION, peak=SCALE_ANIM_PEAK):
    if t_since_start < 0 or t_since_start > duration:
        return 1.0
    phase = t_since_start / duration
    return 1.0 + (peak - 1.0) * math.sin(phase * math.pi)


def render_frame(width, height, group, active_idx, t, font, sub_y, style="stroke",
                 highlight="box", highlight_bg=COLOR_HIGHLIGHT_BG, highlight_text=COLOR_HIGHLIGHT_TEXT):
    """Render subtitle overlay. Returns RGBA Image."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    lines = split_group_lines(group)

    # Measure all lines
    line_metrics = []
    for line in lines:
        text = " ".join(w["word"].upper() for w in line)
        tw, th = measure_text(draw, text, font)
        line_metrics.append((tw, th, text))

    total_th = sum(m[1] for m in line_metrics) + LINE_GAP * (len(lines) - 1)

    # Scale animation
    group_start = group[0]["start"]
    sf = scale_factor(t - group_start)

    # Optional pill background (only for "pill" style)
    if style == "pill":
        max_tw = max(m[0] for m in line_metrics)
        pad_x, pad_y = 28, 18
        pill_w = int((max_tw + pad_x * 2) * sf)
        pill_h = int((total_th + pad_y * 2) * sf)
        pill_x = (width - pill_w) // 2
        pill_y = sub_y - pad_y
        pill_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        pd = ImageDraw.Draw(pill_layer)
        pd.rounded_rectangle(
            [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
            radius=20, fill=COLOR_PILL_BG,
        )
        img = Image.alpha_composite(img, pill_layer)
        draw = ImageDraw.Draw(img)

    # Draw shadow layer first (blurred)
    shadow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)

    # Render text line by line
    cur_y = sub_y
    word_flat_idx = 0

    for li, line_words in enumerate(lines):
        ltw, lth, _ = line_metrics[li]
        line_x = (width - int(ltw * sf)) // 2

        # First pass: shadow
        sx = line_x + SHADOW_OFFSET[0]
        sy_shadow = cur_y + SHADOW_OFFSET[1]
        for w in line_words:
            token = w["word"].upper()
            spacer = " " if w != line_words[-1] else ""
            full_token = token + spacer
            shadow_draw.text((sx, sy_shadow), full_token, font=font, fill=COLOR_SHADOW,
                             stroke_width=STROKE_WIDTH, stroke_fill=COLOR_SHADOW)
            tw_tok, _ = measure_text(shadow_draw, full_token, font)
            sx += int(tw_tok * sf)

        cur_y += lth + LINE_GAP

    # Blur shadow
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=SHADOW_BLUR))
    img = Image.alpha_composite(img, shadow_layer)
    draw = ImageDraw.Draw(img)

    # Second pass: actual text with stroke + highlights
    cur_y = sub_y
    word_flat_idx = 0

    for li, line_words in enumerate(lines):
        ltw, lth, _ = line_metrics[li]
        line_x = (width - int(ltw * sf)) // 2

        cx = line_x
        for w in line_words:
            token = w["word"].upper()
            spacer = " " if w != line_words[-1] else ""
            full_token = token + spacer
            token_only = token  # without space for highlight box

            is_active = (word_flat_idx == active_idx)

            # Measure token (without trailing space for box)
            tw_token, th_token = measure_text(draw, token_only, font)
            tw_full, _ = measure_text(draw, full_token, font)

            if is_active and highlight == "box":
                # Draw highlight box behind word
                box_pad_x, box_pad_y = 16, 10
                box_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                box_draw = ImageDraw.Draw(box_layer)
                box_draw.rounded_rectangle(
                    [cx - box_pad_x, cur_y - box_pad_y,
                     cx + tw_token + box_pad_x, cur_y + th_token + box_pad_y],
                    radius=14,
                    fill=highlight_bg,
                )
                img = Image.alpha_composite(img, box_layer)
                draw = ImageDraw.Draw(img)
                # Text on box = dark color
                draw_stroked_text(draw, (cx, cur_y), full_token, font,
                                  fill=highlight_text, stroke_fill=highlight_bg, stroke_width=2)
            elif is_active and highlight == "underline":
                # Normal white text + colored underline
                draw_stroked_text(draw, (cx, cur_y), full_token, font, fill=COLOR_WHITE)
                underline_y = cur_y + th_token + 4
                draw.rounded_rectangle(
                    [cx, underline_y, cx + tw_token, underline_y + 6],
                    radius=3, fill=highlight_bg,
                )
            elif is_active and highlight == "color":
                # Just color the active word
                draw_stroked_text(draw, (cx, cur_y), full_token, font,
                                  fill=highlight_bg[:3] + (255,))
            else:
                # Normal word: white with black stroke
                draw_stroked_text(draw, (cx, cur_y), full_token, font, fill=COLOR_WHITE)

            cx += int(tw_full * sf)
            word_flat_idx += 1

        cur_y += lth + LINE_GAP

    # Apply scale if != 1.0
    if abs(sf - 1.0) > 0.001:
        new_w = int(width * sf)
        new_h = int(height * sf)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ox = (new_w - width) // 2
        oy = (new_h - height) // 2
        result.paste(img.crop((ox, oy, ox + width, oy + height)), (0, 0))
        img = result

    return img

# ─── Main pipeline ───────────────────────────────────────────────────────────

def find_active(groups, t):
    for gi, group in enumerate(groups):
        gs = group[0]["start"]
        ge = group[-1]["end"] + GROUP_TAIL_BUFFER
        if gs <= t <= ge:
            active = -1
            for wi, w in enumerate(group):
                if w["start"] <= t <= w["end"] + 0.15:
                    active = wi
                    break
            if active == -1:
                for wi, w in enumerate(group):
                    if t >= w["start"]:
                        active = wi
            return gi, active
    return None, -1


def process_video(args):
    width, height, fps, total_frames, duration = probe_video(args.video)
    words = load_whisper_words(args.whisper)
    groups = group_words(words, args.max_words)
    font_path = args.font or DEFAULT_FONT
    base_size = args.font_size
    sub_y = compute_subtitle_y(height, args.subject_y)
    highlight_bg = parse_color(args.highlight_color) if args.highlight_color else COLOR_HIGHLIGHT_BG
    highlight_text = COLOR_HIGHLIGHT_TEXT

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames, {duration:.1f}s")
    print(f"Words: {len(words)}, Groups: {len(groups)}, Sub Y: {sub_y}")
    print(f"Style: {args.style}, Highlight: {args.highlight}, Font: {os.path.basename(font_path)}")

    tmpdir = tempfile.mkdtemp(prefix="subsv3_")
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

        t_q = round(t * 30) / 30
        cache_key = (gi, wi, t_q)

        if cache_key not in cache:
            group = groups[gi]
            fsize = auto_font_size(group, font_path, base_size, width)
            font = load_font(font_path, fsize)
            img = render_frame(width, height, group, wi, t, font, sub_y,
                               args.style, args.highlight, highlight_bg, highlight_text)
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
    p = argparse.ArgumentParser(description="Podoro Clips — Pro-level karaoke subtitles v3")
    p.add_argument("--video", required=True)
    p.add_argument("--whisper", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--subject-y", type=float, default=None)
    p.add_argument("--style", choices=["stroke", "pill", "boxword"], default="stroke")
    p.add_argument("--highlight", choices=["box", "color", "underline"], default="box")
    p.add_argument("--highlight-color", type=str, default=None, help="Hex color for highlight, e.g. #FFD700")
    p.add_argument("--font-size", type=int, default=68)
    p.add_argument("--font", type=str, default=None)
    p.add_argument("--max-words", type=int, default=3)
    args = p.parse_args()
    process_video(args)


if __name__ == "__main__":
    main()

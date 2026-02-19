#!/usr/bin/env python3
"""
TikTok/Reels-style subtitle burner.

Burns karaoke-style subtitles onto a video:
  - UPPERCASE, 3-4 words per line, max 2 lines
  - White text with thick black outline glow
  - Current word highlighted in yellow (#FFD700)
  - Centered at 70% of frame height

Usage:
    python subtitles.py --video INPUT.mp4 --vtt SUBS.vtt --output OUTPUT.mp4
    python subtitles.py --video INPUT.mp4 --vtt SUBS.vtt --output OUTPUT.mp4 \
                        --offset -61.0 --crop 1080x1920
"""

import argparse
import html
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Constants ────────────────────────────────────────────────────────────────

FONT_PRIMARY = "/tmp/fonts/TypoldExtended-ExtraBold.otf"
FONT_FALLBACK = "/tmp/fonts/Montserrat-ExtraBold.ttf"

COLOR_DEFAULT = (255, 255, 255)       # white
COLOR_HIGHLIGHT = (255, 215, 0)       # #FFD700 yellow
COLOR_OUTLINE = (0, 0, 0)             # black glow
OUTLINE_WIDTH = 6
FONT_SIZE = 72
MAX_WORDS_PER_LINE = 3
MAX_LINES = 2
Y_POSITION_RATIO = 0.70               # 70% down the frame
LINE_SPACING = 14                      # extra px between lines


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class WordTiming:
    word: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class Cue:
    start: float
    end: float
    words: list[WordTiming] = field(default_factory=list)
    text: str = ""


@dataclass
class DisplayGroup:
    """A group of words that appear on screen together (max 2 lines)."""
    words: list[WordTiming]
    start: float
    end: float


# ── VTT Parsing ──────────────────────────────────────────────────────────────

def parse_timestamp(ts: str) -> float:
    """Parse HH:MM:SS.mmm or MM:SS.mmm to seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


def clean_word(w: str) -> str:
    """Clean HTML entities and tags from a word."""
    w = html.unescape(w)
    w = re.sub(r"<[^>]+>", "", w)
    w = w.strip()
    # Replace YouTube censor pattern [__] with ***
    if re.match(r"^\[?\s*_+\s*\]?$", w):
        w = "***"
    return w


def parse_vtt(path: str, time_offset: float = 0.0) -> list[Cue]:
    """
    Parse YouTube auto-generated VTT with word-level <c> timing tags.
    Deduplicates the repeated cues (YouTube doubles each cue).
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into blocks separated by blank lines
    blocks = re.split(r"\n\n+", content)

    cues: list[Cue] = []

    for block in blocks:
        lines = block.strip().split("\n")
        # Find the timestamp line
        ts_line = None
        text_lines = []
        for line in lines:
            if "-->" in line:
                ts_line = line
            elif ts_line is not None:
                text_lines.append(line)

        if ts_line is None or not text_lines:
            continue

        # Parse timestamps
        ts_match = re.match(
            r"([\d:.]+)\s*-->\s*([\d:.]+)", ts_line
        )
        if not ts_match:
            continue

        start = parse_timestamp(ts_match.group(1)) + time_offset
        end = parse_timestamp(ts_match.group(2)) + time_offset

        # Skip cues completely before 0
        if end <= 0:
            continue

        start = max(0, start)

        # Deduplicate: skip the 10ms "echo" cues (no <c> tags, duration ≤ 0.02s)
        duration = end - start
        has_c_tags = any("<c>" in line or re.search(r"<\d{2}:", line) for line in text_lines)

        if duration <= 0.02 and not has_c_tags:
            continue

        # Parse word timings from <c> tags
        # Format: "FirstWord<TIMESTAMP><c> word2</c><TIMESTAMP><c> word3</c>..."
        # The line may also have a "context" line above (previous cue text repeated)
        # We only care about lines with <c> tags for word timing
        words: list[WordTiming] = []

        for line in text_lines:
            line = line.strip()
            if not line or line == " ":
                continue

            if "<c>" not in line and not re.search(r"<\d{2}:", line):
                # Plain text line (context from previous cue) – skip
                continue

            # Extract word timings
            # Pattern: leading text (first word) then <TIMESTAMP><c> word</c> pairs
            # First word starts at cue start time
            remaining = line

            # Get the first word (before any timestamp tag)
            first_match = re.match(r"^([^<]+?)(<\d{2}:)", remaining)
            if first_match:
                first_word = clean_word(first_match.group(1))
                if first_word:
                    words.append(WordTiming(
                        word=first_word,
                        start=start,
                        end=0  # filled in later
                    ))
                remaining = remaining[first_match.end(1):]

            # Extract all <TIMESTAMP><c> word</c> groups
            pattern = r"<(\d{2}:\d{2}:\d{2}\.\d{3})><c>(.*?)</c>"
            for m in re.finditer(pattern, remaining):
                ts = parse_timestamp(m.group(1)) + time_offset
                word = clean_word(m.group(2))
                if word and ts >= 0:
                    words.append(WordTiming(
                        word=word,
                        start=max(0, ts),
                        end=0
                    ))

        # Fill in end times: each word ends when the next starts
        for i in range(len(words) - 1):
            words[i].end = words[i + 1].start
        if words:
            words[-1].end = end

        # Also clamp first word start
        if words and words[0].end <= words[0].start:
            words[0].end = words[0].start + 0.1

        if words:
            cues.append(Cue(start=start, end=end, words=words,
                            text=" ".join(w.word for w in words)))

    return cues


# ── Group words into display chunks ──────────────────────────────────────────

def group_words_for_display(
    cues: list[Cue],
    font: ImageFont.FreeTypeFont,
    max_line_width: int,
) -> list[DisplayGroup]:
    """
    Flatten all words and regroup into display groups that fit within
    max_line_width pixels, with max MAX_LINES lines per group.
    """
    all_words: list[WordTiming] = []
    for cue in cues:
        all_words.extend(cue.words)

    all_words = [w for w in all_words if w.word.strip()]

    groups: list[DisplayGroup] = []
    i = 0

    while i < len(all_words):
        group_words: list[WordTiming] = []
        lines_used = 0

        while lines_used < MAX_LINES and i < len(all_words):
            # Build one line: add words until width exceeded or MAX_WORDS_PER_LINE
            line_words: list[WordTiming] = []
            for _ in range(MAX_WORDS_PER_LINE):
                if i >= len(all_words):
                    break
                candidate = line_words + [all_words[i]]
                text = " ".join(w.word.upper() for w in candidate)
                bbox = font.getbbox(text)
                width = bbox[2] - bbox[0]
                if line_words and width > max_line_width:
                    break  # this word would overflow, stop this line
                line_words.append(all_words[i])
                i += 1

            group_words.extend(line_words)
            lines_used += 1

        if group_words:
            groups.append(DisplayGroup(
                words=group_words,
                start=group_words[0].start,
                end=group_words[-1].end,
            ))

    return groups


# ── Text rendering ───────────────────────────────────────────────────────────

def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [FONT_PRIMARY, FONT_FALLBACK]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_subtitle_frame(
    frame_bgr: np.ndarray,
    group: DisplayGroup,
    current_time: float,
    font: ImageFont.FreeTypeFont,
) -> np.ndarray:
    """Burn subtitle onto a BGR OpenCV frame using PIL."""
    h, w = frame_bgr.shape[:2]

    # Convert BGR → RGB → PIL
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Split words into lines using width-based breaking
    words = group.words
    max_line_px = int(w * 0.92)
    lines: list[list[WordTiming]] = []
    current_line: list[WordTiming] = []
    for wt in words:
        candidate = current_line + [wt]
        text = " ".join(ww.word.upper() for ww in candidate)
        bbox = font.getbbox(text)
        line_w = bbox[2] - bbox[0]
        if current_line and (line_w > max_line_px or len(candidate) > MAX_WORDS_PER_LINE):
            lines.append(current_line)
            current_line = [wt]
        else:
            current_line.append(wt)
    if current_line:
        lines.append(current_line)

    # Measure lines
    line_texts = [" ".join(wt.word.upper() for wt in line) for line in lines]
    line_bboxes = [font.getbbox(t) for t in line_texts]
    line_heights = [bb[3] - bb[1] for bb in line_bboxes]
    line_widths = [bb[2] - bb[0] for bb in line_bboxes]

    total_height = sum(line_heights) + LINE_SPACING * (len(lines) - 1)
    y_start = int(h * Y_POSITION_RATIO) - total_height // 2

    # Draw each line
    y = y_start
    for line_idx, line_words in enumerate(lines):
        # Center the line
        lw = line_widths[line_idx]
        x_start = (w - lw) // 2

        # Render word by word for karaoke highlight
        x = x_start
        for wi, wt in enumerate(line_words):
            word_text = wt.word.upper()
            # Add space after word (except last in line)
            suffix = " " if wi < len(line_words) - 1 else ""
            display_text = word_text + suffix

            # Determine color: highlight if this word is currently spoken
            if wt.start <= current_time < wt.end:
                color = COLOR_HIGHLIGHT
            else:
                color = COLOR_DEFAULT

            # Draw text with thick black stroke (outline/glow)
            draw.text(
                (x, y), display_text, font=font, fill=color,
                stroke_width=OUTLINE_WIDTH, stroke_fill=COLOR_OUTLINE,
            )

            # Advance x
            word_bbox = font.getbbox(display_text)
            x += word_bbox[2] - word_bbox[0]

        y += line_heights[line_idx] + LINE_SPACING

    # Convert back to BGR numpy
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── Video processing ─────────────────────────────────────────────────────────

def center_crop_frame(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center-crop a frame to target dimensions."""
    h, w = frame.shape[:2]

    # Scale so that the target aspect ratio fits
    target_ratio = target_w / target_h
    src_ratio = w / h

    if src_ratio > target_ratio:
        # Source is wider – crop width
        new_w = int(h * target_ratio)
        x_off = (w - new_w) // 2
        frame = frame[:, x_off : x_off + new_w]
    else:
        # Source is taller – crop height
        new_h = int(w / target_ratio)
        y_off = (h - new_h) // 2
        frame = frame[y_off : y_off + new_h, :]

    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def process_video(
    video_path: str,
    vtt_path: str,
    output_path: str,
    time_offset: float = 0.0,
    crop: tuple[int, int] | None = None,
):
    """Main pipeline: read video, burn subtitles, mux with audio."""
    print(f"Parsing VTT: {vtt_path} (offset={time_offset:+.1f}s)")
    cues = parse_vtt(vtt_path, time_offset=time_offset)
    print(f"  → {len(cues)} cues after dedup")

    # Open video first to know duration
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / fps
    print(f"Video: {src_w}x{src_h} @ {fps:.2f}fps, {total_frames} frames, {video_duration:.1f}s")

    # Filter cues to video duration
    cues = [c for c in cues if c.start < video_duration]
    for c in cues:
        c.end = min(c.end, video_duration)
        for w in c.words:
            w.end = min(w.end, video_duration)
    print(f"  → {len(cues)} cues in video range")

    if crop:
        out_w, out_h = crop
        print(f"  → Cropping to {out_w}x{out_h}")
    else:
        out_w, out_h = src_w, src_h

    # Load font — auto-size so widest realistic line fits in ~90% of width
    font_size = FONT_SIZE
    font = load_font(font_size)
    test_text = "PSYCHOLOGIQUEMENT"  # longest realistic single word
    test_bbox = font.getbbox(test_text)
    test_width = test_bbox[2] - test_bbox[0]
    if test_width > out_w * 0.88:
        font_size = int(font_size * (out_w * 0.85) / test_width)
        font = load_font(font_size)
        print(f"  → Adjusted font size to {font_size}px")
    else:
        print(f"  → Font size: {font_size}px")

    max_line_px = int(out_w * 0.92)
    groups = group_words_for_display(cues, font, max_line_px)
    print(f"  → {len(groups)} display groups")

    if not groups:
        print("ERROR: No subtitle groups found. Check VTT and offset.")
        sys.exit(1)

    # Print first few groups for debugging
    for g in groups[:5]:
        words_str = " ".join(f"{w.word}({w.start:.2f}-{w.end:.2f})" for w in g.words)
        print(f"  [{g.start:.2f}-{g.end:.2f}] {words_str}")

    # Write frames to temp file (no audio)
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (out_w, out_h))

    # Build a quick lookup: for each frame, which group is active?
    frame_idx = 0
    group_idx = 0

    print("Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        # Crop if needed
        if crop:
            frame = center_crop_frame(frame, out_w, out_h)

        # Advance group index
        while group_idx < len(groups) and groups[group_idx].end <= current_time:
            group_idx += 1

        # Check if current group is active
        if group_idx < len(groups):
            g = groups[group_idx]
            if g.start <= current_time < g.end:
                frame = render_subtitle_frame(frame, g, current_time, font)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  {frame_idx}/{total_frames} ({pct:.0f}%)")

    cap.release()
    writer.release()
    print(f"  → Wrote {frame_idx} frames to temp file")

    # Mux with original audio using ffmpeg
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-i", video_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264", "-crf", "22", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    print(f"Muxing audio: {' '.join(ffmpeg_cmd)}")
    subprocess.run(ffmpeg_cmd, check=True)

    os.unlink(tmp_video)
    print(f"Done! Output: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Burn TikTok-style subtitles onto video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--vtt", required=True, help="VTT subtitle file")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Time offset in seconds to apply to VTT timestamps (e.g. -61.0)")
    parser.add_argument("--crop", type=str, default=None,
                        help="Center-crop to WxH (e.g. 1080x1920)")
    args = parser.parse_args()

    crop = None
    if args.crop:
        w, h = args.crop.lower().split("x")
        crop = (int(w), int(h))

    process_video(
        video_path=args.video,
        vtt_path=args.vtt,
        output_path=args.output,
        time_offset=args.offset,
        crop=crop,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate karaoke-style word-by-word subtitles on video using PIL.
Style: TikTok viral - UPPERCASE, 3-4 words per group, white → yellow highlight, glow effect."""

import sys
import json
import os
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def load_word_timestamps(json_path, offset=0.0):
    """Load word timestamps from Whisper JSON, applying time offset."""
    with open(json_path) as f:
        data = json.load(f)
    
    words = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": w["start"] + offset,
                "end": w["end"] + offset
            })
    return words

def group_words(words, max_words=4):
    """Group words into display groups of max N words."""
    groups = []
    current = []
    for w in words:
        current.append(w)
        if len(current) >= max_words:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups

def render_subtitle_frame(width, height, group, active_idx, font, font_small=None):
    """Render a single subtitle frame with karaoke highlighting."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    text = " ".join(w["word"].upper() for w in group)
    
    # Calculate text size
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    # Position: centered, lower third
    x = (width - tw) // 2
    y = int(height * 0.72)
    
    # Draw glow/shadow
    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.text((x, y), text, font=font, fill=(0, 0, 0, 200))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=6))
    img = Image.alpha_composite(img, shadow)
    draw = ImageDraw.Draw(img)
    
    # Draw words one by one
    cur_x = x
    for i, w in enumerate(group):
        word_text = w["word"].upper() + " "
        if i == active_idx:
            color = (255, 255, 0, 255)  # Yellow = active
        else:
            color = (255, 255, 255, 255)  # White = inactive
        
        draw.text((cur_x, y), word_text, font=font, fill=color)
        wbbox = draw.textbbox((0, 0), word_text, font=font)
        cur_x += wbbox[2] - wbbox[0]
    
    return img

def generate_subtitle_frames(words, width, height, fps, total_frames, font_path, font_size=72):
    """Generate all subtitle frames as PNG files."""
    groups = group_words(words)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
        print(f"Warning: Could not load font {font_path}, using default")
    
    tmpdir = tempfile.mkdtemp(prefix="subs_")
    
    # Pre-compute: for each frame, which group and which word is active
    frame_data = []
    for frame_idx in range(total_frames):
        t = frame_idx / fps
        
        # Find active group and word
        active_group = None
        active_word_idx = -1
        
        for gi, group in enumerate(groups):
            group_start = group[0]["start"]
            group_end = group[-1]["end"]
            
            if group_start <= t <= group_end + 0.3:  # small buffer
                active_group = gi
                # Find active word within group
                for wi, w in enumerate(group):
                    if w["start"] <= t <= w["end"] + 0.15:
                        active_word_idx = wi
                        break
                if active_word_idx == -1:
                    # Between words, highlight the one we just passed
                    for wi, w in enumerate(group):
                        if t >= w["start"]:
                            active_word_idx = wi
                break
        
        frame_data.append((active_group, active_word_idx))
    
    # Render unique frames (cache by group+active_word)
    cache = {}
    frame_paths = []
    
    for frame_idx, (gi, wi) in enumerate(frame_data):
        key = (gi, wi)
        
        if gi is None:
            # No subtitle - transparent frame
            path = os.path.join(tmpdir, f"frame_{frame_idx:06d}.png")
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            img.save(path)
            frame_paths.append(path)
            continue
        
        if key not in cache:
            img = render_subtitle_frame(width, height, groups[gi], wi, font)
            cache_path = os.path.join(tmpdir, f"cache_{gi}_{wi}.png")
            img.save(cache_path)
            cache[key] = cache_path
        
        frame_paths.append(cache[key])
    
    return tmpdir, frame_paths, groups

def overlay_subtitles(video_path, output_path, words, font_path, font_size=72):
    """Overlay karaoke subtitles on video."""
    # Get video info
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
        capture_output=True, text=True
    )
    streams = json.loads(probe.stdout)["streams"]
    vstream = next(s for s in streams if s["codec_type"] == "video")
    width = int(vstream["width"])
    height = int(vstream["height"])
    fps_parts = vstream["r_frame_rate"].split("/")
    fps = int(fps_parts[0]) / int(fps_parts[1])
    total_frames = int(float(vstream.get("nb_frames", 0)) or float(vstream.get("duration", 0)) * fps)
    
    print(f"Video: {width}x{height} @ {fps}fps, ~{total_frames} frames")
    print(f"Words: {len(words)}")
    
    # Generate subtitle frames
    tmpdir, frame_paths, groups = generate_subtitle_frames(
        words, width, height, fps, total_frames, font_path, font_size
    )
    
    print(f"Generated subtitle frame cache in {tmpdir}")
    print(f"Groups: {len(groups)}")
    
    # Create a concat file for the overlay frames
    # Actually, easier to use PIL to composite directly
    # Let's use ffmpeg overlay with image sequence
    
    # Write frame list
    listfile = os.path.join(tmpdir, "frames.txt")
    with open(listfile, "w") as f:
        for path in frame_paths:
            f.write(f"file '{path}'\n")
            f.write(f"duration {1/fps}\n")
    
    # Create subtitle video from frames
    sub_video = os.path.join(tmpdir, "subs.mov")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", listfile,
        "-c:v", "png", "-r", str(fps),
        sub_video
    ]
    print(f"Creating subtitle overlay video...")
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Overlay on original video
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", sub_video,
        "-filter_complex", "[0:v][1:v]overlay=0:0:shortest=1[out]",
        "-map", "[out]", "-map", "0:a",
        "-c:v", "libx264", "-crf", "23",
        "-c:a", "copy",
        output_path
    ]
    print(f"Overlaying subtitles...")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Output: {output_path}")
    
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <video.mp4> <whisper.json> <output.mp4> <font.otf> [offset_seconds] [font_size]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    json_path = sys.argv[2]
    output_path = sys.argv[3]
    font_path = sys.argv[4]
    offset = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    font_size = int(sys.argv[6]) if len(sys.argv) > 6 else 72
    
    words = load_word_timestamps(json_path, offset=offset)
    overlay_subtitles(video_path, output_path, words, font_path, font_size)

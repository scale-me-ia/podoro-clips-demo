#!/bin/bash
# Podoro Clips — Full E2E Pipeline Test
# Source 16:9 → Whisper → Reframe v3 (YOLO) → Subtitles v3 (karaoke pro)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

INPUT="${1:-sp_rank1_16x9.mp4}"
OUT_DIR="demo/e2e_test"
mkdir -p "$OUT_DIR"

echo "================================================"
echo "  Podoro Clips — E2E Pipeline Test"
echo "  Input: $INPUT"
echo "  Output: $OUT_DIR/"
echo "================================================"

# Step 1: Whisper transcription (word-level timestamps)
echo ""
echo "▶ Step 1/3: Whisper transcription..."
WHISPER_JSON="$OUT_DIR/transcription.json"
if [ -f "$WHISPER_JSON" ]; then
    echo "  [whisper] Already exists, skipping"
else
    whisper "$INPUT" --model small --language fr --output_format json --word_timestamps True --output_dir "$OUT_DIR/"
    # Whisper names the output after the input file
    BASENAME=$(basename "$INPUT" .mp4)
    mv "$OUT_DIR/${BASENAME}.json" "$WHISPER_JSON" 2>/dev/null || true
fi
echo "  [whisper] ✅ $WHISPER_JSON"

# Step 2: Reframe v3 (YOLO + ByteTrack → 9:16)
echo ""
echo "▶ Step 2/3: Reframe v3 (YOLO + ByteTrack)..."
REFRAMED="$OUT_DIR/reframed_9x16.mp4"
if [ -f "$REFRAMED" ]; then
    echo "  [reframe] Already exists, skipping"
else
    python3 scripts/reframe_v3.py "$INPUT" "$REFRAMED"
fi
echo "  [reframe] ✅ $REFRAMED"

# Step 3: Subtitles v3 (pro karaoke — Montserrat Black, stroke, box highlight)
echo ""
echo "▶ Step 3/3: Subtitles v3 (pro karaoke)..."
FINAL="$OUT_DIR/final_clip.mp4"
python3 scripts/subtitles_v3.py \
    --video "$REFRAMED" \
    --whisper "$WHISPER_JSON" \
    --output "$FINAL" \
    --style stroke \
    --highlight box \
    --max-words 3 \
    --font-size 68

echo ""
echo "================================================"
echo "  ✅ Pipeline complete!"
echo "  📹 $FINAL"
DUR=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$FINAL")
SIZE=$(du -h "$FINAL" | cut -f1)
echo "  ⏱  ${DUR}s — ${SIZE}"
echo "================================================"

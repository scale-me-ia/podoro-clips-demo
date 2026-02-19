#!/usr/bin/env python3
"""
3-Phase Viral Passage Scorer for French Podcast "Sans Permission"
Finds the best 20-30 second clips for TikTok/Reels
"""

import os
import re
import json
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import anthropic
from pydub import AudioSegment
import librosa
import numpy as np


@dataclass
class Passage:
    """Represents a 30-second passage with metadata"""
    start_time: float  # seconds
    end_time: float    # seconds
    text: str
    word_count: int
    phase1_score: float = 0.0
    phase2_score: float = 0.0
    final_score: float = 0.0
    hook_text: str = ""
    best_quote: str = ""
    explanation: str = ""

    def to_dict(self):
        return asdict(self)


def parse_timestamp(ts: str) -> float:
    """Convert VTT timestamp to seconds: '00:01:23.456' -> 83.456"""
    match = re.match(r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})', ts)
    if not match:
        return 0.0
    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000


def parse_vtt(vtt_path: str) -> List[Tuple[float, float, str]]:
    """Parse VTT file and return list of (start, end, text) tuples"""
    entries = []

    with open(vtt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for timestamp line
        if '-->' in line:
            parts = line.split('-->')
            if len(parts) == 2:
                start = parse_timestamp(parts[0].strip().split()[0])
                end = parse_timestamp(parts[1].strip().split()[0])

                # Next line(s) contain the text
                text_lines = []
                i += 1
                while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                    # Clean up VTT markup tags like <00:00:00.480><c>
                    clean = re.sub(r'<[^>]+>', ' ', lines[i])
                    clean = clean.strip()
                    if clean:
                        text_lines.append(clean)
                    i += 1

                text = ' '.join(text_lines).strip()
                if text:
                    entries.append((start, end, text))

        i += 1

    return entries


def build_passages(entries: List[Tuple[float, float, str]],
                   target_duration: float = 30.0,
                   min_duration: float = 20.0,
                   max_duration: float = 45.0,
                   overlap: float = 10.0) -> List[Passage]:
    """
    Build passages with flexible duration (20-45s) that try to end at sentence boundaries.
    This helps find complete narrative arcs instead of cutting mid-story.
    """
    passages = []
    start = 0.0

    # Find max end time
    max_time = max(end for _, end, _ in entries)

    # Sentence-ending markers in French
    sentence_enders = {'.', '!', '?', '…'}

    while start < max_time:
        # Collect entries in the max window
        target_end = start + target_duration
        max_end = start + max_duration
        min_end = start + min_duration

        window_entries = [(s, e, t) for s, e, t in entries if s < max_end and e > start]

        if not window_entries:
            start += (target_duration - overlap)
            continue

        # Find the best end point: a sentence boundary near target_duration
        best_end = target_end
        best_distance = float('inf')

        for entry_start, entry_end, text in window_entries:
            # Check if this entry ends a sentence
            stripped = text.rstrip()
            if stripped and stripped[-1] in sentence_enders and entry_end >= min_end:
                distance = abs(entry_end - target_end)
                if distance < best_distance:
                    best_distance = distance
                    best_end = entry_end

        # Collect text up to best_end
        texts = []
        for entry_start, entry_end, text in entries:
            if entry_start < best_end and entry_end > start:
                texts.append(text)

        if texts:
            full_text = ' '.join(texts)
            word_count = len(full_text.split())

            passages.append(Passage(
                start_time=start,
                end_time=best_end,
                text=full_text,
                word_count=word_count
            ))

        start += (target_duration - overlap)

    return passages


def phase1_text_filter(passages: List[Passage], api_key: str) -> List[Passage]:
    """
    Phase 1: Quick text pre-filter using Claude API
    Score passages 1-10 based on hook strength + emotional potential
    Keep only passages scoring >= 6
    """
    print(f"\n=== PHASE 1: Text Pre-filter ===")
    print(f"Processing {len(passages)} passages in batches of 25...")

    client = anthropic.Anthropic(api_key=api_key)
    filtered = []
    batch_size = 25

    for batch_idx in range(0, len(passages), batch_size):
        batch = passages[batch_idx:batch_idx + batch_size]

        # Build prompt with numbered passages
        passages_text = ""
        for i, p in enumerate(batch):
            passages_text += f"\n#{i+1} [{p.start_time:.1f}s-{p.end_time:.1f}s]:\n{p.text}\n"

        prompt = f"""Tu es un expert en contenu viral pour TikTok/Reels. Analyse ces passages d'un podcast français "Sans Permission" (débats, opinions fortes, sujets controversés).

Pour chaque passage, donne un score de 1 à 10 basé sur:
- Force du HOOK (premières 3 secondes) - CRITIQUE
- Potentiel émotionnel (controverse, surprise, tension)
- Phrases quotables ("mic drop moment")

Réponds UNIQUEMENT avec un JSON array de scores:
[{{"id": 1, "score": X}}, {{"id": 2, "score": Y}}, ...]

Passages:{passages_text}"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse JSON response
            response_text = response.content[0].text.strip()
            # Extract JSON if wrapped in markdown code block
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()

            scores = json.loads(response_text)

            # Apply scores to passages
            for score_obj in scores:
                passage_id = score_obj['id'] - 1  # Convert to 0-indexed
                score = score_obj['score']

                if passage_id < len(batch):
                    batch[passage_id].phase1_score = score
                    if score >= 7:
                        filtered.append(batch[passage_id])

            print(f"  Batch {batch_idx//batch_size + 1}: {len([s for s in scores if s['score'] >= 6])}/{len(scores)} passed filter")

        except Exception as e:
            print(f"  ERROR in batch {batch_idx//batch_size + 1}: {e}")
            # On error, keep all passages from this batch with default score
            for p in batch:
                p.phase1_score = 5.0

    print(f"\nPhase 1 complete: {len(filtered)}/{len(passages)} passages passed (score >= 6)")
    return filtered


_audio_cache = {}

def extract_audio_segment(audio_path: str, start: float, end: float) -> AudioSegment:
    """Extract audio segment from full audio file (cached load)"""
    if audio_path not in _audio_cache:
        print(f"  Loading full audio (one-time)...")
        _audio_cache[audio_path] = AudioSegment.from_file(audio_path)
    audio = _audio_cache[audio_path]
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    return audio[start_ms:end_ms]


def analyze_audio_energy(audio_segment: AudioSegment, passage: Passage) -> Dict[str, float]:
    """
    Phase 2: Analyze audio energy metrics
    Returns normalized metrics (0-1 scale)
    """
    # Convert to numpy array
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    sample_rate = audio_segment.frame_rate

    # Normalize samples
    samples = samples / (2**15)  # 16-bit audio

    # 1. RMS Energy (average and peak)
    rms = np.sqrt(np.mean(samples**2))
    peak_rms = np.max(np.abs(samples))

    # 2. Energy variance (emotional shifts)
    # Split into 1-second windows
    window_size = sample_rate
    n_windows = len(samples) // window_size
    window_energies = []
    for i in range(n_windows):
        window = samples[i*window_size:(i+1)*window_size]
        window_energies.append(np.sqrt(np.mean(window**2)))

    energy_variance = np.var(window_energies) if len(window_energies) > 1 else 0.0

    # 3. Speaking rate (words per second from VTT)
    duration = passage.end_time - passage.start_time
    speaking_rate = passage.word_count / duration if duration > 0 else 0.0

    # 4. Silence ratio
    # Detect silence (threshold at 5% of max amplitude)
    silence_threshold = 0.05 * peak_rms if peak_rms > 0 else 0.01
    silence_mask = np.abs(samples) < silence_threshold
    silence_ratio = np.mean(silence_mask)

    return {
        'rms_avg': float(rms),
        'rms_peak': float(peak_rms),
        'energy_variance': float(energy_variance),
        'speaking_rate': float(speaking_rate),
        'silence_ratio': float(silence_ratio)
    }


def normalize_audio_metrics(passages_with_metrics: List[Tuple[Passage, Dict]]) -> List[Passage]:
    """Normalize audio metrics to 0-1 scale and compute Phase 2 score"""

    # Extract all metrics for normalization
    all_rms = [m['rms_avg'] for _, m in passages_with_metrics]
    all_variance = [m['energy_variance'] for _, m in passages_with_metrics]
    all_rate = [m['speaking_rate'] for _, m in passages_with_metrics]
    all_silence = [m['silence_ratio'] for _, m in passages_with_metrics]

    # Compute min/max for normalization
    def normalize(value, min_val, max_val):
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)

    rms_min, rms_max = min(all_rms), max(all_rms)
    var_min, var_max = min(all_variance), max(all_variance)
    rate_min, rate_max = min(all_rate), max(all_rate)

    scored_passages = []

    for passage, metrics in passages_with_metrics:
        # Normalize each metric
        rms_norm = normalize(metrics['rms_avg'], rms_min, rms_max)
        variance_norm = normalize(metrics['energy_variance'], var_min, var_max)
        rate_norm = normalize(metrics['speaking_rate'], rate_min, rate_max)

        # Silence scoring: some silence is good (for punchlines), too much is bad
        silence = metrics['silence_ratio']
        if silence < 0.15:  # Too little silence = constant talking
            silence_score = 0.7
        elif silence < 0.35:  # Sweet spot
            silence_score = 1.0
        else:  # Too much silence
            silence_score = max(0, 1.0 - (silence - 0.35) / 0.3)

        # Weighted Phase 2 score
        phase2_score = (
            rms_norm * 0.25 +           # Energy level
            variance_norm * 0.35 +      # Emotional shifts (high weight)
            rate_norm * 0.25 +          # Speaking rate
            silence_score * 0.15        # Silence quality
        ) * 10  # Scale to 0-10

        passage.phase2_score = phase2_score
        scored_passages.append(passage)

    return scored_passages


def phase2_audio_analysis(passages: List[Passage], audio_path: str) -> List[Passage]:
    """
    Phase 2: Audio energy analysis for all passages
    """
    print(f"\n=== PHASE 2: Audio Energy Analysis ===")
    print(f"Analyzing {len(passages)} passages...")

    passages_with_metrics = []

    for i, passage in enumerate(passages):
        try:
            # Extract audio segment
            audio_segment = extract_audio_segment(audio_path, passage.start_time, passage.end_time)

            # Analyze audio
            metrics = analyze_audio_energy(audio_segment, passage)
            passages_with_metrics.append((passage, metrics))

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(passages)} passages...")

        except Exception as e:
            print(f"  ERROR analyzing passage at {passage.start_time}s: {e}")
            # Default metrics on error
            passages_with_metrics.append((passage, {
                'rms_avg': 0.1,
                'rms_peak': 0.2,
                'energy_variance': 0.01,
                'speaking_rate': 3.0,
                'silence_ratio': 0.2
            }))

    # Normalize and score
    scored = normalize_audio_metrics(passages_with_metrics)

    print(f"\nPhase 2 complete: {len(scored)} passages scored")
    return scored


def phase3_final_scoring(passages: List[Passage], api_key: str, top_n: int = 20) -> List[Passage]:
    """
    Phase 3: Detailed final scoring with Claude API
    Take top 15-20 passages from Phase 1+2, get detailed scores
    """
    print(f"\n=== PHASE 3: Final Detailed Scoring ===")

    # Sort by combined Phase 1 + 2 scores
    passages.sort(key=lambda p: p.phase1_score + p.phase2_score, reverse=True)
    top_passages = passages[:top_n]

    print(f"Scoring top {len(top_passages)} passages with detailed Claude analysis...")

    client = anthropic.Anthropic(api_key=api_key)

    for i, passage in enumerate(top_passages):
        # Extract first ~100 chars as hook
        hook_text = passage.text[:100] + "..." if len(passage.text) > 100 else passage.text

        prompt = f"""Tu es un expert en contenu viral TikTok/Reels. Score ce passage d'un podcast français "Sans Permission" (débats, opinions tranchées).

PASSAGE [{passage.start_time:.1f}s-{passage.end_time:.1f}s]:
{passage.text}

AUDIO METRICS:
- Énergie audio: {passage.phase2_score:.1f}/10
- Mots: {passage.word_count}

CRITÈRES DE SCORING (total 100 points):

1. HOOK (0-30 points) - MUST start strong
   - Question, provocation, déclaration choc dès le début?
   - Les 3 premières secondes captent-elles l'attention?
   - Réaction forte d'un autre host ("QUOI?!", rire, indignation)?
   - Changement brusque de volume/ton = gold
   - Score 0 si début mou ou mid-sentence

2. ARC NARRATIF COMPLET (0-25 points) - CRITIQUE
   - Le passage raconte-t-il une HISTOIRE COMPLÈTE?
   - Structure: Setup → Développement → Conclusion/Punchline
   - Score 0 si ça coupe en plein milieu d'une anecdote
   - Score 0 si pas de vraie fin (le viewer doit avoir un sentiment de complétude)
   - Idéal: l'anecdote se conclut par une leçon, une chute, ou une réaction forte

3. CONTROVERSE/ÉMOTION (0-25 points)
   - Opinion tranchée, tension, surprise?
   - Potentiel de débat dans les commentaires?
   - Réaction émotionnelle forte (contraste calme → explosion)?
   - Échanges rapides entre hosts = tension = viral

4. PUNCHLINE (0-10 points)
   - Phrase quotable, "mic drop moment"?
   - Moment mémorable à partager?

5. VALEUR ACTIONABLE (0-10 points)
   - Conseil concret, leçon de vie?
   - Insight utile?

BONUS:
- +10: Commence par question/interpellation directe
- +10: Shift émotionnel dans le passage (calme → intense ou inverse)
- +5: Réaction audible d'un autre host (rire, surprise)

MALUS:
- -20: Pas de conclusion — l'histoire est coupée, pas finie
- -15: Commence mid-sentence / manque de contexte
- -10: Trop niche/technique (pas universel)

IMPORTANT: Le passage DOIT raconter une histoire complète. Hook + anecdote + conclusion. Si ça coupe avant la fin de l'histoire, c'est éliminatoire.

Réponds en JSON:
{{
  "score": <0-100>,
  "hook_score": <0-30>,
  "arc_score": <0-25>,
  "emotion_score": <0-25>,
  "punchline_score": <0-10>,
  "value_score": <0-10>,
  "has_conclusion": true/false,
  "hook_text": "<premiers 50 chars du passage>",
  "best_quote": "<phrase la plus quotable>",
  "explanation": "<2-3 phrases expliquant le score>",
  "suggested_trim": "<si le passage est bon mais trop long/court, suggérer un meilleur start/end en secondes>"
}}"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()
            # Extract JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()

            result = json.loads(response_text)

            passage.final_score = result['score']
            passage.hook_text = result.get('hook_text', hook_text)
            passage.best_quote = result.get('best_quote', '')
            passage.explanation = result.get('explanation', '')

            print(f"  Passage {i+1}/{len(top_passages)}: {passage.final_score:.0f} points")

        except Exception as e:
            print(f"  ERROR scoring passage {i+1}: {e}")
            passage.final_score = (passage.phase1_score + passage.phase2_score) * 5  # Fallback
            passage.hook_text = hook_text
            passage.explanation = "Auto-scored (API error)"

    # Sort by final score
    top_passages.sort(key=lambda p: p.final_score, reverse=True)

    print(f"\nPhase 3 complete: Top 5 scores: {[p.final_score for p in top_passages[:5]]}")
    return top_passages


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def print_summary(top_passages: List[Passage]):
    """Print nice summary to stdout"""
    print("\n" + "="*80)
    print("🎯 TOP 5 VIRAL PASSAGES FOR TIKTOK/REELS")
    print("="*80)

    for i, p in enumerate(top_passages[:5], 1):
        print(f"\n#{i} - SCORE: {p.final_score:.0f}/100")
        print(f"⏱️  Time: {format_time(p.start_time)} - {format_time(p.end_time)}")
        print(f"🎣 Hook: {p.hook_text}")
        print(f"💬 Best Quote: {p.best_quote}")
        print(f"📊 Phase Scores: Text={p.phase1_score:.1f}, Audio={p.phase2_score:.1f}")
        print(f"📝 Why: {p.explanation}")
        print("-" * 80)


def main():
    # Configuration
    VTT_PATH = "/tmp/sp-clips/sanspermission.fr.vtt"
    AUDIO_PATH = "/tmp/sp-clips/sanspermission_full.mp4"
    OUTPUT_DIR = "/tmp/podoro-clips-v3/agent-scoring"

    # Get API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    print("🎙️  3-PHASE VIRAL PASSAGE SCORER")
    print(f"📄 VTT: {VTT_PATH}")
    print(f"🎵 Audio: {AUDIO_PATH}")

    # Parse VTT
    print("\n📖 Parsing VTT file...")
    entries = parse_vtt(VTT_PATH)
    print(f"Found {len(entries)} VTT entries")

    # Build passages
    print("\n🔨 Building 30s passages with 10s overlap...")
    passages = build_passages(entries, target_duration=30.0, overlap=10.0)
    print(f"Created {len(passages)} passages")

    # Phase 1: Text pre-filter
    filtered_passages = phase1_text_filter(passages, api_key)

    # Phase 2: Audio analysis
    scored_passages = phase2_audio_analysis(filtered_passages, AUDIO_PATH)

    # Phase 3: Final scoring (top 20)
    final_passages = phase3_final_scoring(scored_passages, api_key, top_n=20)

    # Save results
    print(f"\n💾 Saving results to {OUTPUT_DIR}...")

    # Top 5
    top5_data = [p.to_dict() for p in final_passages[:5]]
    with open(f"{OUTPUT_DIR}/scores_final.json", 'w', encoding='utf-8') as f:
        json.dump(top5_data, f, indent=2, ensure_ascii=False)

    # All scored
    all_data = [p.to_dict() for p in final_passages]
    with open(f"{OUTPUT_DIR}/scores_all.json", 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(top5_data)} top passages to scores_final.json")
    print(f"✅ Saved {len(all_data)} all passages to scores_all.json")

    # Print summary
    print_summary(final_passages)

    print("\n✨ Done!")


if __name__ == '__main__':
    main()

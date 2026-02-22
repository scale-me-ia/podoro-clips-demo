# 🔥 Code Review Adversaire — Rush 2 (Speaker-Aware Reframing)
**Date** : 2026-02-22  
**Reviewer** : Agent adversaire  
**Fichiers** : `scripts/reframe_v3.py`, `scripts/whisper_transcribe.py`, `scripts/diarize.py`  
**Verdict** : ⚠️ CHANGES REQUESTED

---

## Verdict Global

Le Rush 2 **ne livre pas son objectif principal** en l'état. Le speaker-aware reframing est fonctionnellement cassé pour les scénarios à deux speakers alternants (le cas d'usage cœur d'un podcast). Les deux autres fichiers ont des défauts de robustesse non négligeables.

---

## 🔴 ISSUE #1 — W_SPEAKER = 0.4 casse l'hysteresis et bloque le suivi speaker (CRITIQUE)

**Fichier** : `scripts/reframe_v3.py`, lignes config + `compute_smooth_crop`

### Problème

Le ROADMAP spec (T2.1) définit clairement :
```
total = size * 0.4 + center * 0.2 + continuity * 0.3 + speaker * 0.1
```
Somme des poids = **1.0**, normalisée.

Le code livré utilise :
```python
W_SIZE = 0.4
W_CENTER = 0.2
W_CONTINUITY = 0.3
W_SPEAKER = 0.4  # Rush 2: bonus for active speaker (added on top, not normalized)
```
Somme effective = **1.3**. Non normalisé, assumé intentionnel d'après le commentaire.

### Pourquoi c'est cassé

**Scénario** : podcast à 2 speakers (A et B), tous deux présents à l'écran. A parle puis B prend la parole.

- Speaker A actif : score ≈ `0.4 + 0.2 + 0.3 + 0.4 = 1.3` (sujet actuel)
- Speaker B prend la parole : score ≈ `0.4 + 0.2 + 0.0 + 0.4 = 1.0` (pas encore sujet courant, pas de continuity)

Condition de switch : `best_score > current_subject_score + HYSTERESIS_MARGIN`  
→ `1.0 > 1.3 + 0.15 = 1.45` → **FALSE**

**Le système ne switche JAMAIS sur B** tant que A reste visible, car le bonus speaker de B (0.4) ne compense pas la perte de continuity (0.3) + le HYSTERESIS_MARGIN (0.15). Le cadreur reste collé au premier speaker de la vidéo même quand l'autre parle. La feature "speaker-aware" est inopérante dans son cas d'usage principal.

Le problème s'inverse si B était le sujet courant : même blocage. Le commentaire "added on top" ne suffit pas — sans renormalisation ou ajustement de l'hysteresis, les deux systèmes (score et hysteresis) sont désormais incohérents.

### Fix requis

**Option A** (conforme ROADMAP) : revenir à `W_SPEAKER = 0.1`, poids normalisés à 1.0.

**Option B** (dérogation assumée) : si on veut vraiment 0.4, ajuster `HYSTERESIS_MARGIN` à 0.5+ ET s'assurer que le score du speaker challanger peut effectivement battre le courant. Mais cette option est fragile.

**Option C** (propre) : remplacer le bonus additionnel par un multiplicateur ou une override conditionnelle : si le speaker actif est B, passer `best_cx` à B directement et réduire l'alpha de transition.

---

## 🔴 ISSUE #2 — Whisper API : zéro gestion d'erreurs, zéro retry, pas de vérification de taille fichier

**Fichier** : `scripts/whisper_transcribe.py`, fonction `transcribe_whisper`

### Problème

```python
response = client.audio.transcriptions.create(
    model="whisper-1",
    file=f,
    language=language,
    ...
)
```

Aucun `try/except`. Si l'API répond :
- `429 RateLimitError` → exception non catchée, crash pipeline
- `500 / 503` OpenAI outage → crash
- Timeout réseau → crash
- Fichier > 25MB (limite Whisper API) → erreur API non catchée

Le ROADMAP note explicitement au Rush 4 T4.3 : *"Retry automatique sur les appels API (Whisper, Claude, Gemini) — 3 tentatives avec backoff"*. Mais `whisper_transcribe.py` est déjà utilisé en Rush 2 dans la pipeline et n'a aucune résilience.

De plus, `extract_audio` utilise `subprocess.run(..., check=True)` sans `capture_output` sur `stderr` vers l'utilisateur — si ffmpeg échoue, on a juste `CalledProcessError` sans contexte.

### Fix requis

```python
import time

def transcribe_whisper(audio_path: str, language: str = "fr", max_retries: int = 3) -> dict:
    # Vérification taille avant upload
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if size_mb > 24:
        raise ValueError(f"Audio file too large for Whisper API: {size_mb:.1f}MB (max 25MB)")
    
    client = OpenAI()
    for attempt in range(max_retries):
        try:
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(...)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  ⚠️ Whisper API error (attempt {attempt+1}/{max_retries}), retry in {wait}s: {e}")
            time.sleep(wait)
```

---

## 🟠 ISSUE #3 — Phase 1 mapping : track_id = -1 silencieux pollue les stats

**Fichier** : `scripts/reframe_v3.py`, fonction `build_track_speaker_map`

### Problème

Quand ByteTrack n'a pas encore assigné d'ID à une détection (track_id = -1), le code skippe correctement avec `if track_id < 0: continue`. C'est bien.

Mais dans `detect_persons_pass`, les frames non-sampled reçoivent `last_dets` par référence directe :

```python
all_detections.append(last_dets)  # même objet list !
```

Si `last_dets` est muté entre deux appels (ce qui n'arrive pas ici car il est réassigné), ça polluerait silencieusement tous les frames qui l'ont reçu. C'est un **bug latent** : si quelqu'un ajoute du post-processing sur les dets in-place, tous les frames partagés seront corrompus.

Plus sérieusement : quand `SAMPLE_EVERY = 2`, la moitié des frames partagent le même objet list. Dans `build_track_speaker_map`, le vote de frame `i` ET frame `i+1` pèsent pour la même détection → les votes sont **doublés artificiellement** pour les frames non-sampled, biaisant les counts sans que ça soit visible.

### Fix requis

```python
all_detections.append(list(last_dets))  # copie, pas référence
```

Et dans `compute_smooth_crop`, même problème pour les frames interpolés.

---

## 🟠 ISSUE #4 — `tempfile.mktemp()` : race condition (TOCTOU)

**Fichiers** : `scripts/diarize.py` et `scripts/whisper_transcribe.py`

### Problème

```python
tmp_wav = tempfile.mktemp(suffix=".wav", prefix="diarize_")
```

`tempfile.mktemp()` est **officiellement déprécié** dans Python 3. Il retourne un chemin sans créer le fichier — entre le `mktemp()` et le `subprocess.run(...ffmpeg... wav_path...)`, un autre process peut créer un fichier avec le même nom (TOCTOU race condition). Sur un serveur multi-utilisateurs, c'est exploitable.

Les deux scripts font pareil (`mktemp` dans whisper_transcribe.py).

### Fix requis

```python
import tempfile
with tempfile.NamedTemporaryFile(suffix=".wav", prefix="diarize_", delete=False) as f:
    tmp_wav = f.name
# puis cleanup dans finally
```

---

## 🟡 ISSUE #5 — `use_auth_token` déprécié dans pyannote / transformers

**Fichier** : `scripts/diarize.py`, fonction `run_diarization`

### Problème

```python
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=token,
)
```

`use_auth_token` est déprécié depuis `transformers >= 4.34` en faveur de `token=`. Avec les versions récentes, ça génère un `FutureWarning` qui peut polluer les logs et, dans les futures versions, ce paramètre sera supprimé. `pyannote.audio 3.x` suit les mêmes conventions HuggingFace.

### Fix requis

```python
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=token,
)
```

---

## 🟡 ISSUE #6 — Backward compatibility : comportement identique sans --diarization (OK mais fragile)

**Fichier** : `scripts/reframe_v3.py`

### Analyse

Sans `--diarization`, le flow est :
- `diarization = None`, `track_to_speaker = None`
- Dans `compute_smooth_crop` : `if diarization:` → False → `active_speaker = None` → `speaker_score = 0.0`
- Formule effective : `size * 0.4 + center * 0.2 + continuity * 0.3` → max 0.9

C'est **fonctionnellement correct**. La backward compat est assurée.

⚠️ **Mais** : la formule sans speaker donne un total max de 0.9, alors que **avec** diarization active elle donne max 1.3. L'`HYSTERESIS_MARGIN = 0.15` a une signification différente dans les deux modes (15% de 0.9 vs 15% de 1.3). Si un jour on compare des stats cross-mode ou si on ajuste le margin en production, cette asymétrie créera de la confusion.

---

## 🟢 Ce qui est bien fait

- **Fallback diarization vide** (`track_to_speaker = {}` est falsy → `speaker_score = 0.0`) : correctement géré.
- **Edge case trailing silence** dans `detect_silences` : géré explicitement avec `total_duration`.
- **Nettoyage des temp files** : `try/finally` systématique dans `main()` de `reframe_v3.py`.
- **Validation vidéo** : `cap.isOpened()` + `total_frames == 0` checked.
- **ByteTrack error message** : message d'erreur clair avec solution.
- **Stats speaker-framed** : métriques Rush 2 correctement implémentées (%)

---

## Récapitulatif des Issues

| # | Sévérité | Fichier | Description |
|---|----------|---------|-------------|
| 1 | 🔴 CRITIQUE | reframe_v3.py | W_SPEAKER=0.4 + hysteresis incohérents → speaker switch ne fonctionne pas |
| 2 | 🔴 HIGH | whisper_transcribe.py | Zéro gestion erreurs API, pas de retry, pas de check taille fichier |
| 3 | 🟠 MEDIUM | reframe_v3.py | `last_dets` partagé par référence → votes doublés en phase 1 |
| 4 | 🟠 MEDIUM | diarize.py + whisper_transcribe.py | `tempfile.mktemp()` déprécié, race condition TOCTOU |
| 5 | 🟡 LOW | diarize.py | `use_auth_token` déprécié → FutureWarning / future break |
| 6 | 🟡 LOW | reframe_v3.py | Asymétrie de scoring avec/sans diarization (hysteresis non comparable) |

---

## Critères ROADMAP Rush 2 : Checklist

| Critère | Status |
|---------|--------|
| ✅ Speaker actif cadré >80% du temps où il parle | ❌ **FAIL** — Issue #1 bloque le switch |
| ✅ Transitions smooth, pas de ping-pong | ⚠️ Conditionnellement OK (si switch fonctionne) |
| ✅ Pipeline fonctionne avec OU sans diarization | ✅ OK |
| T2.2 Whisper word timestamps | ✅ Implémenté |
| T2.2 Single source of truth pour sous-titres | ✅ Conforme |

**Rush 2 ne peut pas passer le QA en l'état** sur le critère principal (Issue #1).

---

## Actions Requises (par ordre de priorité)

1. **BLOCKER** — Corriger W_SPEAKER et la logique de scoring speaker dans `compute_smooth_Crop` (Issue #1). Tester sur un clip 2-speakers avec transcript.
2. **BEFORE MERGE** — Ajouter retry + file size check dans `whisper_transcribe.py` (Issue #2).
3. **BEFORE MERGE** — Corriger `last_dets` par copie dans `detect_persons_pass` (Issue #3).
4. **NICE TO HAVE** — Remplacer `mktemp()` par `NamedTemporaryFile` (Issue #4) + `use_auth_token` → `token` (Issue #5).

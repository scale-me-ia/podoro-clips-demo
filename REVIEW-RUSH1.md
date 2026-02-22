# REVIEW-RUSH1.md — Code Review Adversaire
**Rush 1 : Person Detection + Tracking (`reframe_v3.py`)**
**Date** : 2026-02-22 | **Reviewer** : Sub-agent adversaire

---

## Summary

- Issues : **3 critical, 4 high, 5 medium**
- Verdict : ⚠️ **CHANGES REQUESTED** (CRITICAL > 0, obligatoire)

---

## Findings

| # | Severity | File:Line | Issue | Fix |
|---|----------|-----------|-------|-----|
| 1 | 🔴 CRITICAL | `reframe_v3.py:407-409` | Temp files jamais nettoyés si erreur | Wrap dans `try/finally` |
| 2 | 🔴 CRITICAL | `reframe_v3.py:196-198` | VFR video → désync A/V sur silence cut | Forcer CFR en entrée ou recalculer timestamps |
| 3 | 🔴 CRITICAL | `pipeline_final.py:449-451` | Fallback v2 ne se déclenche pas si ultralytics absent | Tester l'import, pas l'existence du fichier |
| 4 | 🟡 HIGH | `reframe_v3.py:110-114` | Fallback 0-détections : freeze, pas drift vers center | Appliquer le LERP même sans dets |
| 5 | 🟡 HIGH | `reframe_v3.py:52` | `bytetrack.yaml` non validé → crash opaque | Vérifier la présence ou gérer l'exception |
| 6 | 🟡 HIGH | `reframe_v3.py:50-51` | Vidéo corrompue/vide = output silencieux 0 frames | Valider `total_frames > 0` après cap.open |
| 7 | 🟡 HIGH | `reframe_v3.py:231` | Codec `mp4v` (MPEG-4 Part 2) = double encoding, perte qualité | Utiliser `avc1` (H.264) ou pipe ffmpeg |
| 8 | 🟢 MEDIUM | `reframe_v3.py:27-30` | `W_UNUSED=0.1` absent de la formule → score max = 0.9, pas 1.0 | Inclure dans la somme ou supprimer |
| 9 | 🟢 MEDIUM | `reframe_v3.py:45` | `HYSTERESIS_FRAMES_1S = None` — dead code trompeur | Supprimer |
| 10 | 🟢 MEDIUM | `reframe_v3.py:325-328` | `silence_end` final tronqué si vidéo finit sur silence | Gérer le cas `len(starts) != len(ends)` |
| 11 | 🟢 MEDIUM | `pipeline_final.py:41,45` | `REFRAME_SCRIPT_V2` et `REFRAME_FALLBACK` identiques | Supprimer le doublon |
| 12 | 🟢 MEDIUM | `reframe_v3.py:130` | `target_x = crop_x` affecté mais jamais utilisé (dead var) | Supprimer la ligne |

---

## Analyse détaillée

---

### 🔴 CRITICAL #1 — Temp file leak sur erreur FFmpeg

**Fichier** : `reframe_v3.py`, lignes 290-409

```python
tmp_dir = tempfile.mkdtemp(prefix="reframe_v3_")
tmp_video = os.path.join(tmp_dir, "cropped.mp4")
render_cropped(...)          # peut crash
...
final_encode(...)            # lève RuntimeError si FFmpeg échoue !
# Cleanup — JAMAIS ATTEINT si final_encode raise :
os.remove(tmp_video)
os.rmdir(tmp_dir)
```

`final_encode()` lève explicitement `raise RuntimeError("FFmpeg encoding failed")` à la ligne ~310. Il n'y a **aucun `try/finally`**. En batch de 5 clips, ça laisse 5 dossiers temporaires avec des vidéos non compressées de plusieurs centaines de Mo.

**Fix :**
```python
tmp_dir = tempfile.mkdtemp(prefix="reframe_v3_")
tmp_video = os.path.join(tmp_dir, "cropped.mp4")
try:
    render_cropped(...)
    final_encode(...)
finally:
    if os.path.exists(tmp_video):
        os.remove(tmp_video)
    if os.path.exists(tmp_dir):
        os.rmdir(tmp_dir)
```

---

### 🔴 CRITICAL #2 — VFR video → désync A/V quand silence cut actif

**Fichier** : `reframe_v3.py`, lignes 196-198 + 349-360

`render_cropped` écrit les frames avec `cv2.VideoWriter` à fps **constant** (CFR, la valeur retournée par `cap.get(CAP_PROP_FPS)`). Si la vidéo source est VFR (Variable Frame Rate — cas très courant pour les téléchargements YouTube, screen recordings, etc.), les timestamps dans `tmp_video` **ne correspondent plus** aux timestamps de l'original.

Or, `detect_silences(video_path)` calcule les timestamps de silence sur la vidéo **originale** (VFR). Puis `final_encode` applique ces timestamps à `tmp_video` (CFR). Résultat : les coupures silence sont décalées dans le temps → désynchronisation A/V dans l'output final.

Exemple concret : vidéo YT à 24fps réel mais 29.97fps nominal. OpenCV lit les frames à 29.97fps, écrit `tmp_video` à 29.97fps CFR. La silence à t=30s dans l'original est à t=37.5s dans `tmp_video`. Le cut se fait au mauvais endroit.

**Fix** : Avant `render_cropped`, convertir la source en CFR avec ffmpeg :
```python
# Normalisation CFR
cfr_path = os.path.join(tmp_dir, "cfr_source.mp4")
subprocess.run([
    "ffmpeg", "-y", "-i", video_path,
    "-vf", "fps=fps", "-vsync", "cfr",   # force CFR
    "-c:v", "libx264", "-crf", "0",       # lossless
    cfr_path
], check=True, capture_output=True)
# Utiliser cfr_path dans detect_silences ET render_cropped
```

Ou plus simplement : détecter si VFR avec `ffprobe` et avertir/bloquer.

---

### 🔴 CRITICAL #3 — Fallback v2 cassé si ultralytics non installé

**Fichier** : `pipeline_final.py`, lignes 449-455

```python
if os.path.exists(REFRAME_SCRIPT):     # ← vérifie UNIQUEMENT si le fichier existe
    reframe_script = REFRAME_SCRIPT
else:
    print(f"  [clip {clip_num}] ⚠️  v3 not found, falling back to v2")
    reframe_script = find_script("reframe_v2.py", REFRAME_FALLBACK)
```

Le commentaire dit `"# Fallback: reframe_v2 if YOLO not available"` mais la condition ne teste **pas** si YOLO est disponible. Elle teste si le **fichier** `scripts/reframe_v3.py` existe. Il existera toujours dans le repo.

Si `ultralytics` n'est pas installé, le subprocess `reframe_v3.py` crash avec `ImportError`. `result.returncode != 0` → `produce_clip` retourne `None`. **Pour tous les clips.** Le fallback v2 n'est jamais déclenché.

**Fix** :
```python
def is_yolo_available() -> bool:
    try:
        import importlib
        importlib.util.find_spec("ultralytics")
        return True
    except Exception:
        return False

# Dans produce_clip :
if os.path.exists(REFRAME_SCRIPT) and is_yolo_available():
    reframe_script = REFRAME_SCRIPT
else:
    ...fallback...
```

Ou tester au démarrage du pipeline avec un subprocess rapide :
```python
result = subprocess.run([sys.executable, "-c", "from ultralytics import YOLO"],
                        capture_output=True)
yolo_ok = result.returncode == 0
```

---

### 🟡 HIGH #4 — Fallback 0-détections : freeze au lieu de drift

**Fichier** : `reframe_v3.py`, lignes 110-114

```python
if not dets:
    # Fallback: hold position, drift to center slowly
    target_x = crop_x  # hold           ← dead variable, jamais utilisé
    crop_positions.append(int(round(crop_x)))
    continue                             ← saute le LERP entièrement
```

Le commentaire promet un drift vers le centre. Le code **ne dérive pas**. La variable `target_x` est affectée et immédiatement abandonnée (le `continue` saute tout le bloc de smoothing). Sur une séquence longue sans détection (ex: B-roll, plan large vide), la caméra reste figée à la dernière position, potentiellement hors-centre.

**Fix** :
```python
if not dets:
    # Drift toward center when no person detected
    center_x = float((src_w - crop_w) / 2)
    drift_alpha = 0.02  # très lent, pas de jump visible
    crop_x = crop_x + drift_alpha * (center_x - crop_x)
    crop_x = max(0.0, min(crop_x, float(src_w - crop_w)))
    crop_positions.append(int(round(crop_x)))
    continue
```

---

### 🟡 HIGH #5 — `bytetrack.yaml` : crash opaque si absent

**Fichier** : `reframe_v3.py`, ligne 52

```python
results = model.track(frame, persist=True, tracker="bytetrack.yaml", ...)
```

`bytetrack.yaml` doit être trouvé par ultralytics dans son répertoire de configuration. Sur certaines installations (pip install minimal, environnements isolés), ce fichier n'est pas présent. L'erreur résultante est une exception ultralytics non documentée, difficile à diagnostiquer.

**Fix** : Valider la présence du fichier au démarrage de `detect_persons_pass()`, ou catcher et log clairement :
```python
from ultralytics.utils import SETTINGS
# Ou simplement :
try:
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", ...)
except Exception as e:
    if "bytetrack" in str(e).lower():
        raise RuntimeError(
            "bytetrack.yaml introuvable. Réinstaller ultralytics: pip install ultralytics[tracker]"
        ) from e
    raise
```

---

### 🟡 HIGH #6 — Vidéo corrompue/vide → output 0 frame sans erreur

**Fichier** : `reframe_v3.py`, lignes 57-80

Si `cv2.VideoCapture(video_path)` échoue (fichier corrompu, format non supporté), `cap.read()` retourne `(False, None)` dès le premier appel. Le while loop se termine immédiatement. `all_detections = []`, `crop_positions = []`. `render_cropped` écrit 0 frames. `VideoWriter.release()` crée un mp4 vide valide. `final_encode` encode une vidéo de 0 secondes. Pipeline : aucune exception levée.

**Fix** :
```python
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")
if total_frames == 0:
    raise RuntimeError(f"Video has 0 frames: {video_path}")
```

---

### 🟡 HIGH #7 — Double encoding avec perte qualité (mp4v)

**Fichier** : `reframe_v3.py`, ligne 231

```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # MPEG-4 Part 2
writer = cv2.VideoWriter(tmp_video, fourcc, fps, (OUT_W, OUT_H))
```

`mp4v` est un codec lossy ancien (MPEG-4 Part 2). Toute la vidéo est décodée, croppée, redimensionnée, **recompressée en mp4v**, puis dans `final_encode` **recompressée une seconde fois en H.264**. Deux compressions avec perte sur une vidéo déjà compressée → artefacts cumulatifs, banding, macroblocking sur les zones uniformes (fonds de plateau).

**Fix** : Utiliser H.264 lossless pour l'intermédiaire :
```python
fourcc = cv2.VideoWriter_fourcc(*'avc1')
```
Ou mieux, pipe directement les frames vers ffmpeg avec `subprocess.Popen` pour éviter une passe d'encodage intermédiaire.

---

### 🟢 MEDIUM #8 — `W_UNUSED = 0.1` non utilisé dans la formule

**Fichier** : `reframe_v3.py`, lignes 27-30

```python
W_SIZE = 0.4; W_CENTER = 0.2; W_CONTINUITY = 0.3; W_UNUSED = 0.1  # réservé
total = (size_score * W_SIZE + center_score * W_CENTER + continuity_score * W_CONTINUITY)
# W_UNUSED jamais dans la formule !
```

Score max = 0.4 + 0.2 + 0.3 = **0.9** (pas 1.0). Le `speaker_score` de Rush 2 devra être ajouté dans la formule. Si ajouté naïvement (`+ speaker_score * 0.1`), ça marchera, mais la situation actuelle est confuse : les weights semblent calibrés mais le résultat ne l'est pas.

**Fix** : Soit inclure `W_UNUSED` dans la formule dès maintenant avec une valeur de 0 par défaut, soit documenter explicitement que le max est 0.9.

---

### 🟢 MEDIUM #9 — `HYSTERESIS_FRAMES_1S = None` : dead code

**Fichier** : `reframe_v3.py`, ligne 45

```python
HYSTERESIS_FRAMES_1S = None  # computed from fps
```

Cette variable globale est **déclarée mais jamais utilisée**. La vraie valeur est calculée inline dans `compute_smooth_crop()` : `hysteresis_frames = int(fps * 1.0)`. La globale trompe le lecteur en laissant penser qu'elle est peuplée quelque part.

**Fix** : Supprimer la ligne, ou la remplacer par `HYSTERESIS_SECONDS = 1.0` et l'utiliser dans `compute_smooth_crop`.

---

### 🟢 MEDIUM #10 — Silence final tronqué silencieusement

**Fichier** : `reframe_v3.py`, lignes 325-328

```python
starts = re.findall(r"silence_start: ([\d.]+)", stderr)
ends = re.findall(r"silence_end: ([\d.]+)", stderr)
for s, e in zip(starts, ends):   # zip tronque si len(starts) != len(ends)
```

ffmpeg peut émettre un `silence_start` final **sans `silence_end`** correspondant si la vidéo se termine pendant un silence. Dans ce cas `len(starts) == len(ends) + 1`. `zip` tronque silencieusement → le dernier silence est ignoré, les quelques dernières secondes de silence ne sont pas coupées.

**Fix** :
```python
if len(starts) != len(ends):
    print(f"  ⚠️  silence: {len(starts)} starts, {len(ends)} ends — trailing silence ignored")
    # Si on veut couper jusqu'à la fin :
    if len(starts) > len(ends):
        ends.append(str(total_duration))
```

---

### 🟢 MEDIUM #11 — `REFRAME_SCRIPT_V2` et `REFRAME_FALLBACK` en doublon

**Fichier** : `pipeline_final.py`, lignes 41 et 45

```python
REFRAME_SCRIPT_V2 = os.path.join(os.path.dirname(__file__), "scripts", "reframe_v2.py")
REFRAME_FALLBACK  = os.path.join(os.path.dirname(__file__), "scripts", "reframe_v2.py")
```

Deux constantes identiques. `REFRAME_SCRIPT_V2` n'est référencée nulle part dans le code. Confusion.

**Fix** : Supprimer `REFRAME_SCRIPT_V2`, garder `REFRAME_FALLBACK`.

---

### 🟢 MEDIUM #12 — Variable `target_x` morte (dead variable)

**Fichier** : `reframe_v3.py`, ligne 130 (dans le bloc `if not dets:`)

```python
target_x = crop_x  # hold
```

Cette affectation est suivie immédiatement d'un `continue` qui saute le reste du loop body, dont la partie qui utiliserait `target_x`. Variable affectée mais jamais lue. Confuse pour un futur mainteneur.

---

## Cross-check ROADMAP Rush 1 vs Implémentation

| Critère ROADMAP | Implémenté ? | Remarque |
|-----------------|-------------|---------|
| YOLOv8n person detection | ✅ | OK |
| ByteTrack tracking | ✅ | OK, `persist=True` |
| Subject selection (size + center + continuity) | ✅ | OK |
| Speaker score (optional) | ⬜ | Réservé W_UNUSED=0.1, Rush 2 |
| Hysteresis > 1s | ✅ | `fps * 1.0` |
| Hysteresis margin > 0.15 | ⚠️ | **Absent.** Le ROADMAP dit "nouveau score > ancien + 0.15" mais la formule n'implémente pas ce delta — elle regarde juste qui est le meilleur scorer, avec continuity_score comme proxy. |
| Smooth crop (lerp 0.06) | ✅ | OK |
| Max move 5%/frame | ✅ | OK |
| Faster alpha 0.12 on subject switch | ✅ | OK |
| Silence cutting from v2 | ✅ | OK (avec bug #10) |
| Fallback 0 personnes → center | ⚠️ | Hold, pas drift (bug #4) |
| Tests sur raw clip | ? | Non vérifiable en review |
| Intégration pipeline_final.py | ✅ | Fait mais fallback cassé (bug #3) |

**Hysteresis margin manquante** : Le ROADMAP exige "nouveau score > ancien + 0.15 margin". Le code fait juste "best_id != current_subject_id" sans comparer les scores des deux subjects. La vraie hysteresis par margin n'est pas implémentée — c'est le `hysteresis_frames` (1 seconde) qui fait office de garde-fou, pas une comparaison de scores.

---

## Verdict

⚠️ **CHANGES REQUESTED**

3 criticals bloquants :
1. **Temp leak** → en prod batch, le disque se remplit
2. **VFR desync** → tous les clips de contenu YouTube auront le silence cut décalé
3. **Fallback v2 cassé** → si ultralytics non dispo, pipeline sort 0 clips sans message clair

Priorité de fix : **CRITICAL #3 en 5 min** (check import), **CRITICAL #1 en 10 min** (try/finally), **CRITICAL #2 nécessite réflexion** (VFR handling).

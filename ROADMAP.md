# 🎬 Podoro Clips — BMAD Roadmap (22/02/2026)

**Objectif** : Pipeline auto podcast → clips viraux 9:16 avec recadrage intelligent sur les personnages
**Workspace** : `/Users/OpenClaw/.openclaw/workspace-podoro-clips/podoro-clips/`
**Repo** : github.com/scale-me-ia/podoro-clips
**Score actuel cadrage** : 81/100 (B2 MediaPipe face) — mais échoue en plan large

---

## 🔍 Diagnostic technique (22/02/2026)

### Le problème core
Le reframe actuel (reframe_v2.py + reframe_smart.py) utilise **MediaPipe Face Detection** (BlazeFace short range) qui :
1. **Détecte des VISAGES** (pas des personnes) → en plan large, les visages sont trop petits → detection_confidence < 0.25 → miss
2. **Quand pas de face détectée** → hold la dernière position ou center crop → personne au milieu de l'écran sur du vide
3. **Pas de tracking persistant** → quand il retrouve un visage, il peut jump sur un autre
4. **reframe_smart.py** a ajouté smoothing (lerp 0.92) + hysteresis (15 frames) + face buffer (30 frames) → améliore la stabilité mais ne résout pas le problème fondamental : pas de détection en plan large

### Versions existantes
| Script | Approche | Limite |
|--------|----------|--------|
| `reframe_face.py` | MediaPipe basique, crop moyen | Statique, pas de tracking |
| `reframe_v1.py` | Haar Cascade | Encore plus basique |
| `reframe_v2.py` | MediaPipe per-frame + silence cut | Jump entre segments, fail en plan large |
| `reframe_smart.py` | MediaPipe + lerp + hysteresis + buffer | Smooth mais toujours basé sur face detection |
| `reframe_diarize.py` | MediaPipe + pyannote speaker map | Sait qui parle mais même problème de détection |

### Ce qui manque
1. **Détection de PERSONNES** (pas juste visages) — YOLO/MediaPipe Pose fonctionne en plan large
2. **Tracking persistant** — SORT/DeepSORT/ByteTrack pour garder l'identité des personnes entre frames
3. **Sélection intelligente du sujet** — règle : toujours un humain visible et centré
4. **Smooth camera** — interpolation cubique, pas de jumps

---

## 🔴 Rush 1 — Person Detection + Tracking (core fix) (3-4h)
**Objectif** : Remplacer le face detection par du person detection + tracking. Le crop doit TOUJOURS cadrer sur une personne.

### T1.1 — Nouveau script `reframe_v3.py` (from scratch)
Architecture :
```
Input 16:9 → Person Detection (YOLO/ultralytics) → Tracking (ByteTrack) → Subject Selection → Smooth Crop → Output 9:16
```

- [ ] Installer `ultralytics` (YOLOv8) : `pip3 install ultralytics`
  - YOLOv8n (nano) suffit pour person detection, ~6ms/frame sur M4
  - Détecte la classe "person" (class 0) avec bounding box complet (pas juste le visage)
  - Fonctionne en plan large, plan moyen, gros plan — c'est LE fix
- [ ] Tracking avec ByteTrack (intégré dans ultralytics) :
  ```python
  from ultralytics import YOLO
  model = YOLO("yolov8n.pt")
  results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0])
  # Chaque personne a un track_id persistant entre frames
  ```
- [ ] **Subject Selection** — algorithme de sélection du sujet à cadrer :
  1. **Score par personne par frame** :
     - `size_score` = bbox area / frame area (plus gros = plus proche = plus important)
     - `center_score` = 1 - distance du centre horizontal (personne déjà au centre = bonus)
     - `continuity_score` = même track_id que le sujet précédent ? bonus 0.3
     - `speaker_score` (optionnel, si diarization dispo) = est-ce que cette personne parle ?
  2. **Formule** : `total = size * 0.4 + center * 0.2 + continuity * 0.3 + speaker * 0.1`
  3. **Hysteresis** : ne changer de sujet que si le nouveau score est > ancien + 0.15 pendant > 1s
  4. **Fallback** : si 0 personne détectée (rare avec YOLO) → hold dernière position 2s → center crop
- [ ] **Smooth Crop** :
  - Target crop_x calculé pour centrer le sujet sélectionné
  - Interpolation exponentielle : `crop_x = lerp(crop_x, target, alpha=0.06)` (très smooth)
  - Limiter la vitesse max de déplacement : max 5% de la largeur source par frame
  - Pour les changements de sujet (nouveau track_id) : transition plus rapide alpha=0.12 sur 0.5s

### T1.2 — Tests comparatifs
- [ ] Tester sur le clip existant `/tmp/sp-clips/sp_rank1_raw.mp4` (32s, interview)
- [ ] Tester sur un passage plan large (2+ personnes éloignées)
- [ ] Comparer côte à côte : reframe_v2 vs reframe_v3
- [ ] Métriques :
  - % de frames avec une personne visible dans le crop
  - Stabilité du crop (avg pixel movement per frame)
  - Jumps > 50px entre frames consécutifs (doit être 0)

### T1.3 — Intégration dans pipeline_final.py
- [ ] Remplacer l'appel à `reframe_v2.py` par `reframe_v3.py` dans `pipeline_final.py`
- [ ] Garder reframe_v2 comme fallback (si YOLO pas installé)
- [ ] Le silence cutting reste dans reframe_v3 (reprendre le code de v2)

### T1.4 — QA Rush 1
- [ ] Vidéo output du clip test → screenshot frame 0, frame milieu, frame fin
- [ ] Vérifier : une personne est TOUJOURS visible et centrée
- [ ] Vérifier : transitions smooth, 0 jump visible
- [ ] Comparer visuellement avec reframe_v2 sur le même clip
- [ ] Push sur GitHub

**Critères de validation Rush 1** :
- ✅ Personne détectée dans >95% des frames (vs ~70% avec face detection)
- ✅ 0 jump visible entre frames consécutifs
- ✅ Le sujet le plus pertinent est toujours centré
- ✅ Fonctionne en plan large (2+ personnes) ET gros plan

---

## 🟡 Rush 2 — Speaker-Aware Reframing (2-3h)
**Objectif** : Quand il y a 2+ personnes, cibler celui qui parle (si diarization dispo).

### T2.1 — Intégration pyannote diarization
- [ ] Reprendre la logique de `reframe_diarize.py` (speaker_face_map) mais avec YOLO person detection
- [ ] Phase 1 : analyser les segments single-person → mapper track_id ↔ speaker_label
- [ ] Phase 2 : quand 2+ personnes, le speaker actif a un bonus de +0.4 dans le score de sélection
- [ ] Si pas de diarization → fallback sur le score size+center (toujours une personne cadrée)

### T2.2 — Whisper word-level timestamps
- [ ] Utiliser Whisper API (déjà dans la pipeline) pour les word timestamps précis
- [ ] Les word timestamps servent aussi pour les sous-titres → single source of truth
- [ ] Remplacer le VTT YouTube (triplé, imprécis) par Whisper comme source primaire

### T2.3 — QA Rush 2
- [ ] Tester sur un passage avec 2 speakers qui alternent
- [ ] Vérifier que le crop suit le speaker actif
- [ ] Vérifier que les transitions entre speakers sont smooth (pas de ping-pong)
- [ ] Push sur GitHub

**Critères de validation Rush 2** :
- ✅ Le speaker actif est cadré dans >80% du temps où il parle
- ✅ Transitions entre speakers : smooth, pas de ping-pong
- ✅ Pipeline fonctionne avec OU sans diarization

---

## 🟢 Rush 3 — Sous-titres next-level (2-3h)
**Objectif** : Sous-titres esthétiques style TikTok pro, pas juste fonctionnels.

### T3.1 — Refactor subtitles avec Whisper word timestamps
- [ ] Supprimer la dépendance au VTT YouTube
- [ ] Input : JSON Whisper avec word-level timestamps
- [ ] Garder le style karaoke (highlight word par word) mais améliorer :

### T3.2 — Améliorations visuelles
- [ ] **Animation d'entrée** : les mots apparaissent avec un léger scale-up (1.0 → 1.05 → 1.0 sur 100ms)
- [ ] **Ombre portée améliorée** : drop shadow au lieu du glow blur (plus net, plus lisible)
- [ ] **Couleur highlight dynamique** : au lieu de jaune fixe, le mot actif peut avoir une couleur qui pulse légèrement
- [ ] **Position adaptative** : les sous-titres se positionnent pour ne pas chevaucher le visage du sujet
  - Récupérer la position Y du sujet depuis le tracking YOLO
  - Si sujet en bas → sous-titres en haut (et vice-versa)
  - Fallback : 70% de la hauteur (position actuelle)
- [ ] **Background pill** : option d'un fond semi-transparent derrière le texte (style CapCut)
- [ ] **Emoji/punctuation styling** : les ! et ? sont un peu plus grands

### T3.3 — Font & sizing
- [ ] Tester 2-3 fonts alternatives (Montserrat Bold existe déjà en fallback)
- [ ] Auto-sizing : font_size s'adapte à la longueur du mot le plus long dans le groupe
- [ ] Max 2 lignes, 3-4 mots par ligne (déjà en place, vérifier)

### T3.4 — QA Rush 3
- [ ] Screenshot des sous-titres sur 3 frames différents
- [ ] Comparer avant/après visuellement
- [ ] Vérifier la lisibilité sur fond clair ET fond sombre
- [ ] Vérifier le positionnement adaptatif (ne chevauche pas le visage)
- [ ] Push sur GitHub

**Critères de validation Rush 3** :
- ✅ Sous-titres lisibles sur tout type de fond
- ✅ Animation d'entrée smooth
- ✅ Positionnement adaptatif (pas de chevauchement visage)
- ✅ Look "pro TikTok", pas amateur

---

## 🔵 Rush 4 — Pipeline E2E + Multi-source (2-3h)
**Objectif** : Pipeline complète et robuste, capable de traiter n'importe quel podcast vidéo.

### T4.1 — Pipeline unifiée `pipeline_v2.py`
- [ ] Refactor `pipeline_final.py` → `pipeline_v2.py` avec :
  1. **Input flexible** : URL YouTube OU fichier local (mp4/mkv/webm)
  2. **Download** : yt-dlp si URL YouTube
  3. **Transcription** : Whisper API (word timestamps) — remplace le VTT
  4. **Scoring** : Gemini + Claude hybride (inchangé, ça marche bien)
  5. **Extraction** : ffmpeg clip
  6. **Reframe** : reframe_v3 (YOLO + ByteTrack)
  7. **Sous-titres** : subtitles_v2 (Whisper-based, esthétiques)
  8. **Post-prod** : color grade léger + Ken Burns optionnel
  9. **Output** : `output/{podcast_name}/clip_{N}_{score}.mp4` + `results.json`

### T4.2 — CLI propre
```bash
# Depuis YouTube
python pipeline_v2.py --url "https://youtube.com/watch?v=..." --out ./output/

# Depuis fichier local
python pipeline_v2.py --video ./podcast.mp4 --out ./output/

# Options
  --max-clips 5          # nombre max de clips
  --min-score 60         # score minimum pour générer un clip
  --skip-subtitles       # sans sous-titres
  --no-diarization       # sans speaker tracking
  --style tiktok|reels   # préréglages de format
```

### T4.3 — Gestion d'erreurs robuste
- [ ] Retry automatique sur les appels API (Whisper, Claude, Gemini) — 3 tentatives avec backoff
- [ ] Si un clip échoue → log l'erreur, passer au suivant (pas de crash pipeline)
- [ ] Progress bar (tqdm) pour le processing
- [ ] Résumé final : X clips générés, Y échoués, temps total, coût API estimé

### T4.4 — QA Rush 4
- [ ] Test E2E depuis une URL YouTube → 3 clips en output
- [ ] Test depuis un fichier local
- [ ] Test avec une vidéo courte (<5min) et une longue (>1h)
- [ ] Vérifier les résultats JSON
- [ ] Push sur GitHub avec README mis à jour

**Critères de validation Rush 4** :
- ✅ Pipeline E2E fonctionnelle (YouTube → clips)
- ✅ 0 crash sur erreur individuelle
- ✅ CLI intuitive avec options
- ✅ README à jour avec exemples

---

## 🟣 Rush 5 — Intégration Podoro + Distribution (2-3h)
**Objectif** : Connecter la pipeline clips à l'app Podoro et distribuer automatiquement.

### T5.1 — Edge Function `extract-clips` (déjà déployée mais à vérifier)
- [ ] Vérifier l'Edge Function existante `extract-clips` sur Supabase
- [ ] Si elle n'utilise pas la v3 → la mettre à jour
- [ ] Trigger : quand un épisode passe en `published` → lancer l'extraction de clips
- [ ] Stocker les clips dans Supabase Storage

### T5.2 — Distribution automatique
- [ ] Post automatique sur les réseaux (via Buffer API ou APIs natives) :
  - Instagram Reels
  - TikTok
  - Twitter/X
  - LinkedIn (optionnel)
- [ ] Schedule : post les clips à des heures optimales (pas tous d'un coup)

### T5.3 — ✅ DONE (Neo, 23/02)
- Backend infra fait : trigger Postgres, LaunchAgent clip_worker, Edge Functions
- Reste : UI + distribution (voir Rush 6)

### T5.4 — QA Rush 5
- [ ] Test E2E : nouvel épisode → pipeline → clips stockés
- [ ] Push sur GitHub

**Critères de validation Rush 5** :
- ✅ Pipeline automatique : épisode publié → clips générés
- ✅ Clips stockés dans Supabase Storage

---

## 🟣 Rush 6 — Dashboard Admin Clips sur podoro.fr (2-3h)
**Objectif** : Page `/admin/clips` dans l'app Podoro (https://podoro.fr/admin) pour tracker et gérer tous les clips générés.
**Workspace frontend** : `/Users/OpenClaw/.openclaw/workspace-anthropic/Capsule/web-app/`

### T6.1 — Page `/admin/clips`
- [ ] Route `/admin/clips` dans le router existant
- [ ] Vue tableau/grille de tous les clips avec :
  - Thumbnail preview
  - Titre de l'épisode source
  - Score du clip
  - Hook text
  - Durée
  - Statut (pending → processing → ready → distributed → error)
  - Date de création
- [ ] Filtres : par épisode, par statut, par score
- [ ] Tri : par date, score, statut

### T6.2 — Player & Actions
- [ ] Player vertical intégré (click sur un clip → modal avec preview 9:16)
- [ ] Boutons d'action par clip :
  - ▶️ Preview
  - 📥 Download
  - 🗑️ Supprimer
  - 🔄 Relancer la génération
  - 📤 Distribuer manuellement (choix réseau)
- [ ] Bulk actions : distribuer / supprimer plusieurs clips

### T6.3 — Stats overview
- [ ] Compteurs en haut de page : total clips, en attente, distribués, erreurs
- [ ] Graphique clips générés par semaine (optionnel)

### T6.4 — QA Rush 6
- [ ] Page accessible et fonctionnelle sur podoro.fr/admin
- [ ] Player preview fonctionne
- [ ] Actions CRUD opérationnelles
- [ ] Responsive (desktop + mobile)
- [ ] Push sur GitHub

**Critères de validation Rush 6** :
- ✅ Dashboard admin clips fonctionnel sur podoro.fr/admin
- ✅ Preview, download, suppression, relance
- ✅ Filtres et stats

---

## 🔵 Rush 7 — YouTube Video Scanner (2-3h)
**Objectif** : Scanner automatiquement YouTube pour trouver les versions vidéo des derniers épisodes de podcasts suivis, et lancer le processus de clips.

### T7.1 — Migration DB
- [ ] Ajouter colonne `youtube_channel_id` (text, nullable) à la table `podcasts`
- [ ] Ajouter colonne `youtube_url` (text, nullable) à la table `episodes`
- [ ] Ajouter colonne `has_video` (boolean, default false) à la table `episodes`

### T7.2 — Script/Edge Function `scan-youtube-videos`
- [ ] Pour chaque podcast actif ayant un `youtube_channel_id` :
  1. Utiliser `yt-dlp --flat-playlist` ou YouTube Data API v3 (search.list) pour lister les dernières vidéos de la chaîne
  2. Matcher les vidéos avec les épisodes existants par titre (fuzzy match — les titres YouTube et RSS diffèrent souvent légèrement)
  3. Si match trouvé → mettre à jour `episodes.youtube_url` et `has_video = true`
  4. Si l'épisode est `published` et `has_video = true` et pas de clips existants → créer un run dans la table clips pour trigger le clip_worker
- [ ] Scoring de matching : Levenshtein distance ou token overlap > 70%
- [ ] Log les matches et les misses pour debug

### T7.3 — Mapping initial des chaînes YouTube
Podcasts actifs à mapper :
| Podcast | YouTube probable |
|---------|----------------|
| Le Gratin | @legratin |
| La Martingale | @LaMartingale |
| Génération Do It Yourself | @GenerationDoItYourself |
| Sans Permission | @SansPermission |
| Marketing Mania | @MarketingMania |
| Tribu Indé | @TribuInde |
| Le Panier | @LePanier |
| TheBBoost | @TheBBoost |
| Serial Entrepreneurs | @serialentrepreneurs |
| 2 Heures de Perdues | @2heuresdeperdues |
| L'Envolée | à chercher |
| Little Big Things | à chercher |

- [ ] Script de résolution : pour chaque podcast, chercher la chaîne YouTube via `yt-dlp "ytsearch:{podcast_name} podcast"` et valider manuellement
- [ ] Insérer les `youtube_channel_id` dans la table podcasts

### T7.4 — Cron automatique
- [ ] pg_cron ou cron OpenClaw : lancer `scan-youtube-videos` toutes les 6h
- [ ] Ne scanner que les épisodes des 7 derniers jours (pas tout l'historique)
- [ ] Rate limiting : max 50 requêtes YouTube par scan

### T7.5 — UI dans /admin/clips
- [ ] Ajouter un bouton "🔍 Scanner YouTube" dans le dashboard admin clips
- [ ] Afficher un indicateur "📹 Vidéo dispo" sur les épisodes qui ont un `youtube_url`
- [ ] Permettre de lancer manuellement la génération de clips depuis un épisode avec vidéo

### T7.6 — QA Rush 7
- [ ] Tester le scan sur 2-3 podcasts connus (Sans Permission, Marketing Mania)
- [ ] Vérifier le matching titre
- [ ] Vérifier que le clip_worker se lance automatiquement
- [ ] Push sur GitHub

**Critères de validation Rush 7** :
- ✅ Scanner trouve les vidéos YouTube des épisodes récents
- ✅ Matching titre > 80% de précision
- ✅ Clips auto-générés pour les épisodes avec vidéo
- ✅ Bouton scan manuel dans l'admin

---

## 🟤 Rush 8 — Distribution Réseaux Sociaux (2-3h)
**Objectif** : Distribution automatique des clips sur les réseaux.

### T7.1 — APIs réseaux sociaux
- [ ] Instagram Reels (via Instagram Graph API)
- [ ] TikTok (via TikTok Content Posting API)
- [ ] Twitter/X (via Twitter API v2)
- [ ] LinkedIn (optionnel)
- [ ] Compléter l'Edge Function `distribute-clips` avec les vraies APIs

### T7.2 — Scheduling
- [ ] Post à des heures optimales (pas tous d'un coup)
- [ ] File d'attente avec espacement configurable

### T7.3 — QA Rush 7
- [ ] Test distribution sur au moins 1 réseau
- [ ] Vérifier que le statut se met à jour dans le dashboard admin

**Critères de validation Rush 7** :
- ✅ Distribution fonctionnelle sur au moins 1 réseau
- ✅ Statut mis à jour dans le dashboard

---

## Règles BMAD
1. **Rushs séquentiels** — Rush N+1 ne commence pas sans QA + Code Review Rush N
2. **Code Review Adversaire** après chaque rush
3. **Protocole HALT** si blocage
4. **Rush 1 est CRITIQUE** — c'est le core fix. Pas de rush suivant sans validation du recadrage.
5. **Tester sur de VRAIES vidéos** — pas de mock, pas de test synthétique
6. **Agent sur Opus** pour Rush 1 (technique complexe), Sonnet pour le reste
7. **Commits propres**, push GitHub à chaque rush

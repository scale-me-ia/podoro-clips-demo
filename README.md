# Podoro Clips Engine

Pipeline automatique de génération de clips vidéo courts (Reels/Shorts) depuis des épisodes de podcast.

---

## Architecture Rush 5

```
[Podcast RSS]
     │ episode.status → 'published'
     ▼
[DB Webhook] ──► trigger-clips (Edge Function)
                       │ INSERT clip_pipeline_runs (status=pending)
                       ▼
             [clip_worker.py — Mac Mini]
                  polls every 60s
                       │
                       ├── Download audio/video
                       ├── pipeline_v2.py (Whisper + Claude + FFmpeg)
                       ├── Upload to Storage bucket 'clips'
                       └── INSERT clips (status=ready)
                                │
                                ▼
                    [extract-clips (Edge Function)]
                         GET ?episode_id=xxx
                              │
                              ▼
                    [distribute-clips (Edge Function)]
                         POST {clip_id, platforms}
                              │
                              ▼
                    [distributions table]
                    (instagram/tiktok/twitter/linkedin)
```

### Edge Functions
| Fonction | Méthode | Description |
|----------|---------|-------------|
| `extract-clips` | GET `?episode_id=xxx` | Retourne les clips `status=ready` d'un épisode |
| `extract-clips` | POST `{episode_id}` | Crée un pipeline run (pending) |
| `trigger-clips` | POST (webhook) | Déclenché par DB webhook quand episode passe à `published` |
| `distribute-clips` | POST `{clip_id, platforms}` | Crée des distributions (pending) pour les réseaux sociaux |

### Mac Mini Worker
- `clip_worker.py` — Poll Supabase toutes les 60s, exécute `pipeline_v2.py`
- `output_worker/{run_id}/` — Répertoire de sortie par run
- Storage bucket `clips` — `{episode_id}/{clip_index}_{slug}.mp4`

---

## Déclencher manuellement un clip

```bash
SUPA_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
EPISODE_ID="votre-episode-uuid"

# 1. Créer un pipeline run via extract-clips
curl -s -X POST \
  "https://agypzrkevayucfvmawee.supabase.co/functions/v1/extract-clips" \
  -H "Authorization: Bearer $SUPA_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"episode_id\": \"$EPISODE_ID\", \"triggered_by\": \"manual\"}"

# 2. Résultat : {"run_id": "xxx", "status": "pending", "episode_title": "..."}
# Le worker va le ramasser dans les 60s.

# 3. Vérifier les clips générés (après traitement)
curl -s \
  "https://agypzrkevayucfvmawee.supabase.co/functions/v1/extract-clips?episode_id=$EPISODE_ID" \
  -H "Authorization: Bearer $SUPA_KEY"

# 4. Distribuer un clip
curl -s -X POST \
  "https://agypzrkevayucfvmawee.supabase.co/functions/v1/distribute-clips" \
  -H "Authorization: Bearer $SUPA_KEY" \
  -H "Content-Type: application/json" \
  -d '{"clip_id": "clip-uuid", "platforms": ["instagram", "tiktok"]}'
```

---

## Configurer le LaunchAgent (Mac Mini)

Le `clip_worker.py` tourne en daemon sur le Mac Mini via un LaunchAgent macOS.

```bash
# 1. Copier le plist
cp com.podoro.clip-worker.plist ~/Library/LaunchAgents/

# 2. Activer le service
launchctl load ~/Library/LaunchAgents/com.podoro.clip-worker.plist
launchctl start com.podoro.clip-worker

# 3. Vérifier le statut
launchctl list | grep podoro

# 4. Voir les logs
tail -f /tmp/podoro-clip-worker.log
tail -f /tmp/podoro-clip-worker.err

# 5. Arrêter le service
launchctl stop com.podoro.clip-worker
launchctl unload ~/Library/LaunchAgents/com.podoro.clip-worker.plist
```

**Test manuel sans daemon :**
```bash
cd /Users/OpenClaw/.openclaw/workspace-podoro-clips/podoro-clips

# Traiter un run spécifique
python3 clip_worker.py --run-id <run-uuid>

# Traiter le prochain run en attente et quitter
python3 clip_worker.py --once

# Mode daemon avec interval custom
python3 clip_worker.py --interval 30
```

---

## Configurer le Database Webhook dans Supabase

Pour déclencher automatiquement un pipeline run quand un épisode passe à `published` :

1. Aller sur **https://supabase.com/dashboard/project/agypzrkevayucfvmawee/database/webhooks**
2. Cliquer **"Create a new hook"**
3. Renseigner :
   - **Name** : `on_episode_published`
   - **Table** : `episodes`
   - **Events** : ✅ `UPDATE`
   - **Type** : `Supabase Edge Functions`
   - **Edge Function** : `trigger-clips`
4. Cliquer **Create**

Le webhook enverra automatiquement le payload `{type, table, schema, record, old_record}` à `trigger-clips` à chaque update d'un épisode. La fonction filtre et n'agit que quand `status` passe à `published`.

---

## DB Schema Rush 5

### Table `clips` (colonnes ajoutées)
| Colonne | Type | Description |
|---------|------|-------------|
| `episode_id` | uuid | FK → episodes.id |
| `video_url` | text | URL publique Storage |
| `thumbnail_url` | text | Miniature du clip |
| `duration_seconds` | integer | Durée du clip |
| `score` | integer | Score de pertinence (0-100) |
| `clip_index` | integer | Index dans l'épisode |
| `format` | text | `9:16` par défaut |
| `hook` | text | Phrase d'accroche |
| `transcript` | jsonb | Transcription word-level |
| `status` | text | `pending` / `ready` / `failed` |

### Table `distributions` (colonnes ajoutées)
| Colonne | Type | Description |
|---------|------|-------------|
| `clip_id` | uuid | FK → clips.id |

---

## TODO — Social API Keys

Pour activer la distribution réelle (actuellement en `pending`) :

```bash
# Instagram
supabase secrets set INSTAGRAM_API_TOKEN=xxx --project-ref agypzrkevayucfvmawee

# TikTok
supabase secrets set TIKTOK_API_KEY=xxx TIKTOK_API_SECRET=xxx --project-ref agypzrkevayucfvmawee

# Twitter/X
supabase secrets set TWITTER_API_KEY=xxx TWITTER_API_SECRET=xxx \
  TWITTER_ACCESS_TOKEN=xxx TWITTER_ACCESS_TOKEN_SECRET=xxx \
  --project-ref agypzrkevayucfvmawee

# LinkedIn
supabase secrets set LINKEDIN_ACCESS_TOKEN=xxx --project-ref agypzrkevayucfvmawee
```

Ensuite, décommenter les `switch` dans `distribute-clips/index.ts` et implémenter chaque plateforme.

---

## Pipeline CLI — pipeline_v2.py (Rush 4)

Pipeline E2E standalone : YouTube URL ou fichier local → clips viraux 9:16 avec sous-titres.

### Installation

```bash
# Dépendances Python
pip install anthropic openai google-generativeai \
    ultralytics supervision pillow tqdm yt-dlp ffmpeg-python

# FFmpeg (macOS)
brew install ffmpeg

# Variables d'environnement
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-proj-...       # pour Whisper
```

### Usage

```bash
# Depuis une URL YouTube
python3 pipeline_v2.py --url "https://youtube.com/watch?v=..." --out ./output/

# Depuis un fichier local
python3 pipeline_v2.py --video ./podcast.mp4 --out ./output/

# Avec options avancées
python3 pipeline_v2.py --video ./podcast.mp4 \
    --max-clips 5 \
    --min-score 7 \
    --out ./output/ \
    --language fr

# Sans sous-titres (plus rapide)
python3 pipeline_v2.py --video ./podcast.mp4 --skip-subtitles

# Sans diarisation (compatible, déjà désactivée par défaut)
python3 pipeline_v2.py --video ./podcast.mp4 --no-diarization

# Dry-run : détection seulement, pas de rendu vidéo
python3 pipeline_v2.py --video ./podcast.mp4 --dry-run

# Avec transcription existante (skip Whisper)
python3 pipeline_v2.py --video ./podcast.mp4 --whisper-json ./transcription.json
```

### Options complètes

| Option | Défaut | Description |
|--------|--------|-------------|
| `--url URL` | — | URL YouTube à télécharger (yt-dlp) |
| `--video PATH` | — | Fichier local (mp4/mkv/webm) |
| `--out DIR` | `./output` | Répertoire de sortie |
| `--max-clips N` | 3 | Nombre max de clips à générer |
| `--min-score N` | 0 | Score minimum 1-10 pour inclure un clip |
| `--language LANG` | `fr` | Langue Whisper |
| `--dry-run` | — | Détection uniquement, pas de vidéo |
| `--no-reframe` | — | Skip reframing (pas de conversion 9:16) |
| `--no-subs` / `--skip-subtitles` | — | Skip sous-titres |
| `--no-diarization` | — | Compatibilité (diarisation non utilisée) |
| `--whisper-json PATH` | — | JSON Whisper existant (skip transcription) |
| `--anthropic-key KEY` | `$ANTHROPIC_API_KEY` | Clé API Claude |
| `--openai-key KEY` | `$OPENAI_API_KEY` | Clé API OpenAI (Whisper) |

### Output

```
output/
└── {podcast_name}/
    ├── clip_1_8.mp4        # clip_N_score.mp4
    ├── clip_2_7.mp4
    ├── clip_3_6.mp4
    ├── transcription.json  # Whisper word-level
    └── results.json        # Métadonnées complètes
```

### Résumé final (exemple)

```
======================================================================
  📊 FINAL SUMMARY
======================================================================
  ✅ Clips generated  : 3
  ❌ Clips failed     : 0
  ⏱️  Total time       : 187s (3.1 min)
  💰 Estimated cost   : ~$0.0234 USD
  📁 Output dir       : ./output/podcast_name/
     📹 clip_1_8.mp4 (67s, 42.3 MB)
     📹 clip_2_7.mp4 (71s, 45.1 MB)
     📹 clip_3_6.mp4 (58s, 36.8 MB)
======================================================================
```

### Worker Supabase

`pipeline_v2.py` — Tourne sur Mac Mini, orchestré par `clip_worker.py` :
- 🎙️ Download audio/video (yt-dlp ou URL directe)
- 📝 Transcription Whisper (word-level timestamps)
- 🤖 Scoring Claude (highlight detection)
- ✂️ Découpe FFmpeg
- 📱 Reframe 9:16 (YOLO face tracking)
- 🔠 Burn-in sous-titres (word-level, animés)
- ☁️ Upload Supabase Storage

---

## Supabase Project

- **Ref** : `agypzrkevayucfvmawee`
- **URL** : `https://agypzrkevayucfvmawee.supabase.co`
- **Dashboard** : https://supabase.com/dashboard/project/agypzrkevayucfvmawee

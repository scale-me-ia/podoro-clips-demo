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

## Pipeline principal

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

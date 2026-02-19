# Sans Permission × yt-dlp — Highlights Test

**Épisode**: Oussama : "Je dors 4h/nuit depuis claude code"
**Durée totale**: 1h38
**Transcript**: Auto-captions YouTube (fr) → nettoyé → analysé par Claude

## 🏆 Top 5 Passages Viraux (identifiés par IA)

| Rank | Timestamp | Durée | Hook | Format |
|------|-----------|-------|------|--------|
| 1 | 01:01–01:33 | 32s | "86% des startups US utilisent du code chinois !" | money_reveal |
| 2 | 02:03–02:33 | 30s | "La barrière de la langue explose grâce à l'IA !" | counterintuitive |
| 3 | 00:31–01:01 | 30s | "La Chine innove, ce n'est plus du copier-coller !" | counterintuitive |
| 4 | 02:33–03:03 | 30s | "Traduire en direct avec des AirPods magiques !" | tip |
| 5 | 01:33–02:03 | 30s | "Les open sources chinoises dominent la tech mondiale" | tension |

## Clips disponibles (Rank #1)
- `sp_rank1_16x9.mp4` — Format paysage 1920×1080 (YouTube / LinkedIn)
- `sp_rank1_9x16.mp4` — Format vertical 1080×1920 (TikTok / Reels)

## Pipeline
1. yt-dlp → auto-captions YouTube en VTT
2. Parsing + nettoyage du transcript
3. Claude Haiku → identification des 5 meilleurs passages avec hooks
4. yt-dlp --download-sections → extraction des clips exacts
5. ffmpeg → conversion 9:16 vertical

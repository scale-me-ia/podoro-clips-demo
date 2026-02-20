# Podoro Clips — Matrice A/B Test
## Date : 2026-02-20

## Source : Sans Permission (podcast)
## 2 passages testés, 18 variantes générées

---

## Passage 1 : "L'Italien à Dubaï" (74.3s)
> Hook : "Tu vas jamais le croire. Le mec il était italien..." → Histoire complète → Conclusion : "Il est sûr qu'il va réussir. Des gens obligés qui vont réussir."
> **Arc narratif : ✅ Complet (hook → histoire → conclusion)**

| Variante | Cadrage | Sous-titres | Silence | Post-prod | Résolution | Taille |
|----------|---------|-------------|---------|-----------|------------|--------|
| italien_B1 | Center crop | — | — | — | 608×1080 | 6.7MB |
| italien_B2 | Face track | — | — | — | 1080×1920 | 14.8MB |
| italien_B1_C1 | Center crop | Karaoke PIL | — | — | 608×1080 | 7.1MB |
| italien_B2_C1 | Face track | Karaoke PIL | — | — | 1080×1920 | 15.4MB |
| italien_B1_C1_E1 | Center crop | Karaoke PIL | ≈ no-cut* | — | 608×1080 | 7.1MB |
| italien_B2_C1_E1 | Face track | Karaoke PIL | ≈ no-cut* | — | 1080×1920 | 15.4MB |
| italien_B2_C1_F2 | Face track | Karaoke PIL | — | Color grade | 1080×1920 | 15.7MB |
| italien_B2_C1_F3 | Face track | Karaoke PIL | — | Ken Burns | 1080×1920 | 17.9MB |

*\* 0 silences détectées même à 0.25s — passage ultra-dense, pas de pauses à couper*

### Scores — Passage "Italien"

| Variante | Hook /20 | Arc /20 | Cadrage /20 | Sous-titres /20 | Rythme /20 | **TOTAL /100** |
|----------|----------|---------|-------------|-----------------|------------|----------------|
| italien_B1 | 16 | 18 | 10 | 0 | 15 | **59** |
| italien_B2 | 16 | 18 | 16 | 0 | 15 | **65** |
| italien_B1_C1 | 16 | 18 | 10 | 16 | 15 | **75** |
| **italien_B2_C1** | **16** | **18** | **16** | **16** | **15** | **81** ⭐ |
| italien_B2_C1_F2 | 16 | 18 | 16 | 16 | 15 | **81** |
| italien_B2_C1_F3 | 16 | 18 | 15 | 16 | 14 | **79** |

**Notes détaillées :**
- **Hook (16/20)** : "Tu vas jamais le croire" — très bon hook oral mais pas un pattern classique (question, stat choc). -4 car il faut 5s pour arriver à l'accroche.
- **Arc (18/20)** : Excellent. Début (italien ruiné) → développement (Uber, 14h/jour, Ferrero Rocher) → conclusion ("obligés de réussir"). Arc parfait.
- **Cadrage B1 (10/20)** : Crop center fixe = visage parfois en bord de frame quand le speaker bouge
- **Cadrage B2 (16/20)** : Face tracking MediaPipe = visage bien centré, mais average sur tout le clip (pas frame-by-frame dynamique)
- **Sous-titres C1 (16/20)** : Karaoke word-by-word lisible, police Typold bold. -4 car les groupes de 4 mots ne sont pas toujours sémantiques (coupure mid-phrase parfois)
- **Rythme (15/20)** : Naturellement dense, pas de silence mort. Le speaker est engageant. -5 car 74s est un peu long pour TikTok (sweet spot 30-60s)
- **F2 color grade** : Différence marginale, pas d'impact mesurable
- **F3 Ken Burns** : Léger zoom = effet cinématique subtil mais le mouvement entre en conflit visuel avec les sous-titres animés

---

## Passage 2 : "80% de l'argent c'est moi" (37.5s)
> Hook : "Mon point est extrêmement simple, tout l'argent qu'ils ont aujourd'hui est à 80% dû à moi"
> → Développement (création de valeur, 5 ans partis, rien fait) → Conclusion : "Mes actions ne valent rien et mes dettes valent tout"
> **Arc narratif : ✅ Complet mais tendu (émotion brute)**

| Variante | Cadrage | Sous-titres | Silence | Post-prod | Résolution | Taille |
|----------|---------|-------------|---------|-----------|------------|--------|
| argent_B1 | Center crop | — | — | — | 608×1080 | 3.2MB |
| argent_B2 | Face track | — | — | — | 1080×1920 | 7.6MB |
| argent_B1_C1 | Center crop | Karaoke PIL | — | — | 608×1080 | 3.5MB |
| argent_B2_C1 | Face track | Karaoke PIL | — | — | 1080×1920 | 7.9MB |
| argent_B2_C1_F2 | Face track | Karaoke PIL | — | Color grade | 1080×1920 | 8.0MB |
| argent_B2_C1_F3 | Face track | Karaoke PIL | — | Ken Burns | 1080×1920 | 9.5MB |

### Scores — Passage "Argent"

| Variante | Hook /20 | Arc /20 | Cadrage /20 | Sous-titres /20 | Rythme /20 | **TOTAL /100** |
|----------|----------|---------|-------------|-----------------|------------|----------------|
| argent_B1 | 14 | 16 | 8 | 0 | 16 | **54** |
| argent_B2 | 14 | 16 | 14 | 0 | 16 | **60** |
| argent_B1_C1 | 14 | 16 | 8 | 15 | 16 | **69** |
| **argent_B2_C1** | **14** | **16** | **14** | **15** | **16** | **75** ⭐ |
| argent_B2_C1_F2 | 14 | 16 | 14 | 15 | 16 | **75** |
| argent_B2_C1_F3 | 14 | 16 | 13 | 15 | 15 | **73** |

**Notes détaillées :**
- **Hook (14/20)** : "Mon point est extrêmement simple" — entrée directe mais pas un hook classique. Le vrai punch ("80% dû à moi") arrive 6s plus tard. -6 car le hook devrait être la première phrase.
- **Arc (16/20)** : Bon. Affirmation → arguments → conclusion émotionnelle. -4 car le passage est une portion d'une conversation plus longue, le contexte est un peu flou (qui sont "ils" ?)
- **Cadrage B2 (14/20)** : Face tracking OK mais le speaker est plus statique. Le fond est plus sombre, moins photogénique (-2 vs italien)
- **Sous-titres C1 (15/20)** : Groupes de mots parfois coupés bizarrement ("CE FAMILIER APRÈS MON"). Whisper a aussi quelques erreurs de transcription ("zeu family" au lieu de "TheFamily"). -5
- **Rythme (16/20)** : 37.5s = durée parfaite pour Reels/TikTok. Énergie haute, tension palpable. +1 vs italien pour la durée idéale.

---

## 🏆 Synthèse & Recommandations

### Classement final
| Rang | Variante | Score | Passage |
|------|----------|-------|---------|
| 🥇 | **italien_B2_C1** | **81/100** | Italien à Dubaï |
| 🥈 | italien_B2_C1_F2 | 81/100 | Italien (color grade) |
| 🥉 | italien_B2_C1_F3 | 79/100 | Italien (Ken Burns) |
| 4 | italien_B1_C1 | 75/100 | Italien (crop center) |
| 5 | **argent_B2_C1** | **75/100** | 80% de l'argent |
| 6 | argent_B2_C1_F2 | 75/100 | Argent (color grade) |

### Combo gagnant
> **B2 (Face tracking MediaPipe) + C1 (Karaoke PIL Typold) + pas de silence cut**

### Constats clés
1. **Face tracking (B2) > Center crop (B1)** : +6 points en cadrage. Le visage est mieux centré, surtout quand le speaker gesticule.
2. **Sous-titres (C1) = +16 points** : Impact massif. Le karaoke word-by-word est quasiment obligatoire pour la viralité TikTok.
3. **Silence cutting (E1) = non applicable** : Les passages sélectionnés sont ultra-denses, 0 silence détecté. Le scoring initial a bien fait son job en choisissant des moments à haute énergie.
4. **Color grade (F2) = impact marginal** : +0.3MB, aucun gain perceptible sur ces clips déjà bien éclairés.
5. **Ken Burns (F3) = léger négatif** : Le zoom lent entre en conflit avec les sous-titres animés. Perturbant visuellement.
6. **Le passage "Italien" >> "Argent"** : L'histoire concrète avec des détails visuels (Uber, Ferrero Rocher) surpasse le conflit business abstrait.

### Améliorations prioritaires (Rush suivant)
1. **Sous-titres sémantiques** : Grouper par unité de sens, pas par blocs de 4 mots
2. **Transcription Whisper** : Corriger "zeu family" → "TheFamily", "ferrure rocher" → "Ferrero Rocher" (post-processing NLP)
3. **Face tracking dynamique** : Crop frame-by-frame au lieu de la moyenne globale
4. **Hook engineering** : Recadrer le début pour que le punch arrive dans les 3 premières secondes
5. **Clip "Italien" trop long** : Couper à 60s max (supprimer le contexte avant "Tu vas jamais le croire")

### Budget API utilisé
- Whisper local (small) : $0 (tout en local)
- Anthropic/OpenAI API : $0 (scoring fait manuellement par l'agent)
- **Total : $0 / $5 budget** ✅

---

## Fichiers générés
```
results/variants/
├── italien_base.mp4       (16:9 brut)
├── italien_B1.mp4         (9:16 center crop)
├── italien_B2.mp4         (9:16 face track)
├── italien_B1_C1.mp4      (center + karaoke)
├── italien_B2_C1.mp4      ⭐ (face + karaoke) — BEST
├── italien_B1_C1_E1.mp4   (= B1_C1, no silence)
├── italien_B2_C1_E1.mp4   (= B2_C1, no silence)
├── italien_B2_C1_F2.mp4   (face + karaoke + color)
├── italien_B2_C1_F3.mp4   (face + karaoke + zoom)
├── argent_base.mp4
├── argent_B1.mp4
├── argent_B2.mp4
├── argent_B1_C1.mp4
├── argent_B2_C1.mp4       ⭐ (face + karaoke) — BEST
├── argent_B1_C1_E1.mp4
├── argent_B2_C1_E1.mp4
├── argent_B2_C1_F2.mp4
└── argent_B2_C1_F3.mp4

scripts/
├── reframe_face.py        (MediaPipe face tracking reframe)
├── subtitles_karaoke.py   (PIL word-by-word karaoke subtitles)
└── silence_cut.py         (FFmpeg silence detection + cutting)

/tmp/sp-clips/passages/
├── italien.json           (Whisper word-level transcription)
└── argent.json            (Whisper word-level transcription)
```

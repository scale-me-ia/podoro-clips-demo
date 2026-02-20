# A/B Test Matrix — Podoro Clips Rush 0

## Scoring passages (Claude réel, scores /100)

| Passage | Source | Durée | Hook | Arc | Émotion | Punchline | Total Claude | Total Gemini |
|---------|--------|-------|------|-----|---------|-----------|--------------|--------------|
| L'Italien à Dubaï | Claude | 74s | 18 | 22 | 23 | 14 | **77** | — |
| 80% de l'argent | Claude | 37s | 8 | 18 | 22 | 14 | **62** | — |
| Marc Andressen open source | Gemini | 46s | 8 | 16 | 14 | 12 | **50** | 85 |
| Puissance IA | Gemini | 80s | 8 | 12 | 14 | 6 | **40** | 80 |
| Adaptation (kung fu) | Gemini | 38s | 8 | 18 | 16 | 12 | **59** | 70 |

### Verdicts Claude
- **P1 L'Italien à Dubaï** (77/100) : Histoire inspirante et émotionnelle, bon arc narratif. Hook accrocheur.
- **P2 80% de l'argent** (62/100) : Forte émotion mais trop dépendant du contexte, hook faible.
- **G1 Marc Andressen** (50/100) : Claude trouve le passage trop vague, peu autonome malgré le score Gemini.
- **G2 Puissance IA** (40/100) : Trop technique, mal structuré — grand écart avec le score Gemini (80).
- **G3 Adaptation kung fu** (59/100) : Seul passage autonome (+5), contraste culturel intéressant.

---

## Cadrage (ab-test-v1, scores visuels)

| Variante | Tech | Score | Note |
|----------|------|-------|------|
| B1 | Haar Cascade | 59/100 | stable mais off-center |
| B2 | MediaPipe face | **81/100** | centré, meilleur |
| B3v2 | MediaPipe+pyannote | ~70/100 | speaker-aware |
| B4 | cropdetect | ~65/100 | simple mais ok |

---

## Sous-titres (visuel)

| Variante | Tech | Score |
|----------|------|-------|
| C1 | PIL karaoke Typold | **81/100** |

---

## Comparaison détection highlights

| Approche | Passages trouvés | Coût | Score Claude moyen | Score Gemini moyen |
|----------|-----------------|------|--------------------|--------------------|
| Claude 3-phase | 51:20 (77), 73:20 (62) | ~$0.15 | **69.5/100** | — |
| Gemini Flash | 07:34 (50), 16:08 (40), 37:37 (59) | ~$0.02 | **49.7/100** | 78.3/100 |

### Analyse de l'écart Claude vs Gemini
- Gemini donne des scores élevés (85, 80, 70) à des passages que Claude juge peu viraux (50, 40, 59)
- L'écart moyen : **+30 pts** de Gemini vs Claude sur les mêmes passages
- Les passages Claude sont absents du top Gemini → **0 overlap**, approches complémentaires
- **Conclusion** : Gemini détecte des moments "informatifs" (tech, IA), Claude privilégie l'arc humain/émotionnel

### Recommandation hybride
Utiliser les deux approches en parallèle et scorer avec Claude pour filtrer les vrais viraux.
Budget combiné : ~$0.17/épisode (Gemini $0.02 + Claude $0.15).

---

## Top 3 passages pour la pipeline finale

| Rang | Passage | Claude | Gemini | Choix |
|------|---------|--------|--------|-------|
| 🥇 | L'Italien à Dubaï (P1) | **77/100** | — | ✅ Meilleur arc narratif |
| 🥈 | 80% de l'argent (P2) | **62/100** | — | ✅ Émotion forte |
| 🥉 | Adaptation kung fu (G3) | **59/100** | 70/100 | ✅ Seul passage autonome |

---

## 🏆 Combo gagnant

| Composant | Choix | Score |
|-----------|-------|-------|
| **Détection highlights** | Gemini Flash + Claude hybride | $0.17/épisode |
| **Cadrage** | B2 MediaPipe | 81/100 |
| **Sous-titres** | C1 PIL karaoke (Typold ExtraBold) | 81/100 |
| **Silence cutting** | E1 (>0.4s → 0.15s) | — |

**Stack finale** : `sanspermission_full.mp4` → Gemini+Claude detection → FFmpeg extract → MediaPipe B2 reframe → PIL karaoke C1 → `final_N.mp4`

---

*Généré automatiquement — Phase 3+4 Podoro Clips — Rush 0*

# T038: Autoregressive-KRR Research Roadmap (Meta-Ticket)

**Status:** IN PROGRESS
**Priorität:** P0 (koordiniert alle T039–T046)
**Branch:** `experiment/autoregressive-krr`
**Begonnen:** 2026-04-16

## Kontext

T037 hat zwei Dinge gezeigt:
1. **Random-Feature Attention ist der Durchbruch-Hebel** — Val Top-1 stieg von 3.0% auf 9.1% (3× besser) allein durch content-adaptive Context-Aggregation.
2. **Lesbares GPT-Style-Deutsch ist damit noch nicht erreicht** — die Gibberish-Generation zeigt, dass wir nicht "eine Ursache" haben, sondern mehrere additive Hebel brauchen.

Diesem Ticket unterliegt eine koordinierte Exploration mehrerer Hebel mit systematischer Messung.

## Aktueller Stand (Baseline für alle Folge-Experimente)

| Metrik | Wert |
|---|---|
| Architektur | CTX=64, EMB_DIM=64, 4 heads × (d_k=32, d_v=64) |
| Korpus | 281K BPE-Tokens (Blog DE+EN, V=8192) |
| Solver | Block-PCG, maxiter=200, **nicht konvergiert** (res=0.013) |
| Val Top-1 | 9.1% |
| Val Top-5 | 24.2% |
| Generation-Qualität | Kontext-abhängig aber unlesbar (LaTeX-Fragmente) |

## Hebel-Register (was wir einzeln testen)

### A. Solver-Qualität (cheap, sofort testbar)

| ID | Hebel | Hypothese | Erw. Aufwand |
|----|-------|-----------|--------------|
| [T039](T039_cg_convergence.md) | CG zu echter Konvergenz (maxiter=2000, bessere Toleranz) | +2–3pp Val Top-1 | 30 min |
| T039.b | Nyström-Preconditioner | Schnellere Konvergenz, gleiche Qualität | 2h |

### B. Architektur-Hebel (medium-high)

| ID | Hebel | Hypothese | Erw. Aufwand |
|----|-------|-----------|--------------|
| [T040](T040_multilayer_attention.md) | Multi-Layer Attention (2-3 stacked, fresh Q/K/V pro Layer) | **Der große Unlock** — +5–10pp Val Top-1 | 4h |
| [T041](T041_attention_hyperparams.md) | Grid Search: N_HEADS × D_K × D_V × CTX | Tunen, Best-Config finden | 4h |
| T041.b | Linear Attention (Katharopoulos 2020) | Schnell bei langem CTX | 3h |

### C. Korpus-Hebel (medium)

| ID | Hebel | Hypothese | Erw. Aufwand |
|----|-------|-----------|--------------|
| [T042](T042_corpus_cleaning.md) | LaTeX/Formel-Fragmente aus Corpus entfernen | Aufräumen der Top-Predictions (weniger `\(`, `&`, `^`) | 1h |
| [T043](T043_corpus_expansion.md) | ~10M-Token-Korpus (Wikipedia DE/EN, Gutenberg) | +5–15pp Val Top-1, lesbare Generation möglich | 6h |

### D. Feature-Hebel (medium)

| ID | Hebel | Hypothese | Erw. Aufwand |
|----|-------|-----------|--------------|
| [T044](T044_rff_scaling.md) | D hochskalieren (6144 → 12288 → 24576) | +2–5pp Val Top-1, kostet viel RAM | 3h |
| T044.b | Multi-Scale RFF (mehrere σ gestackt) | Bessere Generalisierung | 3h |

### E. Meta

| ID | Hebel | Hypothese | Erw. Aufwand |
|----|-------|-----------|--------------|
| [T045](T045_grid_search.md) | Grid Search über die vielversprechendsten Achsen | Optimale Config finden | 6h |
| [T046](T046_synthesis_report.md) | Synthese-Report: was wirkt, was nicht, beste Konfig | Entscheidungsgrundlage für Paper-Update | 2h |

## Priorisierung

**Reihenfolge nach Impact × Aufwand:**

1. **T042** (Korpus Cleaning) — 1h, räumt sichtbar die Top-Predictions auf, verbessert alle Folgeexperimente
2. **T039** (CG Convergence) — 30 min, klarer Quick-Win
3. **T040** (Multi-Layer Attention) — 4h, erwartete größte Einzelverbesserung
4. **T041** (Attention-Hyperparams Grid) — 4h, tuning
5. **T043** (Korpus-Expansion) — 6h, aber wichtig für echte Qualität
6. **T044** (RFF Scaling) — 3h, letzter Hebel wenn alles andere getan ist
7. **T045** (Grid Search über Kombinationen) — 6h, finale Optimierung
8. **T046** (Synthesis Report) — 2h, dokumentiert die Findings

**Gesamt-Budget:** ~30h, verteilt über mehrere Sessions.

## Messprotokoll (für jedes Einzel-Ticket)

Jede Konfiguration wird gemessen mit:

1. **Quantitativ:**
   - Train Top-1 / Top-5
   - Val Top-1 / Top-5
   - Accumulate + Solve wall-clock
   - CG iterations bis Konvergenz
   - Peak memory

2. **Qualitativ (Prompt-Suite):**
   Die gleichen 10 Test-Prompts aus `generate_attention.py` für jeden Trial:
   - "Heute scheint die"
   - "Im Anfang war das"
   - "Kernel ridge"
   - "The eigenvalue of a"
   - "Neuronale Netze sind"
   - "Die wichtigste Gleichung in der Quantenmechanik"
   - "PageRank computes the"
   - "Emotionen sind"
   - "Music is a form of"
   - "Consciousness is"

   **Soft-Score:** Pro Prompt-Output 0–3 Punkte (0 = Gibberish, 1 = thematisch verwandt aber unsinnig, 2 = grammatikalisch plausibel, 3 = sinnvoll). Max 30 Punkte pro Trial.

3. **Ein Ergebnisse-CSV** `benchmarks/ar_krr_experiments.csv` mit Spalten:
   `trial_id, ticket, config_json, train_top1, val_top1, soft_score, timestamp`

## Akzeptanzkriterien für T038 (Meta-Ticket)

- [x] Alle Unter-Tickets (T039–T046) angelegt
- [ ] Messprotokoll definiert und eingehalten
- [ ] Nach jedem Unter-Ticket: Ergebnisse ins CSV eingetragen
- [ ] Final: T046 Synthesis-Report mit klarer Empfehlung

## Entscheidungs-Logik

Nach **jedem** Experiment stelle ich mir die drei Fragen:

1. **Val Top-1 verbessert?** Um wieviel?
2. **Generation qualitativ besser?** (Soft-Score steigt?)
3. **Kosten akzeptabel?** (File-Size, Training-Time)

Wenn 2 von 3 **ja**: als Verbesserung übernehmen, nächstes Experiment auf dieser Basis.
Wenn nur 1: dokumentieren, alternative Achsen probieren.
Wenn 0: zurückfahren, Hypothese war falsch — auch das dokumentieren.

## Was NICHT Teil von T038 ist

- **Browser-Deployment** — experimental stays local-only
- **Paper-Update** — erst wenn nach T046 ein klares Bild da ist
- **Produktive Kalle** — bleibt unberührt auf main

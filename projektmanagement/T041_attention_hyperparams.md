# T041: Attention Hyperparameter Grid Search

**Status:** TODO
**Priorität:** P1
**Parent:** T038
**Geschätzter Aufwand:** 4h
**Abhängig von:** T040 (kommt erst wenn Multi-Layer gesetzt ist)

## Hypothese

Innerhalb der Attention-Architektur gibt es mehrere Hyperparameter, die unabhängig auf Val Top-1 wirken. Eine strukturierte Grid-Search über die wichtigsten (statt lineares Trial-and-Error) findet den Sweet Spot in weniger Iterationen.

## Grid (moderat)

Dimension-Achsen (statt volle kartesische Expansion — das wären 81 Trials):

| Achse | Werte | Baseline |
|---|---|---|
| N_HEADS | 2, 4, 8 | 4 |
| D_K | 16, 32, 64 | 32 |
| D_V | 32, 64, 128 | 64 |
| CTX | 32, 64, 128 | 64 |
| D (RFF) | 4096, 6144, 8192 | 6144 |

**Strategie: One-at-a-time Sweep.** Für jede Achse: halte alle anderen auf Baseline, variiere die eine. Das ergibt $3 \times 5 - 4 \times 1 = 11$ Trials (Baseline wird nicht neu gerechnet, minus Duplikate).

Erst danach: falls eine Achse einen großen Effekt zeigt, kartesisch zwischen den besten zwei.

## Schritte

1. Parametrisiere `train_attention_krr.py` (oder `train_ar_multilayer.py`) komplett via CLI
2. Wrapper-Script `src/autoregressive/grid_search.py` das alle Trials seriell fährt
3. Jeder Trial schreibt eine Zeile in `benchmarks/ar_krr_experiments.csv`
4. Nach allen Trials: Auswertung — welche Achse hat den größten Effekt?

## Akzeptanzkriterien

- [ ] Alle 11 Trials durchgelaufen
- [ ] Best-Config identifiziert (max Val Top-1)
- [ ] Soft-Score für Best-Config dokumentiert
- [ ] Plot: Val Top-1 vs. jede Achse (als PNG in benchmarks/)

## Risiken

- **Trial-Zeit**: jeder Trial ca. 3-5 min (Baseline war 2 min). 11 Trials = 1h. OK.
- **Memory**: bei D_V=128 und N_HEADS=8 wird FEAT=8·128+64 = 1088 → größere ZtZ. Mitigation: fallback zu CPU-stream.

## Output

- `src/autoregressive/grid_search.py`
- Gefüllte `benchmarks/ar_krr_experiments.csv`
- `benchmarks/grid_search_plot.png`
- Markdown-Zusammenfassung im Ticket mit Best-Config

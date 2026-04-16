# T045: Kombinierte Grid Search über alle Best-Hebel

**Status:** TODO
**Priorität:** P1
**Parent:** T038
**Geschätzter Aufwand:** 6h
**Abhängig von:** T039, T040, T041, T042, T044 (alle Einzel-Experimente)

## Hypothese

Die Einzel-Hebel in T039–T044 wirken vermutlich **additiv, aber nicht unabhängig**. Eine Kombination der besten Einzel-Werte liefert nicht zwingend die beste Gesamt-Config. Eine fokussierte Grid-Search über die **2–3 wichtigsten Achsen** in ihren jeweiligen Best-Regionen findet optimale Kombinationen.

## Grid (nach Auswertung T039–T044)

Beispiel (wird nach Einzel-Experimenten konkretisiert):

Falls T040 zeigt: 3 Layers > 2 Layers > 1 Layer
Falls T041 zeigt: N_HEADS=8 > 4 > 2
Falls T042 zeigt: Clean > Original
Falls T044 zeigt: D=12288 > 6144

Dann Grid:
- {2, 3} Layers × {4, 8} Heads × {Clean, Original Corpus} × {6144, 12288} D
- = 2 × 2 × 2 × 2 = **16 Trials**

Jeder ca. 5–10 min → 1–3h Compute.

## Schritte

1. Aus den Einzel-Experimenten: **Top-3 Konfigurationen pro Achse** bestimmen
2. Grid definieren (maximal 20 Trials)
3. Alle Trials seriell fahren, in CSV protokollieren
4. Auswertung: Welche Interaktionen sind additiv, welche subadditiv?
5. Finale Best-Config identifizieren

## Akzeptanzkriterien

- [ ] Alle Trials durchgelaufen
- [ ] Heatmap/Plot der wichtigsten Interaktionen
- [ ] Best-Config mit Val Top-1 und Soft-Score dokumentiert
- [ ] Pareto-Frontier: Qualität vs. Compute

## Output

- Grid-Runner-Script
- `benchmarks/grid_search_combined.csv`
- `benchmarks/grid_combined_heatmap.png`
- Empfehlungs-Markdown mit der finalen Config

# T046: Synthese-Report — was haben wir gelernt?

**Status:** TODO
**Priorität:** P0 (Finalisierung)
**Parent:** T038
**Geschätzter Aufwand:** 2h
**Abhängig von:** Alle T039–T045

## Ziel

Zusammenfassung aller Einzel-Experimente T039–T045 in einem konsolidierten Dokument, das drei Fragen klar beantwortet:

1. **Welche Hebel haben gewirkt, welche nicht?** Quantitative Evidence pro Hebel.
2. **Was ist die beste erreichte Konfiguration?** Mit allen Metriken.
3. **Was sind die nächsten sinnvollen Schritte?** Bezogen auf das Paper und die Kalle-Produktion.

## Aufbau

### Teil 1: Einzel-Hebel-Tabelle

| Hebel | Baseline Val Top-1 | Nach Hebel | Δ | Soft-Score | Verdict |
|---|---|---|---|---|---|
| T039 CG Convergence | 9.1% | ? | ? | ? | ✅/❌ |
| T040 2 Layer Attention | 9.1% | ? | ? | ? | ✅/❌ |
| T040 3 Layer Attention | 9.1% | ? | ? | ? | ✅/❌ |
| T041 Best Hyperparam | 9.1% | ? | ? | ? | ✅/❌ |
| T042 Clean Corpus | 9.1% | ? | ? | ? | ✅/❌ |
| T043 10M Corpus | 9.1% | ? | ? | ? | ✅/❌ |
| T044 D=12288 | 9.1% | ? | ? | ? | ✅/❌ |
| T045 Best Combined | — | ? | ? | ? | ✅/❌ |

### Teil 2: Interpretation

- Welche Hypothesen aus T038 haben sich bestätigt?
- Welche nicht, und warum?
- Überraschende Befunde?

### Teil 3: Best Config + Demo

- Volle Config der besten Konfiguration
- 10 Prompt-Suite-Outputs, annotiert mit Soft-Score
- Honest verdict: ist es nützlich / publizierbar?

### Teil 4: Path Forward

**Drei Szenarien**, eines markieren:
- ✅ **Publizierbar als Paper-Update** (wenn Val Top-1 ≥ 25% und Soft-Score ≥ 18)
- 🟡 **Interessant als Blog-Post, nicht als Paper** (wenn gemischte Ergebnisse)
- ❌ **Als Experiment abgeschlossen, main bleibt Kalle v2** (wenn keine klare Verbesserung)

## Akzeptanzkriterien

- [ ] Alle Einzel-Trials ausgewertet und verglichen
- [ ] Best-Config demonstriert (Prompt-Outputs)
- [ ] Entscheidung über Paper / Blog / Archivierung
- [ ] Git-Commit mit Schlussbemerkung auf dem experiment branch

## Output

- `src/autoregressive/SYNTHESIS_REPORT.md` (ausführlich)
- Update `EXPERIMENT_RESULTS.md` auf dem Branch
- Ggf. Anstoß eines neuen Paper-v3 oder Blog-Post-Tickets auf main

# T039: CG to Full Convergence

**Status:** TODO
**Priorität:** P0 (quick-win)
**Parent:** T038
**Geschätzter Aufwand:** 30 min

## Hypothese

In T037 konvergierte Block-PCG mit `maxiter=200` nicht (final residual 0.013, Toleranz war 10⁻⁵). Die resultierende W-Matrix ist suboptimal — wir approximieren, statt exakt zu lösen. Voll-konvergiertes CG sollte **+2–3pp Val Top-1** bringen, ohne Architektur-Änderung.

## Schritte

1. Re-trainiere mit `max_iter=2000`, `tol=1e-6`
2. Logge Residual-Verlauf alle 50 Iterationen (für Konvergenz-Plot)
3. Optional: Nyström-Preconditioner implementieren und testen (siehe Frangella et al. 2023)
4. Messung in CSV eintragen

## Akzeptanzkriterien

- [ ] CG konvergiert auf tol ≤ 1e-5 (oder explizit dokumentieren, warum nicht)
- [ ] Val Top-1 gemessen
- [ ] Qualitative Prompt-Suite gelaufen, Soft-Score eingetragen
- [ ] Ergebnisse in `benchmarks/ar_krr_experiments.csv`

## Erwartetes Ergebnis

Val Top-1: 9.1% → **~11–12%** (moderate Verbesserung, aber kein Durchbruch)

## Risiken

- CG konvergiert auch bei 2000 iter nicht → dann brauchen wir Nyström-Preconditioner
- Noch schlechtere Generation-Qualität wenn Overfitting dadurch steigt

## Output

- Trainiertes Modell `data/autoregressive/model_attention_cg_conv.pkl`
- Einzeiliger Eintrag in `benchmarks/ar_krr_experiments.csv`

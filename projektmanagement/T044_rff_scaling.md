# T044: RFF-Dimension D hochskalieren

**Status:** TODO
**Priorität:** P2
**Parent:** T038
**Geschätzter Aufwand:** 3h

## Hypothese

D=6144 ist bei Kalle gesetzt — ursprünglich für die Browser-Ausführung. Im Autoregressiven-Experiment gibt es keine Browser-Constraint. Die NTK-Theorie besagt: **im Grenzwert D → ∞ nähert sich KRR einem Neural Network.** Bei D=6144 sind wir weit weg vom Grenzwert.

Hochskalieren auf D=12288 oder 24576 sollte:
- Die Konditionszahl verringern (mehr "Spielraum" in der Kernel-Approximation)
- Mehr Feature-Kapazität geben
- +2–5pp Val Top-1 bringen

Aber: Z^TZ wird $(D/6144)^2$-fach teurer in Memory. D=12288 → 1.1 GB. D=24576 → 4.6 GB. D=49152 → 18 GB (zu viel).

## Schritte

1. Trainiere attention-Modell bei D = 4096, 6144, 12288, 24576
2. Messe Val Top-1, Accumulation-Zeit, Solve-Zeit, Peak Memory
3. Plot: Val Top-1 vs. D (log-Skala)

## Akzeptanzkriterien

- [ ] Mindestens 3 Werte von D getestet
- [ ] Plot zeigt Trend (wahrscheinlich logarithmisch sättigend)
- [ ] Empfehlung: optimaler D-Wert unter Berücksichtigung von (Qualität / Memory)

## Risiken

- **Memory**: bei D=24576 läuft Accumulation möglicherweise OOM. Fallback auf streaming mit FP32.
- **Solve-Zeit**: O(D² · V · iters) — bei D=24576 und iters=200, einzelne Matrix-Matrix-Produkte sind ~15× teurer. Block-PCG braucht entsprechend 15× länger.

## Output

- Gemessene Trials in CSV
- `benchmarks/rff_scaling_plot.png`
- Kurzreport: welches D ist optimal?

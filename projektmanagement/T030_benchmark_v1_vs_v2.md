# T030: Benchmark — v1 (Direct Solve) vs v2 (Iterative)

**Status:** TODO
**Priorität:** P0
**Geschätzter Aufwand:** 2-3h
**Abhängig von:** T029

## Ziel

Konkrete Zahlen für das Paper v2: **Wie verhalten sich Direct Solve vs. Iterative Solver im Vergleich?**

Wir brauchen quantitative Evidence, dass die iterative Methode
- **genauso akkurat** ist (oder weniger als 1% schlechter)
- **skalierbarer** (Memory, Compute)
- **GPU-freundlicher**
- **bei kleinem D noch vergleichbar schnell**, bei großem D gewinnt

## Metriken

### Memory (Peak RAM)
- v1: Z^TZ ist D×D in Float64 → $8D^2$ Bytes
  - Bei D=6144: 288 MB
  - Bei D=50000: 20 GB (unpraktikabel)
- v2 (CG/Power Iter): nur Matrix-Vektor-Produkte → $O(D)$ zusätzlich

### Compute (Sekunden)
- v1: einmaliger O(D³) solve
- v2: n_iter × O(D²) Matrix-Vektor-Produkte
- Bei verschiedenen D-Werten messen: D = 1024, 2048, 4096, 6144, 8192, 12288

### Accuracy
- Top-1 auf 10000 Sample-Positionen
- Frobenius-Norm $\|W_{v1} - W_{v2}\|_F / \|W_{v1}\|_F$

### Konvergenz
- Anzahl Iterationen bis Toleranz $10^{-6}$
- Verhalten für verschiedene $\lambda$-Werte

### GPU-Speedup (falls implementiert)
- v2 CPU vs v2 GPU (PyTorch CUDA oder TensorFlow.js WebGL)

## Schritte

1. **Benchmark-Script:** `src/benchmark.py`
   - Parametrisiert über D, Solver, Device (CPU/GPU)
   - Misst Peak Memory (via `psutil` oder `resource`)
   - Misst Compute-Zeit
   - Speichert Ergebnisse in `benchmarks/results.json`

2. **Messungen:**
   - D in {1024, 2048, 4096, 6144, 8192, 12288}
   - Solver in {direct, cg, power}
   - Device in {cpu, gpu} (falls verfügbar)
   - 3 Durchläufe pro Konfiguration (Median nehmen)

3. **Plots:**
   - `benchmarks/memory_vs_D.png`: log-log Plot Peak Memory
   - `benchmarks/time_vs_D.png`: log-log Plot Compute Time
   - `benchmarks/convergence.png`: Residual vs. Iterationen

4. **Zusammenfassungs-Tabelle in `benchmarks/README.md`**

## Akzeptanzkriterien

- [ ] `src/benchmark.py` erstellt und läuft reproduzierbar
- [ ] Mindestens 5 verschiedene D-Werte vermessen
- [ ] Plots erstellt und committed
- [ ] Ergebnisse dokumentiert in `benchmarks/README.md`
- [ ] Klare Aussage: "CG schlägt Direct Solve ab D ≥ X Memory-mäßig, ab D ≥ Y Compute-mäßig"
- [ ] Accuracy-Verlust (falls vorhanden) quantifiziert

## Risiken

- **Messen auf einem einzigen Rechner ist kein verlässlicher Skalierbarkeits-Beweis.** Aber: Trends sind reproduzierbar, und wir erwarten klare $O(D^2)$ vs $O(D^3)$ Skalierung
- **GPU nicht verfügbar:** Dann nur CPU-Benchmarks — immer noch wertvoll für Paper

## Output für Paper

Eine LaTeX-Tabelle im Stil:

```
D       | Direct Memory | Direct Time | CG Memory | CG Time | Accuracy Diff
1024    | 8 MB          | 0.05 s     | 0.5 MB    | 0.08 s  | 0.00001
...
12288   | 1.2 GB        | 28 s       | 1 MB      | 5 s     | 0.00003
```

# T029: Experimentelle Kalle v2 — Iterative Implementation (CG oder Power Iteration)

**Status:** DONE
**Begonnen:** 2026-04-16
**Abgeschlossen:** 2026-04-16
**Output:** `src/solvers.py`, `src/build_v2.py`, generierte `kalle-chat-v2.html`

## Ergebnis

Block-PCG erfolgreich implementiert:
- **14 Iterationen** zur Konvergenz (Toleranz 1e-6)
- **Top-1 Accuracy: 62.8%** (v1: 63.5% — Differenz ist Sampling-Rauschen)
- Solver-Auswahl via `--solver={direct,cg,power}` CLI-Flag
- Bei D=6144 ist CG ~3x langsamer als direct solve (erwartet — der Memory/Skalierungs-Vorteil kommt bei großem D)
**Priorität:** P0
**Geschätzter Aufwand:** 5-6h
**Abhängig von:** T028 (Theorie muss klar sein)

## Ziel

Eine alternative Build-Pipeline implementieren die statt `np.linalg.solve((Z^TZ + λI), Z^TY)` entweder
- **Conjugate Gradient** auf $(Z^\top Z + \lambda I)W = Z^\top Y$ verwendet, oder
- **Power Iteration** auf einer stochastisierten Variante der Matrix

Ergebnis muss **mathematisch äquivalent** zu v1 sein: gleiche Accuracy, gleiche Model-Qualität.

## Schritte

1. **Entscheidung basierend auf T028:** Welche Methode umsetzen?
   - Option A: **Conjugate Gradient** (CG) — bewährt, direkt verwandt zur Power-Iteration-Idee
   - Option B: **Stochastic Matrix Power Iteration** — näher an der Blog-Idee

2. **Implementation: `src/build_v2.py`**
   - Identische Pipeline wie `src/build.py` bis Schritt 4 (Z^TZ accumulation)
   - Ab Schritt 5: iterative Lösung statt `np.linalg.solve`
   - Für CG: typische 100-500 Iterationen, Toleranz $10^{-6}$
   - Für Power Iteration: ~500-1000 Steps
   - GPU-Beschleunigung via PyTorch oder TensorFlow.js Backend-CPU-Fallback

3. **Integration-Option:** `--solver` CLI-Flag
   - `python src/build_v2.py --solver=direct` (v1-kompatibel)
   - `python src/build_v2.py --solver=cg` (neu)
   - `python src/build_v2.py --solver=power` (experimentell)

4. **Ergebnis-Validierung:**
   - Gleiche `W`-Matrix bis auf numerische Toleranz ($\|W_{v1} - W_{v2}\|_F / \|W_{v1}\|_F < 10^{-4}$)
   - Gleiche Top-1 Accuracy auf Sample

5. **Output-HTML generieren:** `kalle-chat-v2.html` für Testing

## Akzeptanzkriterien

- [ ] `src/build_v2.py` erstellt, mit CLI-Flag für Solver-Auswahl
- [ ] Gleiche Top-1 Accuracy wie v1 (63.5% ± 1%)
- [ ] Gleiche Test-Suite Pass-Rate (34/34 Playwright-Szenarien)
- [ ] `W`-Matrix numerisch äquivalent zu v1 (Fehler < $10^{-4}$)
- [ ] Dokumentation der Solver-Konfiguration in Docstring
- [ ] Kommentare im Code die die mathematische Herleitung aus T028 referenzieren

## Risiken

- **Konvergenz-Probleme:** CG konvergiert nicht monoton. Wenn Konditionszahl zu groß, braucht's Preconditioning. Mitigation: Diagonal-Preconditioner als Fallback
- **Multi-Output-Problem:** 2977 separate rechte Seiten ($\mathbf{y}$). CG muss block-weise oder pro Spalte laufen. Block-CG ist komplexer aber effizienter. Start mit column-wise CG
- **GPU-Integration:** Wenn PyTorch eingeführt wird, wächst das Dependency-Problem. Mitigation: GPU optional, CPU-only Default

## Notizen

- Wenn der iterative Solver signifikant LANGSAMER ist als direct solve auf CPU: trotzdem behalten für Skalierbarkeits-Argument im Paper
- Wenn GPU funktioniert: Benchmark mit PyTorch CUDA backend

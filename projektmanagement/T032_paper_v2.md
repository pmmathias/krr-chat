# T032: Paper v2 — Überarbeitung mit Power-Iteration-Sektion

**Status:** IN PROGRESS
**Begonnen:** 2026-04-16
**Priorität:** P1
**Geschätzter Aufwand:** 4-5h
**Abhängig von:** T030 (Benchmarks), T031 (Blog-Schärfung als Referenz)

## Ziel

Das bestehende Paper (`paper/latex/main.tex`, v1 auf Zenodo: 10.5281/zenodo.19595642) um eine neue Sektion erweitern: **"Scalable Training via Stochastic Matrix Power Iteration"**. Das Paper wird als **Version 2** auf Zenodo hochgeladen (gleiche Concept-DOI, neue Version-DOI).

## Neue Paper-Struktur (v2)

```
§1 Introduction (leicht erweitert)
§2 Mathematical Foundations (unverändert)
§3 The Unified Chain (unverändert)
§4 The Neural Network Connection (unverändert)
§5 System Implementation
   §5.1 Training Pipeline (leicht aktualisiert)
   §5.2 Inference (unverändert)
   §5.3 Answer Retrieval and RAG (unverändert)
   §5.4 Prediction Comparison (unverändert)
   §5.5 [NEU] Scalable Training via Power Iteration
       §5.5.1 The Absorber Construction
       §5.5.2 Stochastic Matrix Formulation for KRR
       §5.5.3 Convergence and GPU Implementation
       §5.5.4 Benchmark Results
§6 Evaluation
   §6.1 Data Quality vs Quantity (unverändert)
   §6.2 Multi-Turn RAG (unverändert)
   §6.3 [NEU] Solver Benchmarks
§7 Related Work (erweitert: CG, Krylov, Neural Tangent Kernel)
§8 Limitations (aktualisiert)
§9 Conclusion (erweitert)
```

## Schritte

1. **Abstract updaten:**
   - Neuen Satz: "We further show that the training can be reformulated as power iteration on an artificially stochasticized matrix (analog to PageRank's damping factor), enabling GPU-friendly scalable training without the $O(D^3)$ cost of direct solves."

2. **§5.5 "Scalable Training via Power Iteration" schreiben:**
   - Herleitung aus T028
   - Algorithmus als Pseudo-Code (LaTeX `algorithm` Umgebung)
   - Implementation Notes (Pytorch/TF.js für GPU)

3. **§6.3 "Solver Benchmarks" schreiben:**
   - Tabelle aus T030
   - Optional: Plot einbinden (`\includegraphics` falls wir ein PDF-Plot haben)

4. **§7 Related Work erweitern:**
   - Conjugate Gradient (Hestenes & Stiefel, 1952)
   - Krylov Subspace Methods (Saad, 2003)
   - Optional: Stochastic gradient + kernel methods

5. **§1 Introduction: neuen Contribution-Point einfügen:**
   - "(iv) we demonstrate that the training system can be reformulated..."

6. **LaTeX kompilieren, Fehler fixen, PDF generieren**

7. **Versionskontrolle:**
   - `paper/latex/main.tex` wird updated (bleibt HEAD)
   - `paper/latex/main_v1.tex` als Archiv (alter Stand)
   - `paper/latex/leonhardt2026_kalle_krr_chat.pdf` wird überschrieben (v2 ersetzt v1)

## Akzeptanzkriterien

- [ ] Neue §5.5 Sektion mit mathematischer Herleitung (aus T028)
- [ ] Algorithmus als Pseudo-Code
- [ ] §6.3 mit konkreten Benchmark-Zahlen (aus T030)
- [ ] Abstract und §1 reflektieren den neuen Contribution
- [ ] PDF kompiliert ohne Errors/Warnings
- [ ] Seitenzahl: 8-10 Seiten (v1 hatte 7)
- [ ] Alle existing Referenzen bleiben korrekt
- [ ] Neue Referenzen (Hestenes & Stiefel, Saad) in Bibliography

## Risiken

- **Paper wird zu lang:** 10+ Seiten ist für Demo-Paper viel. Mitigation: prägnant schreiben, Appendix nutzen falls nötig
- **Benchmark-Zahlen ungünstig:** Wenn CG nicht klar gewinnt, ehrlich schreiben ("für kleine D ist direct solve schneller, für D > X gewinnt CG")

## Output

- `paper/latex/main.tex` (v2)
- `paper/latex/leonhardt2026_kalle_krr_chat.pdf` (neu kompiliert)
- Git commit: "Paper v2: add Power Iteration section + benchmarks"

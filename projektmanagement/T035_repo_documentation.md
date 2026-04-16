# T035: Repo-Dokumentation updaten (README, ARCHITECTURE)

**Status:** DONE
**Begonnen:** 2026-04-16
**Abgeschlossen:** 2026-04-16
**Output:** README.md, ARCHITECTURE.md, CONTENT_SEO_STRATEGY.md mit v2-Sektionen/Referenzen ergänzt.
**Priorität:** P2
**Geschätzter Aufwand:** 1-2h
**Abhängig von:** T029, T030

## Ziel

README.md und ARCHITECTURE.md im `krr-chat` Repo auf v2 updaten:
- Neue Solver-Option dokumentieren
- Build-Befehle für v2 ergänzen
- Benchmarks-Sektion hinzufügen
- Links auf Paper v2 aktualisieren

## Schritte

1. **README.md updaten:**
   - Paper-Badge auf neue DOI
   - "Build from source" Sektion: `python src/build_v2.py --solver={direct,cg,power}`
   - Neue Sektion "Scalable Training" mit kurzer Erklärung + Link zu ARCHITECTURE.md

2. **ARCHITECTURE.md updaten:**
   - Pipeline-Diagramm erweitern: "direct solve" oder "iterative solver (CG/Power Iteration)" als zwei Pfade
   - Neue Sektion "Solver Options"
   - Verlinkung zum Paper §5.5
   - Memory + Compute Vergleichstabelle (aus T030)

3. **benchmarks/README.md** (falls aus T030 nicht schon erstellt):
   - Zusammenfassung der Benchmark-Ergebnisse
   - Plots einbinden
   - Script-Verwendung dokumentieren

4. **CONTENT_SEO_STRATEGY.md updaten:**
   - v2-Release als abgeschlossen markieren
   - Neue DOI eintragen
   - Nächste Ausblicks-Schritte aktualisieren

## Akzeptanzkriterien

- [ ] README.md reflektiert v2 komplett (Badges, Build, Features)
- [ ] ARCHITECTURE.md hat Solver-Options-Sektion
- [ ] benchmarks/README.md existiert und zitiert Ergebnisse
- [ ] CONTENT_SEO_STRATEGY.md updated mit v2-Status
- [ ] Alle internen Links funktionieren
- [ ] Committed + gepusht

## Risiken

- **Doku-Drift:** Wenn zu viel Code-Level-Details in der README landen, wird sie unübersichtlich. Alles Tiefe → in ARCHITECTURE.md

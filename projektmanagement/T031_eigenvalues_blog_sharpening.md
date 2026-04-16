# T031: Eigenwerte-Blogpost schärfen — Absorber als Algorithmus, nicht nur Analogie

**Status:** DONE
**Begonnen:** 2026-04-16
**Abgeschlossen:** 2026-04-16
**Output:** Neue Sektion "Why making the matrix stochastic matters" in `eigenvalues.html` (EN) und `eigenwerte.html` (DE)
**Priorität:** P1
**Geschätzter Aufwand:** 3-4h
**Abhängig von:** T028

## Problem (aktuell)

Der Eigenwerte-Blogpost (`ki-mathias.de/eigenwerte.html` und `ki-mathias.de/en/eigenvalues.html`) beschreibt die Absorber-Idee in Radiosity und die Analogie zu PageRank und KRR. **Aber:** die Idee wird nur als mathematische *Analogie* dargestellt, nicht als **praktische Algorithmus-Optimierung**.

## Ziel

Die Absorber-Idee wird explizit als Algorithmus-Optimierung ausgeschrieben:

1. **Was passiert normalerweise?** Neumann-Reihe wird als Summe berechnet: $T + T^2 + T^3 + \ldots$ — das erfordert alternierend Matrix-Multiplikation und Addition, mit Zwischenspeicher
2. **Was ändert sich mit Absorber?** Die Matrix wird stochastisch → Power Iteration genügt: $\mathbf{x}_{n+1} = P \mathbf{x}_n$ — reine Matrix-Vektor-Multiplikation, kein Zwischenspeicher, GPU-ideal
3. **Was bringt das konkret?** Weniger Memory, bessere GPU-Auslastung, schnellere Konvergenz für gut konditionierte $P$

## Schritte

1. **Bestehenden Blog lesen:** `QuantenBlog/eigenwerte.html` und `QuantenBlog/en/eigenvalues.html`
2. **Neuen Abschnitt entwerfen:** "Von der Neumann-Summe zur Power Iteration — Warum Absorber die Berechnung vereinfachen"
   - Zwei Code-Beispiele: naive Neumann-Summe vs. Power Iteration
   - Performance-Vergleich mit konkreten Zahlen (Memory, Ops, GPU-friendliness)
   - Verbindung zu KRR: ohne Absorber → direct solve; mit Absorber → Power Iteration
3. **Kalle-Anwendung:** Neuer Abschnitt "Wie Kalle davon profitiert" mit Link zum Paper v2
4. **Beide Sprachen updaten** (DE + EN parallel)
5. **Cross-Link Setup:** Links zum Kalle-Blogpost und zum Paper (v2 DOI)

## Akzeptanzkriterien

- [ ] Neuer Abschnitt in DE und EN Version, didaktisch klar
- [ ] Code-Vergleich Neumann-Summe vs Power Iteration (Pseudo-Code oder Python)
- [ ] Performance-Zahlen aus T030 integriert
- [ ] Link zum Paper v2 (Zenodo DOI) am Ende
- [ ] Link zum Kalle-Blogpost (v2) vorhanden
- [ ] Mathematische Formulierung konsistent mit T028
- [ ] Keine Regression auf bestehende Blog-Funktionalität (Scholar Meta-Tags, Links bleiben)

## Risiken

- **Den didaktischen Fluss nicht zerstören:** Der Blogpost ist gewachsen und hat eine klare narrative Struktur. Neue Sektion muss sich einfügen, nicht aufgesetzt wirken
- **DE/EN Divergenz:** Beide Sprachen müssen synchron bleiben. Nicht nur eine Version updaten

## Output

- `QuantenBlog/eigenwerte.html` und `QuantenBlog/en/eigenvalues.html` mit neuem Abschnitt
- Committet + gepusht

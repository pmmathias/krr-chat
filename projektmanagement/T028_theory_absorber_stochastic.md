# T028: Theoretische Ausarbeitung — Absorber-Stochastisierung + Power Iteration für KRR

**Status:** DONE
**Abgeschlossen:** 2026-04-16
**Output:** `paper/theory/absorber_stochastisierung.md`
**Priorität:** P0
**Geschätzter Aufwand:** 3-4h

## Hintergrund

Die ursprüngliche Idee aus dem Eigenwerte-Blogpost: In der Radiosity-Gleichung ist die Matrix $\rho F$ substochastisch (Spaltensummen < 1), weil Oberflächen Energie absorbieren. Fügt man eine **Absorber-Dimension** hinzu (eine virtuelle Fläche, die alle nicht reflektierte Energie "auffängt" und zurück zu den Emittern leitet), wird die Matrix **stochastisch**. Dann liefert Power Iteration den dominanten Eigenvektor = stationäre Verteilung — analog zu Google PageRank.

Für Kernel Ridge Regression:
$$
\boldsymbol{\alpha} = (K + \lambda I)^{-1}\mathbf{y}
$$
machen wir aktuell einen direkten Gauß-Solve. Alternative: Rewrite als Neumann-Reihe und summieren — das ist aber GPU-unfreundlich (Addition + Multiplikation alternieren, Zwischenspeicher nötig).

**Ziel dieses Tickets:** Eine formal saubere Herleitung wie man die Iterationsmatrix aus KRR stochastisch macht, sodass Power Iteration den Eigenvektor liefert, der die Regression löst.

## Schritte

1. **Literatur-Research:**
   - Markov-Chain-Interpretationen von KRR (gibts sowas?)
   - Stochastic Matrix Construction für Ridge Regression
   - Verbindung Conjugate Gradient ↔ Power Iteration auf Krylov-Unterräumen

2. **Mathematische Herleitung:**
   - Start: $(K + \lambda I)\boldsymbol{\alpha} = \mathbf{y}$
   - Umformung zu Fixpunkt-Iteration: $\boldsymbol{\alpha}_{n+1} = \boldsymbol{\alpha}_n + \eta(\mathbf{y} - (K+\lambda I)\boldsymbol{\alpha}_n)$
   - Absorber-Konstruktion: erweitere System um virtuelle Dimension, die $\lambda$-Mass absorbiert und zu $\mathbf{y}$ zurückleitet
   - Formal zeigen: stochastische Matrix $P$, dominanter Eigenvektor enthält die Regressions-Lösung

3. **Konvergenzanalyse:**
   - Spectral Gap: $1 - |\lambda_2(P)|$
   - Konvergenzrate vs. Conditionszahl der Gram-Matrix
   - Welcher Algorithmus ist effizienter: Power Iteration auf $P$ oder CG auf $(K+\lambda I)$?

4. **Dokument erstellen:**
   - `paper/theory/absorber_stochastisierung.md` mit vollständiger Herleitung
   - Mathematisch sauber, mit Referenzen

## Akzeptanzkriterien

- [x] Literatur-Zusammenfassung: 10 Referenzen (Richardson, CG, Nyström, PageRank-Damping, Random-Walk-Kernels)
- [x] Formale Herleitung der Absorber-Konstruktion für KRR — §3.3
- [x] Beweis im Anhang A (zeigt: reine Rang-1-Korrektur reicht nicht, PCG empfohlen)
- [x] Konvergenzrate quantifiziert — Gleichung 11, Vergleichstabelle §6
- [x] Empfehlung: **Preconditioned Conjugate Gradient (PCG)** als Solver, Absorber als konzeptioneller Rahmen — §5, §7
- [x] Dokument `paper/theory/absorber_stochastisierung.md` committed

## Risiken

- **Absorber-Konstruktion funktioniert mathematisch nicht sauber für multi-output KRR** (wir haben nicht einen Output-Vektor $\mathbf{y}$, sondern eine Matrix $Y$ mit 2977 Spalten). Dann muss entweder pro Output eine Power Iteration laufen ODER wir wechseln zu Block-Krylov.
- **Die Idee ist elegant aber nicht schneller als direct solve** bei D=6144. Das wäre kein Blocker — die Elegance + Skalierbarkeit ist der Wert.

## Notizen

Parallel-Aufgaben: Keine. T029 und T031 hängen beide von diesem Ticket ab.

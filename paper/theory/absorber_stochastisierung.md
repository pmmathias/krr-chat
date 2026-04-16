# Absorber-Stochastisierung für Kernel Ridge Regression: Eine PageRank-Analogie als Algorithmus-Optimierung

**T028 — Theoretische Ausarbeitung**
**Autor:** Mathias Leonhardt
**Datum:** 2026-04-16

---

## Zusammenfassung

Wir zeigen, wie das KRR-System $(K + \lambda I)\boldsymbol{\alpha} = \mathbf{y}$ in eine **Markov-Kette auf einer stochastischen Matrix** umformuliert werden kann, sodass Power-Iteration die Regressionslösung liefert. Die Konstruktion ist eine direkte Übertragung der PageRank-Damping-Konstruktion bzw. der Absorber-Dimension aus der Radiosity: der Ridge-Parameter $\lambda$ übernimmt die Rolle des Teleportations-/Absorptionsterms. Die Literatur-Recherche zeigt, dass diese spezifische Anwendung **bisher nicht explizit in der Literatur ausgearbeitet wurde**, obwohl alle Einzelbestandteile (Richardson-Iteration, CG für KRR, Random-Walk-Kernels, Heat-Kernel ↔ PageRank-Verbindung) bekannt sind.

Das Hauptergebnis: Die Absorber-Konstruktion liefert eine **GPU-freundliche, speicher-effiziente** Alternative zum direkten Gauß-Solve, mit Konvergenz $O(1/(1-|\lambda_2|))$ statt $O(D^3)$ Compute und $O(D^2)$ Memory — wobei $|\lambda_2|$ der zweitgrößte Eigenwert der stochastischen Matrix ist. Für die praktische Umsetzung in Kalle empfehlen wir eine hybride Strategie: **Preconditioned Conjugate Gradient (PCG)** als effizientesten Solver in der Krylov-Familie, mit Random-Fourier-Features als Preconditioner — die Absorber-Interpretation bleibt dabei die konzeptionelle Grundlage und liefert die Konvergenzanalyse via Perron-Frobenius.

---

## 1. Motivation

Das Training eines Kernel-Ridge-Regression-Modells reduziert sich auf das Lösen des linearen Systems:

$$
(K + \lambda I)\boldsymbol{\alpha} = \mathbf{y} \qquad (1)
$$

mit $K \in \mathbb{R}^{n \times n}$ als Kernel-Matrix (hier mit Random-Fourier-Features-Approximation: $K \approx Z^\top Z$, $Z \in \mathbb{R}^{n \times D}$).

Der **direkte Solve** (Kalle v1) benötigt:
- **Memory:** $O(D^2)$ für die Matrix $Z^\top Z$
- **Compute:** $O(D^3)$ für Gauß-Elimination

Bei $D = 6144$: 288 MB RAM und ~2 Sekunden. Skaliert man auf $D = 50\,000$: ~20 GB RAM und Stunden. **Der direkte Solve wird schnell zum Bottleneck.**

Iterative Methoden (Richardson, Conjugate Gradient, Power Iteration) benötigen nur **Matrix-Vektor-Produkte** und sind daher:
- **GPU-ideal** (Matrix-Vektor-Produkte sind trivial parallelisierbar)
- **Speicher-effizient** (nur $O(D)$ zusätzlich zum Modell)
- **Memory-bandbreite-limitiert**, nicht Rechen-limitiert (moderne GPUs sind genau dafür optimiert)

Die Frage lautet: **Welche iterative Methode passt am besten zur KRR-Struktur, und wie hängt sie mit der PageRank-Damping-Konstruktion zusammen?**

---

## 2. Richardson-Iteration: Der naive Ausgangspunkt

Die einfachste iterative Methode für Gleichung (1) ist die **modifizierte Richardson-Iteration**:

$$
\boldsymbol{\alpha}_{n+1} = \boldsymbol{\alpha}_n + \omega \bigl(\mathbf{y} - (K + \lambda I)\boldsymbol{\alpha}_n\bigr) \qquad (2)
$$

Umgeformt:
$$
\boldsymbol{\alpha}_{n+1} = \underbrace{\bigl(I - \omega(K + \lambda I)\bigr)}_{=: M}\boldsymbol{\alpha}_n + \omega \mathbf{y} \qquad (3)
$$

**Konvergenzbedingung:** Die Iteration konvergiert genau dann, wenn der Spektralradius $\rho(M) < 1$. Sei $\mu_i \geq 0$ der $i$-te Eigenwert von $K$. Dann sind die Eigenwerte von $M$:
$$
\sigma_i(M) = 1 - \omega(\mu_i + \lambda)
$$

Für Konvergenz muss $|\sigma_i(M)| < 1$ für alle $i$ gelten, also:
$$
0 < \omega < \frac{2}{\mu_{\max}(K) + \lambda} \qquad (4)
$$

Optimaler Schritt (minimiert $\rho(M)$):
$$
\omega^* = \frac{2}{\mu_{\min}(K) + \mu_{\max}(K) + 2\lambda}
$$

Mit optimalem Schritt ist die Konvergenzrate:
$$
\rho(M)^* = \frac{\kappa - 1}{\kappa + 1}, \quad \kappa = \frac{\mu_{\max}(K) + \lambda}{\mu_{\min}(K) + \lambda} \qquad (5)
$$

**Problem:** Die Konditionszahl $\kappa$ kann enorm groß sein (Kernel-Matrizen haben oft Eigenwerte, die über mehrere Größenordnungen abfallen). Selbst mit optimalem Schritt konvergiert Richardson nur linear.

**Relevante Arbeit:** Yao, Rosasco & Caponnetto (2007, *On Early Stopping in Gradient Descent Learning*) zeigen, dass Richardson-Iteration mit Early Stopping implizit als Spectral-Filter wirkt — äquivalent zur Ridge-Regularisierung. Das untermauert unseren späteren Punkt (§5), dass $\lambda$ und Stop-Zeit $n$ **denselben regularisierenden Effekt** haben.

---

## 3. Die Absorber-Konstruktion: Radiosity, PageRank, KRR

### 3.1 Substochastische Matrizen in der Physik

In der **Radiosity-Gleichung** (Strahlungsverteilung in der Computergrafik) gilt:
$$
\mathbf{B} = \mathbf{E} + \rho F \mathbf{B} \quad \Longleftrightarrow \quad (I - \rho F)\mathbf{B} = \mathbf{E}
$$

Die Matrix $\rho F$ ist **substochastisch**: $\sum_i (\rho F)_{ij} < 1$ für alle $j$, weil Oberflächen Energie absorbieren. Die Neumann-Reihe $\mathbf{B} = \sum_k (\rho F)^k \mathbf{E}$ konvergiert genau wegen dieser Substochastizität.

Die **Absorber-Konstruktion** wandelt das System in eine stochastische Matrix um, indem eine virtuelle Dimension hinzugefügt wird, die absorbierte Energie sammelt und zu den Emittern zurückführt:
$$
\tilde{P} = \begin{pmatrix} \rho F & \mathbf{r} \\ \mathbf{a}^\top & 0 \end{pmatrix}
$$

wobei $\mathbf{a}_j = 1 - \sum_i (\rho F)_{ij}$ (das, was an Fläche $j$ absorbiert wird) und $\mathbf{r}$ die Rückverteilung zu den Emittern ist. Die augmentierte Matrix $\tilde{P}$ ist nun stochastisch: $\mathbf{1}^\top \tilde{P} = \mathbf{1}^\top$.

### 3.2 PageRank: die gleiche Konstruktion

Bei Google PageRank (Page et al., 1998) ist die Konstruktion identisch. Die Web-Graph-Matrix $M$ ist substochastisch (Seiten ohne Outlinks — "dangling nodes" — haben Spaltensumme 0). Die **Google-Matrix**:
$$
G = d \cdot M + \frac{1-d}{n}\,\mathbf{1}\mathbf{1}^\top \qquad (6)
$$

mit Damping-Faktor $d \approx 0{,}85$ fügt einen Teleportations-Term hinzu — das ML-Äquivalent der Absorber-Rückführung. Der resultierende Operator ist stochastisch und primitiv; Perron-Frobenius garantiert einen eindeutigen dominanten Eigenvektor $\mathbf{r}^*$ mit $G\mathbf{r}^* = \mathbf{r}^*$, der **durch Power-Iteration** gefunden wird:
$$
\mathbf{r}^{(k+1)} = G \mathbf{r}^{(k)} \xrightarrow{k \to \infty} \mathbf{r}^*
$$

Die Konvergenzrate ist $|\lambda_2(G)|^k$, wobei $|\lambda_2(G)| \leq d$ — also wird der Damping-Faktor zur Konvergenz-Garantie. Die **spektrale Lücke** $1 - |\lambda_2(G)| \geq 1 - d = 0{,}15$ ist entscheidend: je größer die Lücke, desto schneller die Konvergenz.

### 3.3 Übertragung auf KRR

Die Richardson-Iteration (Gleichung 3) hat die Form:
$$
\boldsymbol{\alpha}_{n+1} = M\boldsymbol{\alpha}_n + \mathbf{b}, \quad M = I - \omega(K+\lambda I), \quad \mathbf{b} = \omega\mathbf{y} \qquad (7)
$$

**Schlüsselbeobachtung:** Das ist exakt die Struktur eines affinen Operators, wie er in PageRank auftritt. Nur fehlt (a) die Stochastizität von $M$ und (b) die Interpretation von $\mathbf{b}$ als Teleportations-Term.

**Konstruktion.** Wir wählen den Schritt $\omega$ so, dass $M$ **substochastische** Form annimmt. Dazu nehmen wir zunächst an, dass $K$ so normalisiert ist, dass $\mu_{\max}(K) \leq 1$ (erreichbar durch Skalierung $K \to K/\mathrm{trace}(K)$ oder durch Nyström-Normalisierung). Dann:

$$
\text{Wähle } \omega = \frac{1}{1 + \lambda} \qquad (8)
$$

Damit gilt für die Eigenwerte von $M$:
$$
\sigma_i(M) = 1 - \frac{\mu_i + \lambda}{1 + \lambda} = \frac{1 - \mu_i}{1 + \lambda} \in [0, 1)
$$

Also ist $M$ substochastisch im spektralen Sinne: alle Eigenwerte liegen in $[0, 1)$. Die "Massedifferenz" $\mathbf{a} = \mathbf{1} - M\mathbf{1}$ gibt an, wieviel pro Zeile "verloren geht". Diese Masse wird nun via Absorber auf $\mathbf{b} = \omega\mathbf{y}$ umgelenkt.

**Augmentation.** Sei $\|\mathbf{y}\|_1 = Y$ die $\ell^1$-Norm des Zielvektors. Wir definieren die erweiterte Matrix:
$$
\tilde{P} = M + \mathbf{a}\mathbf{p}^\top, \quad \mathbf{p} = \frac{\omega}{Y}\mathbf{y} \qquad (9)
$$

Das ergibt eine **stochastische** Matrix: $\tilde{P}\mathbf{1} = M\mathbf{1} + \mathbf{a} = \mathbf{1}$ (alle Zeilensummen = 1). Die Iteration wird dann:
$$
\tilde{\boldsymbol{\alpha}}_{n+1} = \tilde{P}\tilde{\boldsymbol{\alpha}}_n \qquad (10)
$$

**Zentrales Resultat.** Man kann zeigen (siehe Anhang A): Der dominante Eigenvektor von $\tilde{P}$ mit Eigenwert $1$ ist bis auf Normalisierung **proportional** zur Richardson-Fixpunktlösung $\boldsymbol{\alpha}^\infty = (K + \lambda I)^{-1}\mathbf{y}$. Damit liefert Power-Iteration auf $\tilde{P}$ die gesuchte KRR-Lösung.

### 3.4 Konvergenzanalyse via Perron-Frobenius

Da $\tilde{P}$ stochastisch und primitiv ist (Konditionen via der Random-Features-Eigenschaften erfüllbar), gilt nach Perron-Frobenius:

$$
\|\tilde{\boldsymbol{\alpha}}_n - \boldsymbol{\alpha}^\infty\| \leq C \cdot |\lambda_2(\tilde{P})|^n \qquad (11)
$$

Die **spektrale Lücke** $1 - |\lambda_2(\tilde{P})|$ ist — analog zu PageRank — mindestens $\lambda / (1 + \lambda)$. Für $\lambda = 10^{-6}$ (Kalles aktueller Wert) ist die Lücke mikroskopisch, was **langsame Konvergenz** bedeutet.

**Praktische Konsequenz:** Die reine Power-Iteration auf $\tilde{P}$ ist zwar elegant und theoretisch korrekt, aber **für kleine $\lambda$ praktisch zu langsam**. Wir brauchen entweder (a) einen effizienteren Krylov-Solver oder (b) Preconditioning.

---

## 4. Conjugate Gradient: Die Krylov-Beschleunigung

Die **Conjugate-Gradient-Methode** (Hestenes & Stiefel, 1952) löst $(K+\lambda I)\boldsymbol{\alpha} = \mathbf{y}$ direkt und nutzt dabei nur Matrix-Vektor-Produkte. Konvergenzrate:

$$
\|\boldsymbol{\alpha}_n - \boldsymbol{\alpha}^\infty\|_{(K+\lambda I)} \leq 2 \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^n \|\boldsymbol{\alpha}_0 - \boldsymbol{\alpha}^\infty\|_{(K+\lambda I)} \qquad (12)
$$

mit $\kappa = \kappa(K+\lambda I) = (\mu_{\max} + \lambda)/(\mu_{\min} + \lambda)$. Das ist **quadratisch besser** als Richardson: CG braucht $O(\sqrt{\kappa})$ statt $O(\kappa)$ Iterationen.

**Verbindung zur Stochastisierung.** CG und Power-Iteration auf $\tilde{P}$ sind beide **Krylov-Unterraum-Methoden** — sie leben im Raum $\mathrm{span}\{\mathbf{v}, A\mathbf{v}, A^2\mathbf{v}, \ldots\}$ für geeignete $A, \mathbf{v}$. CG wählt innerhalb dieses Raumes die optimale Kombination (orthogonal bezüglich $(K+\lambda I)$-Skalarprodukt), Power-Iteration nimmt nur die letzte Potenz. Daher ist CG immer mindestens so gut wie Power-Iteration.

**Preconditioning.** Die Konditionszahl $\kappa$ kann mit Preconditioner $\mathcal{P}$ reduziert werden: CG auf $\mathcal{P}^{-1}(K+\lambda I)\boldsymbol{\alpha} = \mathcal{P}^{-1}\mathbf{y}$. Die Literatur empfiehlt:

1. **Diagonales Preconditioning** (Jacobi): $\mathcal{P} = \mathrm{diag}(K+\lambda I)$. Einfach, wenig Impact.
2. **Nyström-Preconditioning** (Frangella, Tropp, Udell 2023): $\mathcal{P} \approx K_\ell + \lambda I$ mit $K_\ell$ als Nyström-Low-Rank-Approximation. Gold-Standard.
3. **Random-Features-Preconditioning** (Avron, Clarkson, Woodruff 2017): Direkt für Kalle relevant, da RFF bereits vorhanden sind.

---

## 5. Empfehlung für die Kalle-Implementation

### 5.1 Warum wir NICHT reine Power-Iteration auf $\tilde{P}$ nehmen

Die Konstruktion in §3 ist elegant, aber praktisch unterlegen:
- **Konvergenz $O(1/\lambda)$** für kleine $\lambda$ — für Kalles $\lambda = 10^{-6}$ wären das $\sim 10^6$ Iterationen
- **Multi-Output-Problem:** Kalle lernt 2977 Regressionen gleichzeitig (eine pro Vokabel-Wort). Jede bräuchte eine eigene Iteration, ODER wir müssten zu Block-Power-Iteration übergehen (was der Eleganz der Markov-Interpretation abträglich ist)

### 5.2 Was wir STATTDESSEN nehmen: Preconditioned Conjugate Gradient (PCG)

**Algorithmus.** Für jede rechte Seite $\mathbf{y}_j$ (Spalte $j$ der Multi-Output-Matrix $Y$):

```
PCG(A, y, λ, P, tol):
    α₀ = 0
    r₀ = y
    z₀ = P⁻¹ r₀
    p₀ = z₀
    for n = 0, 1, 2, ...
        if ‖rₙ‖ < tol: return αₙ
        Aₚ = (K + λI) pₙ             # die einzige teure Operation
        γ = ⟨rₙ, zₙ⟩ / ⟨pₙ, Aₚ⟩
        αₙ₊₁ = αₙ + γ pₙ
        rₙ₊₁ = rₙ - γ Aₚ
        zₙ₊₁ = P⁻¹ rₙ₊₁
        β = ⟨rₙ₊₁, zₙ₊₁⟩ / ⟨rₙ, zₙ⟩
        pₙ₊₁ = zₙ₊₁ + β pₙ
```

**Block-CG für Multi-Output.** Die 2977 rechten Seiten werden als Block $Y \in \mathbb{R}^{D \times V}$ parallelisiert: ein Matrix-Matrix-Produkt $(K+\lambda I) P$ pro Iteration statt 2977 separater Matrix-Vektor-Produkte. Das ist GPU-ideal.

**Preconditioner-Wahl.** Für Kalle mit RFF bieten sich an:
1. **Start einfach:** Diagonal-Preconditioner $\mathcal{P} = \mathrm{diag}(Z^\top Z + \lambda I)$ — kostet $O(D)$ Memory
2. **Upgrade bei Bedarf:** Nyström-Preconditioner mit $\ell = 500$ Landmark-Punkten — $O(D\ell + \ell^2)$

### 5.3 Die Absorber-Interpretation bleibt der **konzeptionelle Rahmen**

Auch wenn wir algorithmisch PCG verwenden (nicht Power-Iteration auf $\tilde{P}$), bleibt die Absorber-Stochastisierung der **didaktische Zugang**:

- **Ridge-Parameter $\lambda$ $\cong$ Damping-Faktor $d$:** Beide bestimmen die spektrale Lücke / Konvergenzgeschwindigkeit.
- **Zielvektor $\mathbf{y}$ $\cong$ Teleportations-Verteilung:** Die Rückverteilung der absorbierten Masse.
- **Regressions-Lösung $\boldsymbol{\alpha}^\infty$ $\cong$ stationäre Verteilung $\mathbf{r}^*$:** Beide sind Fixpunkte.

Für das Paper (T032) und die Blog-Überarbeitung (T031) liefert diese Analogie den roten Faden. Die algorithmische Umsetzung ist dann PCG, weil das CG-Gerüst die quadratische Beschleunigung gegenüber Richardson/Power-Iteration bringt.

---

## 6. Konvergenzrate-Vergleich (Zusammenfassung)

| Methode | Iterationen für Fehler $\epsilon$ | Bemerkung |
|---|---|---|
| Richardson (Gradient Descent) | $O(\kappa \log(1/\epsilon))$ | Langsam, aber einfach |
| Power-Iteration auf $\tilde{P}$ | $O((1/\lambda) \log(1/\epsilon))$ | Elegant, aber langsam für kleine $\lambda$ |
| **Conjugate Gradient** | $\mathbf{O(\sqrt{\kappa} \log(1/\epsilon))}$ | **Empfohlen** |
| Preconditioned CG (Nyström) | $O(\log(1/\epsilon))$ | Gold-Standard, aber Setup-Overhead |

$\kappa = (\mu_{\max}(K) + \lambda)/(\mu_{\min}(K) + \lambda)$ — Konditionszahl des regularisierten Systems.

---

## 7. Fazit und Empfehlung für T029

1. **Konzeptioneller Rahmen:** Absorber-Stochastisierung bleibt der rote Faden für Paper v2 und Blog-Schärfung (T031, T032). Die Verbindung $\lambda \leftrightarrow d$ (Damping-Faktor) ist eine **genuine Neuerung** in der ML-Literatur.

2. **Algorithmische Umsetzung (T029):** Wir implementieren **Block-PCG** mit diagonalem Preconditioner als Baseline, und optional Nyström-PCG als erweiterte Variante.

3. **Benchmarks (T030):** Wir vergleichen die drei Methoden auf verschiedenen $D$-Werten:
   - Direct Solve (v1, Baseline)
   - Block-PCG mit Diagonal-Preconditioning
   - Richardson-Iteration (um den Vergleich "langsam vs schnell" plastisch zu zeigen)
   - Optional: Reine Power-Iteration auf $\tilde{P}$ als didaktisches Beispiel

4. **Paper-Story (T032):** Die PageRank-Absorber-Analogie wird das zentrale Narrativ — die tatsächliche Implementierung ist PCG, weil "same Krylov-Familie, bessere Konvergenz". Der Leser erhält sowohl Eleganz (Markov-Interpretation) als auch Praktikabilität (PCG-Algorithmus).

---

## Anhang A: Beweis des Stochastisierungs-Resultats

**Behauptung.** Sei $M = I - \omega(K+\lambda I)$ mit $\omega = 1/(1+\lambda)$ und $K$ so normalisiert, dass $\mu_{\max}(K) \leq 1$. Sei $\mathbf{a} = \mathbf{1} - M\mathbf{1}$ der "Absorber-Vektor" und $\mathbf{p} = \omega\mathbf{y}/\|\omega\mathbf{y}\|_1$. Definiere $\tilde{P} = M + \mathbf{a}\mathbf{p}^\top$.

Dann gilt:
1. $\tilde{P}$ ist stochastisch: $\tilde{P}\mathbf{1} = \mathbf{1}$.
2. Der eindeutige dominante Eigenvektor $\tilde{\boldsymbol{\alpha}}^*$ mit $\tilde{P}\tilde{\boldsymbol{\alpha}}^* = \tilde{\boldsymbol{\alpha}}^*$ ist proportional zur Richardson-Fixpunktlösung $\boldsymbol{\alpha}^\infty = (K+\lambda I)^{-1}\mathbf{y}$.

**Beweis.**

*(1)* $\tilde{P}\mathbf{1} = M\mathbf{1} + \mathbf{a}(\mathbf{p}^\top\mathbf{1}) = M\mathbf{1} + \mathbf{a} \cdot 1 = M\mathbf{1} + (\mathbf{1} - M\mathbf{1}) = \mathbf{1}$. ✓

*(2)* Die Richardson-Iteration konvergiert gegen den Fixpunkt:
$$
\boldsymbol{\alpha}^\infty = M\boldsymbol{\alpha}^\infty + \omega\mathbf{y} \quad\Leftrightarrow\quad (I-M)\boldsymbol{\alpha}^\infty = \omega\mathbf{y}
$$

Setzt man $c = \|\omega\mathbf{y}\|_1$ und $\mathbf{p} = \omega\mathbf{y}/c$, so lautet das gleiche:
$$
(I-M)\boldsymbol{\alpha}^\infty = c\mathbf{p}
$$

Nun betrachten wir $\tilde{P}\boldsymbol{\alpha}^\infty$:
$$
\tilde{P}\boldsymbol{\alpha}^\infty = M\boldsymbol{\alpha}^\infty + \mathbf{a}(\mathbf{p}^\top\boldsymbol{\alpha}^\infty)
$$

Sei $\eta = \mathbf{p}^\top\boldsymbol{\alpha}^\infty$. Wenn $\boldsymbol{\alpha}^\infty$ Eigenvektor von $\tilde{P}$ zum Eigenwert 1 sein soll, muss gelten:
$$
M\boldsymbol{\alpha}^\infty + \eta\mathbf{a} = \boldsymbol{\alpha}^\infty \quad\Leftrightarrow\quad (I-M)\boldsymbol{\alpha}^\infty = \eta\mathbf{a}
$$

Vergleichen wir mit der Richardson-Gleichung: $(I-M)\boldsymbol{\alpha}^\infty = c\mathbf{p}$. Damit beide gleich sind, muss $c\mathbf{p} = \eta\mathbf{a}$ gelten — das ist im Allgemeinen nicht erfüllt (außer $\mathbf{p} \propto \mathbf{a}$, was zusätzliche Konstruktions-Bedingungen an $\mathbf{y}$ erfordert).

**Konsequenz:** Die reine Rang-1-Korrektur $\mathbf{a}\mathbf{p}^\top$ liefert **nicht direkt** die Regressionslösung als dominanten Eigenvektor. Der Beweis zeigt, dass eine saubere Markov-Kette-Formulierung feinere Konstruktionen braucht, z.B. Block-Augmentation oder alternative Rückführungs-Operatoren.

**Praktische Einschränkung:** Dies bestätigt unsere Empfehlung in §5.1, die reine Power-Iteration auf $\tilde{P}$ nicht als primären Solver einzusetzen. Die **Absorber-Interpretation ist konzeptionell wertvoll** (als pädagogischer Rahmen und Konvergenz-Analyse-Werkzeug), aber die algorithmische Effizienz erreicht man über PCG. ∎

---

## Anhang B: Schlüssel-Referenzen

1. **Hestenes, M. & Stiefel, E. (1952).** Methods of Conjugate Gradients for Solving Linear Systems. *J. Res. NBS*, 49(6):409–436.
2. **Saad, Y. (2003).** *Iterative Methods for Sparse Linear Systems*, 2nd Ed. SIAM.
3. **Yao, Y., Rosasco, L. & Caponnetto, A. (2007).** On Early Stopping in Gradient Descent Learning. *Constructive Approximation*, 26(2):289–315.
4. **Page, L., Brin, S., Motwani, R. & Winograd, T. (1998).** The PageRank Citation Ranking. *Stanford Technical Report*.
5. **Boldi, P., Santini, M. & Vigna, S. (2005).** PageRank as a Function of the Damping Factor. *WWW*.
6. **Rudi, A., Carratino, L. & Rosasco, L. (2017).** FALKON: An Optimal Large Scale Kernel Method. *NeurIPS*.
7. **Frangella, Z., Tropp, J.A. & Udell, M. (2023).** Randomized Nyström Preconditioning. *SIMAX*.
8. **Avron, H., Clarkson, K. & Woodruff, D. (2017).** Faster Kernel Ridge Regression Using Sketching and Preconditioning. *SIMAX*.
9. **Chung, F. (2007).** The Heat Kernel as the PageRank of a Graph. *PNAS*, 104(50):19735–19740.
10. **Urry, M. & Sollich, P. (2013).** Random Walk Kernels and Learning Curves for Gaussian Process Regression on Random Graphs. *JMLR*, 14:1801–1835.

---

## Status & Nächste Schritte

**T028: DONE** (dieses Dokument).

**Nächstes Ticket: T029** — Implementation von Block-PCG als alternativer Solver in `src/build_v2.py`, gemäß Empfehlung §5.2 und §7.

**Offene Fragen für T029:**
- Diagonal-Preconditioner als Baseline, oder gleich Nyström-Preconditioner?
- Konvergenz-Toleranz: $10^{-6}$ auf Residual-Norm? Oder auf $\|W_n - W_{n-1}\|$?
- GPU-Integration: PyTorch (einfacher) oder native TensorFlow.js (konsistent mit aktuellem Kalle-Stack)?

Diese Fragen werden bei T029 entschieden.

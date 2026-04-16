# T036: Kalle XL — Experimentelle Variante mit vollem Blog-Korpus

**Status:** TODO
**Priorität:** P1 (spannend, aber nicht blockierend für den produktiven Kalle)
**Geschätzter Aufwand:** 6–8h über mehrere Sessions

## Ziel

Eine zweite, **experimentelle** Kalle-Variante bauen (`kalle-xl.html`), die auf dem **gesamten ki-mathias.de Blog** (DE + EN, 30 Artikel, ~197K Tokens) trainiert wird — zusätzlich zu Kalles bestehendem Dialog-Korpus.

**Der produktive Kalle (`index.html`) bleibt unverändert.** Kalle XL ist ein paralleles Experiment, kein Ersatz.

## Motivation (aus der T030-Scaling-Analyse)

Der Scaling-Benchmark hat gezeigt: Der Solver bleibt bei 3–5% der Trainingszeit, auch bei 7× mehr Daten. Das heißt: **Das große Korpus ist training-technisch machbar**. Die eigentliche Frage ist, ob ein Kalle mit 18K-Wort-Vokabular qualitativ besser ist — insbesondere für RAG-Queries über Blog-Themen wie Quantenmechanik, Psychologie, Musik, Logik, Emergenz.

**Hypothese:** Der produktive Kalle ist stark im Dialog (kuratierte Pairs), Kalle XL wird stärker im Content-Q&A (breites Blog-Vokabular + viele Chunks).

## Konzept

### Zwei Variante Kalle:

| | Kalle (produktiv) | Kalle XL (experimental) |
|---|---|---|
| Korpus | 2174 kuratierte Dialog-Pairs | Dialog-Pairs + 30 Blog-Artikel als Q&A-Pairs |
| Vokabular | 2977 Wörter | ~18K–22K Wörter |
| RAG-Chunks | 29 (nur Eigenwerte-Post) | ~200–300 (alle Posts in kleineren Einheiten) |
| File-Größe | 56 MB | ~120–180 MB (LFS nötig) |
| Sweet Spot | Dialog, Smalltalk, einfache Math | Blog-Q&A, Querverweise, Themenbreite |
| Deployment | `index.html`, GitHub Pages | `kalle-xl.html`, separat verlinkt |

## Schritte

### Phase 1: Korpus-Engineering (3h)
1. **Blog-Chunking verfeinern:** Jeden der 30 Artikel in H2/H3-Abschnitte (60–120 Wörter pro Chunk). Erwarte ~200–300 Chunks.
2. **Q&A-Pairs pro Chunk generieren** (automatisch via Claude API oder manuell für die wichtigsten Posts):
   - Pro Chunk 10–20 Pairs im `kontext {chunk} frage {q}` Format
   - Ziel: ~3000–5000 neue RAG-Pairs
3. **Bestehende Kalle-Pairs behalten** (2174 Dialog-Pairs) — Kalle XL kann also beides.

### Phase 2: Training-Pipeline-Anpassung (1h)
1. Neuer Build-Modus in `build_v2.py`: `--corpus data/corpus-xl.md`
2. Möglicherweise `--D 8192` oder `--D 12288` (mehr Kapazität für 5× größeres Vokabular)
3. Chunk-Index erweitern auf alle Blog-Posts (nicht nur eigenwerte.html)
4. Build-Befehl: `python3.11 src/build_v2.py --corpus data/corpus-xl.md --solver cg --D 8192 --output kalle-xl.html`

### Phase 3: Testing (2h)
1. **Playwright-Tests** über Blog-Themenbreite (nicht nur Eigenwerte):
   - Quantenmechanik: "What is the Schrödinger equation?"
   - Psychologie: "Wie hängen Emotionen und Entscheidungen zusammen?"
   - Musik: "What makes a chord dissonant?"
   - Logik: "Explain Gödel's incompleteness"
   - Emergenz: "What is emergence in complex systems?"
   - Mindfulness: "Was ist Achtsamkeit?"
2. Top-1-Accuracy messen auf Sample (nicht nur auf kuratierten Pairs)
3. Benchmarks: File-Size, Load-Time im Browser, RAG-Retrieval-Qualität

### Phase 4: Deployment (1h)
1. `kalle-xl.html` als separates File im Repo (evtl. mit Git LFS, wenn >100 MB)
2. Blog-Post um einen "Kalle XL Beta"-Link ergänzen (optional — nach Bewertung der Qualität)
3. README-Sektion "Variants" ergänzen
4. Kein Ersatz von `index.html` — XL ist zusätzlich

## Akzeptanzkriterien

- [ ] ≥ 3000 neue RAG-Pairs generiert (DE + EN), thematisch über alle Blog-Posts verteilt
- [ ] Chunk-Index erweitert auf alle 30 Blog-Posts (~200+ Chunks)
- [ ] Build läuft durch: `kalle-xl.html` generiert, <200 MB
- [ ] Top-1 ≥ 30% auf gemischtem Sample (realistisches Ziel bei 5× Vocab)
- [ ] Playwright-Suite: 20 Queries über 6+ Blog-Themen, mindestens 60% sinnvolle Antworten
- [ ] Dokumentation: `VARIANTS.md` oder README-Sektion beschreibt Kalle vs. Kalle XL
- [ ] Kein Regression für den produktiven Kalle (unverändert auf GitHub Pages)

## Risiken & Mitigation

| Risiko | Wahrscheinlichkeit | Mitigation |
|---|---|---|
| **File-Size > 100 MB** (GitHub Pages-Grenze) | Hoch (bereits 56 MB bei Kalle) | Git LFS, oder Hosting über Cloudflare Pages / Zenodo als Modell-Download |
| **Top-1 fällt < 20%** (Vocab-Kollisionen) | Mittel | D auf 8192 oder 12288 erhöhen; kuratierte Pairs bleiben drin als Anker |
| **Mobile-Load zu lang** | Hoch bei >100 MB | Lazy-Load, Progress-Bar, Mobile-Warnung oder explizit Desktop-Only |
| **Q&A-Qualität aus Auto-Generierung schlecht** | Mittel | Stichproben-Review, kritische Posts (Eigenwerte, KRR) manuell kuratieren |
| **Blog-freier Text passt nicht in `kontext/frage/bot`-Format** | Mittel | Format vorher testen an 2–3 Beispiel-Chunks; ggf. Rewrite-Prompt für Claude |
| **Produktive Demo leidet** | Niedrig | Strikt getrennte Files; Deployment testet isoliert |

## Offene Entscheidungen (bei Implementation)

Diese werden bei Start des Tickets festgelegt, nicht jetzt:
- **D = 8192 oder 12288?** Je mehr V wächst, desto mehr D sollten wir bereitstellen (Empfehlung: 8192 als Start, 12288 falls Top-1 zu niedrig)
- **Q&A-Generierung: Claude API oder manuell?** Hybrid empfohlen: Auto für "leichte" Posts (Deepfakes, Euler), manuell/review für "schwere" (Eigenwerte, KRR)
- **LFS oder externes Hosting?** Abhängig von finaler Größe. LFS ist einfacher, aber hat eigene Größen-Quotas

## Was NICHT Teil dieses Tickets ist

- Der produktive Kalle bleibt unverändert (kein Ersatz durch XL)
- Kein Paper-Update nötig (Paper v2 bleibt gültig — XL ist eine Scaling-Demo, keine neue Methode)
- Kein Blog-Update pflicht (optional nach Bewertung der Qualität)

## Vorbereitet durch

- T030 Scaling-Benchmark zeigt: Training mit 2.3 M Samples ist machbar (~330s gesamt)
- `src/extract_blog_text.py` extrahiert bereits den Blog-Text sauber
- `src/build_v2.py` akzeptiert beliebige Korpus-Dateien via `--corpus`
- `src/solvers.py` skaliert für größere Probleme (CG mit 7–8 Iterationen bei V=19K)

## Status

**Bereit zum Start**, sobald Mathias grünes Licht gibt. Setup-Skripte aus T030 können wiederverwendet werden; T029-Solver sind erprobt.

# T043: Corpus Expansion (~10M Tokens)

**Status:** TODO
**Priorität:** P1
**Parent:** T038
**Geschätzter Aufwand:** 6h
**Abhängig von:** T042 (Cleaning-Pipeline sollte stehen)

## Hypothese

281K Tokens sind 40.000× weniger als GPT-2's Trainings-Korpus. Selbst ein "kleiner" Korpus der Größenordnung ~10M Tokens würde die Generalisierung drastisch verbessern. Die closed-form KRR-Lösung hat dann $N/D \approx 10^7/6144 \approx 1600$ Samples pro RFF-Feature statt $\sim 50$ wie aktuell — erstmals überdeterminiert, statt grob underdetermined.

**Erwartet:** Val Top-1: 9.1% → **20–30%** (mehr Daten = größter Einzel-Hebel).

## Datenquellen (frei verfügbar, CC-lizenziert)

| Quelle | DE | EN | Geschätzt Tokens |
|---|---|---|---|
| Wikipedia Extracts (Top-Articles) | ~5M | ~5M | 10M |
| Project Gutenberg (Public Domain Bücher) | ~1M | ~3M | 4M |
| Blog-Corpus (wir) | 281K (DE+EN) | | 281K |
| OSCAR (multilingual web) — optional |  | | 100M+ |

Ziel: eine Mischung, die genug DE+EN balance hat und nicht nur Fachtext.

## Schritte

1. **Data-Acquisition**:
   - Wikipedia-Extracts: `wikiextractor` auf aktuelle DE/EN-Dumps, top 500 Artikel je Sprache nach Views
   - Gutenberg: `gutenberg-cleaner` auf je 20 deutsche und englische Klassiker
   - Bereinigen (wie T042)
2. **Verbinden + Mischen**: 50% DE, 50% EN, zufällig gemischt (nicht sequenziell — sonst lernt das Modell "erst alles DE, dann EN")
3. **BPE neu trainieren**: V=16384 (größer wegen mehr Daten)
4. **Training**:
   - Accumulation-Zeit wird ~10× länger (etwa 30-60 min)
   - Block-PCG bleibt machbar (hängt von D und V ab, nicht von N)
5. **Evaluation**: Train/Val Top-1, Prompt-Suite, Soft-Score

## Akzeptanzkriterien

- [ ] Mischungs-Korpus mit ≥ 5M Tokens gebaut
- [ ] Tokenizer trainiert, V=16384
- [ ] Training durchgelaufen (inkl. Accumulation im Hintergrund über Nacht)
- [ ] Val Top-1 ≥ 18% (wenn nicht, ist nicht Data der Hebel)
- [ ] Prompt-Suite Soft-Score ≥ 15 von 30
- [ ] Mindestens **1** der 10 Test-Prompts produziert lesbaren Satz-Anfang

## Risiken

- **Disk/RAM**: 10M Tokens als BPE ~100 MB. ZtZ für V=16384, D=6144 ist noch manageable. ZtY: 6144×16384×8 = 800 MB Float64 → wir müssen Float32 oder streaming nutzen.
- **Download-Zeit**: Wikipedia-Extracts sind GB-weise. Vorab: nur Top-Articles, keine full dump.
- **Mischverhältnis**: falsches DE/EN-Balance kann eine Sprache dominieren. Fix: balance token-count, nicht file-count.

## Entscheidung nach diesem Ticket

Wenn Val Top-1 ≥ 20% UND mindestens 3 Prompts lesbar sind:
  → **Publizierbar** als "Language Modeling via fixed-random attention + KRR on small corpora"
  → Paper-Update planen

Wenn nicht:
  → Multi-Layer (T040) kombiniert mit diesem wäre der nächste Schritt
  → Oder: Hypothese "closed-form KRR reicht nicht für echtes LM" bestätigt

## Output

- `data/autoregressive/corpus_large.txt` (5–10 M tokens, gitignored)
- `data/autoregressive/bpe_large.json`
- Trainiertes Modell (gitignored, ~500 MB)
- CSV-Eintrag
- Kurz-Report in diesem Ticket mit Ergebnis

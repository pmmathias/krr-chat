# T042: Corpus Cleaning (LaTeX/Formeln raus)

**Status:** TODO
**Priorität:** P0 (billig, beeinflusst alle Folgeexperimente)
**Parent:** T038
**Geschätzter Aufwand:** 1h

## Hypothese

Die T037-Generation zeigt viele LaTeX-Fragmente in den Top-Predictions: `\(`, `\)`, `\\`, `^`, `{`, `mathbf`, `frac`, `$$`. Grund: der Blog hat tausende LaTeX-Formeln. BPE lernte diese als häufige Tokens. Das **überträgt sich direkt in die Generation-Qualität**: selbst wenn das Modell perfekt lernt, wird es oft LaTeX-Subtokens vorhersagen, weil die im Corpus statistisch dominant sind.

Cleaning sollte:
- Inline-Math `\( ... \)` entfernen
- Display-Math `$$ ... $$` entfernen
- LaTeX-Commands wie `\frac`, `\mathbf` entfernen
- HTML-Entities nochmal gründlicher
- Mehrfache Leerzeichen auf eins
- Pure Prose übrig lassen

**Erwartet:** Val Top-1 bleibt gleich oder ±1pp (Cleaning entfernt 5-10% der Tokens), aber die **Prompt-Suite Soft-Score steigt deutlich** weil Top-Predictions echte Wörter sind, keine Formel-Fragmente.

## Schritte

1. Neues Script `src/autoregressive/clean_corpus.py`:
   - Lese `data/autoregressive/corpus.txt`
   - Strip LaTeX (Regex-basiert)
   - Output `data/autoregressive/corpus_clean.txt`
2. BPE-Tokenizer neu trainieren (V=8192) auf Clean-Corpus
3. Attention-Training wiederholen auf Clean-Corpus
4. Messung (Quant + Qual)

## Akzeptanzkriterien

- [ ] Clean-Corpus existiert, enthält keine sichtbaren LaTeX-Fragmente mehr
- [ ] Neuer BPE-Tokenizer trainiert, V=8192, sample-decode ohne `\(` `\\` etc.
- [ ] Training durchgelaufen, Val Top-1 gemessen
- [ ] Soft-Score auf Prompt-Suite **≥** der Baseline
- [ ] CSV-Eintrag

## Risiken

- **Corpus schrumpft** von 281K auf 220K–250K Tokens. Leichte Reduktion OK.
- **Mathe-relevante Pseudo-Fachbegriffe** gehen verloren (`eigenvalue`, `matrix` stehen aber im Prosa-Text — bleibt).
- **LaTeX-Regex ist nicht perfekt** — Edge Cases bleiben als Rauschen.

## Output

- `src/autoregressive/clean_corpus.py`
- `data/autoregressive/corpus_clean.txt`
- `data/autoregressive/bpe_tokenizer_clean.json`
- CSV-Eintrag

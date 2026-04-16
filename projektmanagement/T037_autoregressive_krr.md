# T037: Autoregressive KRR — GPT-style Next-Token Predictor

**Status:** COMPLETED — NEGATIVE FINDING (scientifically valuable)
**Priorität:** P1 (Forschung, eigener Branch)
**Branch:** `experiment/autoregressive-krr` (NOT merged to main)
**Begonnen:** 2026-04-16
**Abgeschlossen:** 2026-04-16

## Ergebnis

**Autoregressives KRR ohne neuronale Netze funktioniert auf unserem Korpus NICHT.** Jeder Prompt produziert die gleichen Top-5 Vorhersagen (häufigste Tokens wie `.`, `,`, `the`, `die`). Drei Varianten getestet: Random Embeddings, Word2Vec Embeddings, REPEAT=3. Alle kollabieren auf die marginale Verteilung. Val-Top-1: 2.8–3.1%.

**Wissenschaftlicher Wert:** Das Experiment zeigt, dass Kalles Erfolg NICHT durch KRR als Sprachmodell getragen wird — sondern durch die Retrieval-QA-Struktur des kuratierten Korpus. Das stärkt die "less is more" These der v1/v2 Paper. Full writeup in `src/autoregressive/EXPERIMENT_RESULTS.md`.

**Entscheidung:** Branch bleibt bestehen, wird NICHT auf main gemerged. Artifacts preserved für zukünftige Versuche mit Random-Feature-Attention + größerem Korpus.

## Ziel

Eine radikal andere Kalle-Variante bauen: **kein RAG, keine Q&A-Pairs, keine Retrieval-Tricks** — sondern ein **pures autoregressives Sprachmodell** wie GPT-2. Das System lernt aus rohem Fließtext, das jeweils nächste plausible Token vorherzusagen. Gegeben "Heute scheint die" → predict "Sonne".

**Prämisse:** Wir dürfen ALLES nutzen außer neuronale Netze.
- ✅ BPE-Tokenizer aus dem Web (HuggingFace, SentencePiece, tiktoken)
- ✅ Autoregressive causal context (wie GPT's Prompt)
- ✅ Variable Trainings-Sequenz-Längen
- ✅ Kernel-basierte Attention-Mechanismen (RFF-Projektionen, positional kernels)
- ✅ Closed-form KRR als einzige "Lernmaschine"
- ❌ Keine Backpropagation, keine Gradient Descent, keine Transformers

## Architektur-Skizze

```
Input Text: "Heute scheint die Sonne über Hamburg."

  ↓ BPE Tokenizer (V≈8K)
  
Tokens: [4231, 892, 15, 7112, 34, 56, 9821]

  ↓ Embedding Layer (lookup, 32-dim)
  
Embeddings: [e₀, e₁, e₂, e₃, e₄, e₅, e₆] ∈ ℝ^(N×32)

  ↓ Causal Context Encoding (position t → context from 0..t-1)
  
Context vector c_t ∈ ℝ^FEAT für jede Position t:
  Option A: Fixed-length window + positional weighting (Kalle-Style)
  Option B: Cumulative RFF with exponential decay (Linear Attention)
  Option C: Kernel-based attention (pairwise kernel(token_t, token_i) · v_i)

  ↓ RFF Projection (D=6144, σ=1.5)
  
z_t = √(2/D) · cos(c_t · ω + b) ∈ ℝ^D

  ↓ KRR solve over ALL (z_t, next_token) pairs from the corpus
  
W = (Zᵀ Z + λI)⁻¹ Zᵀ Y ∈ ℝ^(D×V)

  ↓ Inference: argmax(z · W) → next token
```

## Unterschied zu Kalle v1/v2

| | Kalle v1/v2 | T037 (Autoregressive) |
|---|---|---|
| Trainings-Daten | Q&A-Pairs (du:/bot:-Format) | Rohe Fließtexte |
| Tokenizer | Whitespace-split | BPE (Subword) |
| Corpus-Struktur | Dialog-Turns | Kontinuierlicher Text |
| Retrieval | BoW+IDF Pair-Match | Keine Retrieval |
| Generierung | Seed (von Pair) → Continuation | Reine Token-für-Token Generation |
| Inferenz | lookup + animate | autoregressive sampling |

## Core technical questions

1. **Welcher BPE-Tokenizer?**
   - `tiktoken` (OpenAI, GPT-2/4 compatible) — V=50K (zu groß für Browser)
   - `huggingface/tokenizers` — eigenes BPE auf unserem Corpus, V=4K–8K (machbar)
   - Entscheidung: **HuggingFace BPE trainiert auf unserem Mixed DE+EN Corpus, V=8192**

2. **Welcher Context-Mechanismus?**
   - **Option A (Kalle-style):** fester Window CTX=64 mit positionalem Gewicht (linear/cosine)
   - **Option B (Linear Attention):** $c_t = \sum_{i<t} \alpha^{t-i} \cdot e_i$ mit decay $\alpha = 0.95$
   - **Option C (Kernel Attention):** $c_t = \sum_{i<t} k(e_t, e_i) \cdot e_i$ mit Gaussian kernel
   - Start mit **Option A** (einfachste, bekannte Kalle-Pipeline wiederverwendbar), dann **Option B** als Upgrade

3. **Welcher Trainings-Corpus?**
   - PoC: Kalle's Blog DE+EN (~200K tokens, bereits extrahiert)
   - Scale-up: + Wikipedia DE/EN Samples (wenn Zeit)
   - Ziel: 1–5 Mio BPE-tokens für den PoC

4. **Welches V und D?**
   - V = 8192 (BPE) → W = 6144 × 8192 × 2 = 96 MB Float16 (ok)
   - D = 6144 (wie Kalle)

## PoC-Schritte (Minimal Viable Product)

1. **Tokenizer trainieren:**
   - HuggingFace tokenizers auf `/tmp/blog_tokens.txt` (bereits vorhanden aus T036)
   - V=8192 BPE
   - Persistieren als `data/bpe_tokenizer.json`

2. **Corpus aufbereiten:**
   - Blog-Text + optional Kalle-Dialog als rohen Text
   - Tokenisieren → ein langer Token-Stream

3. **Training:**
   - Für jede Position t im Token-Stream: context encoding + next-token-label
   - Streaming Z^TZ + Z^TY Akkumulation (wie Kalle)
   - KRR solve

4. **Inference-Script:**
   - Autoregressive generation given a prompt
   - Greedy: argmax
   - Optional: top-k sampling

5. **Test:**
   - "Heute scheint die" → ? (ideal: "Sonne")
   - "Im Anfang war das" → ? (ideal: "Wort")
   - "Kernel ridge" → ? (ideal: "regression")
   - "Der quantum" → ? (something mechanics-related)

6. **HTML-Demo (optional, später):**
   - Wenn die Inferenz funktioniert: Browser-Demo analog zu Kalle

## Akzeptanzkriterien (MVP)

- [ ] BPE-Tokenizer läuft, V = 8192, kann beliebigen Text encoden/decoden
- [ ] Training durchläuft ohne OOM (D=6144, Corpus ≥ 500K tokens)
- [ ] Inferenz produziert kohärente Token-Sequenzen (nicht gibberish)
- [ ] Mindestens 3 von 10 Test-Prompts ergeben sinnvolle Fortsetzungen
- [ ] Top-1 Accuracy (auf Held-out ~5% des Corpus) ≥ 20% (GPT-2-Small hat ~30-40% auf WebText)
- [ ] Alles auf Branch, nicht auf main

## Risiken

| Risiko | Wahrscheinlichkeit | Mitigation |
|---|---|---|
| **V=8192 × D=6144 matrix ist zu groß für Browser** | Mittel | V=4096, D=4096 als Fallback |
| **Qualität zu schlecht (totaler Gibberish)** | Hoch (realistisch) | Ehrlich dokumentieren als "PoC, below GPT-2 quality" — das Experiment ist der Wert |
| **Attention-Mechanismus ohne NN ist trivial-simple** | Hoch | Das ist die Entdeckung — dokumentieren, was funktioniert |
| **Training braucht zu viel RAM** | Mittel | Streaming Z^TZ wie Kalle, CHUNK=10000 |
| **Keine neuronale Netze → kein "learned" Embedding** | Das ist Prämisse | Random oder Hash-basierte Token-Embeddings |

## Was NICHT teil dieses Tickets ist

- HTML-Browser-Demo (erstmal nur Python-Inference)
- Großer Corpus (PoC auf Blog-Größe reicht)
- Merge auf main (bleibt Branch-only bis Qualität klar ist)

## Referenzen

- Kalle's `src/build_v2.py` als Pipeline-Basis
- Kalle's `src/solvers.py` (Block-PCG funktioniert out-of-the-box)
- GPT-2 paper (Radford et al. 2019) für Architektur-Inspiration
- "Linear Transformers as RNNs" (Katharopoulos et al. 2020) für Linear Attention
- RetNet (Sun et al. 2023) für decay-based context mixing

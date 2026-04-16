# T037: Autoregressive KRR — Experiment Results (Negative Finding)

**Branch:** `experiment/autoregressive-krr`
**Date:** 2026-04-16
**Verdict:** The experiment failed in a scientifically meaningful way.

## What we tried

A pure GPT-style next-token predictor using KRR instead of a transformer:

- **BPE tokenizer** (V = 8,192), trained on the full ki-mathias.de blog (~281K tokens)
- **Fixed-length causal context** (CTX = 64 BPE tokens)
- **Token embeddings**: tried two variants
  - (a) random fixed embeddings (EMB_DIM = 32, unit-normalized)
  - (b) Word2Vec trained on the BPE token stream (Kalle-style)
- **Position-weighted context vector** (linear ramp 0.4 → 1.0)
- **RFF projection** to D = 6,144
- **Block-PCG solve** over all ~267K (context → next-token) pairs

## Results

| Embedding | REPEAT | Training Top-1 | Training Top-5 | Validation Top-1 | Validation Top-5 |
|---|---|---|---|---|---|
| Random | 1 | 18.4% | 84.4% | 3.0% | 10.0% |
| Word2Vec | 1 | 13.2% | 74.6% | 3.1% | 11.5% |
| Word2Vec | 3 | 15.8% | 80.7% | 2.8% | 10.2% |

**All three variants collapse to the same generation behaviour:** every prompt produces the same Top-5 predictions — the most frequent tokens in the corpus (`.`, `,`, `the`, `die`, `:`). Example:

```
Prompt: "Heute scheint die"
Top-5 next: '.', ',', ' the', ' die', ':'
Greedy continuation: "Heute scheint die..... die.."

Prompt: "Im Anfang war das"
Top-5 next: '.', ',', ':', ' the', ' ist'
Greedy continuation: "Im Anfang war das, ( is the \\( – \\( dies Die"

Prompt: "Kernel ridge"
Top-5 next: '.', ',', ' the', ' \\', ':'
Greedy continuation: "Kernel ridge dien_ The.)\\, die{\\)0"
```

The model has **only learned the marginal distribution**, not the conditional. It is mathematically equivalent to always outputting the most frequent token regardless of context.

## Why it fails (analysis)

Three structural reasons, each sufficient on its own to explain the failure:

### 1. Corpus is 40,000× too small for this architecture

GPT-2-Small was trained on ~10 billion tokens. We have ~281K tokens. At D = 6144 RFF features and V = 8192 outputs, the system has $D \cdot V \approx 50$ million parameters — roughly 2.5× more parameters than training samples. This is a massively **under-determined** regression problem; the closed-form solve finds a minimum-norm solution that generalizes to the marginal rather than capturing conditional structure.

### 2. Context representation is too rigid for variable-content conditioning

The position-weighted sum $c_t = \sum_{i=t-64}^{t-1} w_i \cdot e_i$ treats position deterministically, regardless of which tokens are at which positions. In a transformer, Q·K·V lets *content* determine which earlier tokens matter. Here, position alone determines weight — so the representation at position $t$ becomes a smoothed mixture where specific rare words (the ones that carry predictive information) are overwhelmed by the hundreds of common ones in the window.

### 3. The RFF kernel is the *wrong* similarity

The Gaussian kernel measures $L_2$ distance between context vectors. But two contexts with the same bag of common tokens in different orders produce very similar vectors; two contexts with different rare tokens in the same positions produce very different vectors. This is the opposite of what next-token prediction needs. A proper attention mechanism (even without learned parameters) would compare tokens token-to-token, not context-vector-to-context-vector.

## Why this is still a valuable result

**The experiment answers a question Kalle v1/v2 never answered directly:**
*Is Kalle's success due to KRR being a good language model?*
**No. Kalle's success comes from the retrieval-QA framing, not from KRR's language-modeling capacity.**

More precisely:
- Kalle trains on **repeated, curated Q&A pairs** (REPEAT=5). Each (context → next-token) pair is seen many times; the model can memorize them. At inference, the BoW+IDF retrieval finds the *right pair*, and the KRR is essentially validating the memorization.
- Autoregressive KRR has no such structure. Each (context → next-token) pair is unique, nearly never repeated, and the context is part of a continuous stream with no dialogue boundaries.

**The "less is more" finding from the v1 paper is strengthened by this experiment.** The curated corpus isn't just "good data" — it is a *structural fit* for the KRR+BoW+IDF architecture that makes the method work at all.

## What would be required to make autoregressive KRR work

Three things, all expensive:

1. **A corpus that is ≥ 100× larger** (tens of millions of tokens), ideally with repetition of common phrasings to give the closed-form solve enough signal.

2. **A content-adaptive attention mechanism without a neural network**. Candidates:
   - **Random-feature attention** (Choromanski et al., 2020) — random features approximate softmax attention, and the queries/keys/values could themselves be RFF projections of token embeddings (no learned parameters)
   - **Linear attention with fixed projections** (Katharopoulos et al., 2020) — $Q K^\top$ becomes a kernel product
   - **Retention mechanism** (Sun et al., 2023) — exponential decay over positions, but the decay is content-independent so probably insufficient
   
   Crucially, *all of these still require the feature projections to be informative*. Without learned Q/K/V matrices, the random projections would have to be wide enough (D very large) to cover the space — pushing $D$ to $10^5$ or more, which is computationally infeasible for KRR's closed-form solve.

3. **Hierarchical KRR**: first predict token category (syntax), then token within category. Reduces effective $V$ at each step.

None of these are small projects. Each would be a separate paper.

## The deeper lesson

KRR, without learned parameters, is a **similarity-based method**. It excels when the training data has a structure where "similar context → similar target" is a strong signal (Kalle's Q&A retrieval). It fails when the signal requires *content-adaptive aggregation* (which is what attention was invented to provide).

In retrospect: the v1 paper's claim that "KRR is as powerful as a neural network in the infinite-width limit" (via NTK) is technically true, but the *finite-width* gap matters enormously. At our actual $D = 6144$, we are nowhere near the NTK limit — we are in the regime where the feature representation fundamentally constrains what can be learned.

## Decision

This branch remains **experimental and unmerged**. The artifacts (code, tokenizer, trained weights) are preserved for future researchers who might want to extend the work with random-feature attention or larger corpora.

For the production Kalle, the result is clear: **stick with the curated-corpus + retrieval approach**. Kalle v1 (live at pmmathias.github.io/krr-chat) and the v2 paper are the correct artifacts to keep promoting.

## Artifacts on this branch

- `src/autoregressive/prepare_corpus.py` — Corpus extraction + BPE tokenizer training
- `src/autoregressive/train_arkrr.py` — Autoregressive KRR training pipeline
- `src/autoregressive/generate.py` — Inference with top-k/temperature sampling
- `data/autoregressive/corpus.txt` — 1.12 MB plain text from 30 blog articles
- `data/autoregressive/bpe_tokenizer.json` — BPE tokenizer, V=8192
- `data/autoregressive/model.pkl` — Trained W, ω, bias, embeddings (~100 MB, gitignored)

## Reproducing

```bash
# Prepare corpus + train BPE
python3.11 src/autoregressive/prepare_corpus.py

# Train autoregressive KRR (~3 min with REPEAT=1)
python3.11 src/autoregressive/train_arkrr.py

# Try some prompts
python3.11 src/autoregressive/generate.py
python3.11 src/autoregressive/generate.py "Heute scheint die" --n 10
```

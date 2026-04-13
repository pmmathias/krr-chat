# Architecture

## Core Principle

Both chatbots solve the same task as GPT-4 — **predict the next word given context** — but with Kernel Ridge Regression instead of neural networks. No backpropagation, no gradient descent, no attention mechanism. One closed-form matrix solve.

## Pipeline

```
                         OFFLINE (Python, Float64)
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Corpus (2113 pairs)                                                 │
│       │                                                              │
│       ├──→ Word2Vec (gensim, 32-dim)  → Embedding matrix E          │
│       │                                                              │
│       ├──→ Token sequence (5× repeat) → Training contexts            │
│       │         │                                                    │
│       │         ├──→ encode(ctx) = [E[w₁]·α₁ ... E[w₂₄]·α₂₄]      │
│       │         │         │                                          │
│       │         │         └──→ z(x) = √(2/D)·cos(x·ω + b)    [RFF] │
│       │         │                   │                                │
│       │         │                   └──→ Z matrix (streaming,        │
│       │         │                        10K chunks to avoid OOM)    │
│       │         │                                                    │
│       │         └──→ Y (one-hot next-word targets)                   │
│       │                                                              │
│       │         W = (Z^TZ + λI)^-1 Z^TY              [KRR solve]    │
│       │                                                              │
│       ├──→ IDF weights (per-word inverse document frequency)         │
│       │                                                              │
│       └──→ BoW+IDF pair embeddings (32-dim sentence vectors)         │
│                                                                      │
│  Pack all as Float16 + gzip + base64 → inject into HTML template     │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ONLINE (Browser, WebGL GPU)
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  User query                                                          │
│       │                                                              │
│       ├──→ [1] CHUNK RETRIEVAL (RAG)                                 │
│       │    Query keywords vs 29 blog chunks → best chunk             │
│       │    Prompt: "kontext {chunk} frage {query}"                   │
│       │                                                              │
│       ├──→ [2] PAIR MATCHING (BoW+IDF)                               │
│       │    Combined score = 0.65 × keyword + 0.35 × semantic         │
│       │    Cosine similarity on 32-dim embeddings                    │
│       │    → best Q&A pair = the answer (seed)                       │
│       │                                                              │
│       └──→ [3] PREDICTION COMPARISON (KRR)                           │
│            Per seed word: KRR predicts top-3 next words              │
│            Green = KRR agrees with corpus (verbatim knowledge)       │
│            Yellow = KRR would have chosen differently                │
│                                                                      │
│  Multi-turn: lastBotTurn + userInput → combined query                │
│  (AIML <that>-style context, no code tricks)                         │
└──────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Why KRR instead of Neural Networks?

KRR solves the same optimization problem (minimize prediction error + regularization) but in **closed form**. The solution `W = (Z^TZ + λI)^-1 Z^TY` is computed once — no epochs, no learning rate, no convergence monitoring. This makes the entire system **mathematically transparent**: every component maps to a concept from linear algebra (eigenvalues, kernel functions, regularization).

### Why Random Fourier Features?

The Gaussian kernel `k(x,x') = exp(-||x-x'||²/2σ²)` measures similarity between contexts. Computing the full kernel matrix for N training samples would require O(N²) memory. RFF approximates the kernel via random projection: `z(x)^T z(x') ≈ k(x,x')` with D=6144 random features. This reduces the problem to a standard linear system of size D×D.

### Why Word2Vec instead of Hash Encoding?

The original KRR Chat used `hash(word) % 128` — fast but lossy. With 505 words mapped to 128 buckets, ~4 words share each bucket. The model cannot distinguish "pizza" from "comedy" if they hash to the same bucket. Word2Vec (32-dim, trained on the corpus) gives each word a unique vector. Zero collisions. Similar words get similar vectors. Feature dimension drops from 3072 to 768.

### Why BoW+IDF for Matching?

The answer retrieval is not a neural operation — it's weighted bag-of-words with IDF. This is deliberate: for a domain-specific corpus with distinctive keywords (eigenvalue, regularization, PageRank), IDF naturally gives rare technical terms high weight. "eigenvalue" (IDF ≈ 4.2) dominates over "is" (IDF ≈ 0.3). No sentence transformer needed.

### Why is the Template 19 MB?

The HTML template contains ~80 lines of matching logic in vanilla JavaScript. The model weights (ω, W, embeddings, pair vectors) are embedded as base64-encoded Float16+gzip blobs. After browser decompression, the full model occupies ~50 MB in GPU memory. No server, no API calls, no dependencies.

## File Structure

```
krr-chat/
├── index.html              # The chatbot — self-contained (~33 MB), GitHub Pages entry point
├── README.md
├── ARCHITECTURE.md         # ← you are here
├── src/
│   ├── build.py            # Full build pipeline: corpus → Word2Vec → KRR → HTML
│   ├── gen_corpus.py       # Curated corpus generator (6 categories, ~2100 pairs)
│   └── gen_rag_qa.py       # RAG Q&A pair generator from blog chunks
├── tests/
│   └── test_regression.py  # Playwright regression suite (34 scenarios)
└── data/
    ├── corpus.md           # 2113 curated dialog pairs (the training data)
    ├── chunk_index.json    # 29 blog chunks for RAG retrieval
    └── template.html       # HTML/JS template (matching + rendering logic)
```

## Bundling: How everything becomes one HTML file

The key insight: **the entire model runs client-side**. There is no server, no API, no backend. Everything — model weights, matching logic, rendering code — is packed into a single HTML file.

The build script (`src/build.py`) performs this bundling:

1. **Train offline** (Python, Float64): Word2Vec embeddings, RFF projection, KRR solve
2. **Quantize**: Float64 → Float16 (halves size, sufficient precision for inference)
3. **Compress**: gzip level 9 on the raw Float16 bytes
4. **Encode**: base64 for safe embedding in HTML/JavaScript
5. **Inject**: Replace the `var M={...}` blob in `data/template.html` with the new weights
6. **Output**: `index.html` — a self-contained HTML file (~33 MB)

At runtime, the browser reverses this: base64 → gunzip → Float16 → Float32 GPU tensors (via TensorFlow.js WebGL backend). The entire decompression takes ~2 seconds on modern hardware.

The template (`data/template.html`) contains:
- ~80 lines of vanilla JS for BoW+IDF matching and scoring
- ~40 lines for word-by-word rendering with prediction comparison
- ~30 lines for chunk retrieval (RAG)
- ~20 lines for multi-turn context (lastBotTurn concatenation)
- TensorFlow.js for WebGL GPU matrix operations
- The CSS and HTML for the chat interface

## Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| CTX | 24 words | Long enough for multi-turn context via lastBotTurn |
| EMB_DIM | 32 | Word2Vec dimension — sufficient for 1445 vocab |
| FEAT | 768 (24 × 32) | Context encoded as concatenated embeddings |
| D | 6144 | RFF dimension — 8× oversampling of FEAT |
| σ | 1.5 | Kernel bandwidth — empirical compromise (memorization vs generalization) |
| λ | 10⁻⁶ | Regularization — small because training set is dense |
| REPEAT | 5 | Corpus repetition for training signal |

## The Math

Three equations. That's the entire model.

**1. Random Fourier Features** (Rahimi & Recht, 2007):
```
z(x) = √(2/D) · cos(x · ω + b)
```
where ω ~ N(0, 1/σ²), b ~ Uniform(0, 2π). This approximates the Gaussian kernel.

**2. Kernel Ridge Regression** (closed-form):
```
W = (Z^TZ + λI)^-1 Z^TY
```
No gradient descent. One matrix solve. W is the only learned parameter (6144 × 1445 ≈ 8.9M values).

**3. Prediction** (inference):
```
next_word = argmax(z(context)^T · W)
```
A single matrix-vector multiplication per word. Runs in <1ms on WebGL GPU.

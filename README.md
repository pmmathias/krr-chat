<div align="center">

# λ Kalle — KRR Chat

### A language model with no neural network.
### Eigenvalues, kernel ridge regression, and honest corpus engineering.

[![Try Kalle](https://img.shields.io/badge/🚀_Try_Kalle-Live_Demo-22d3ee?style=for-the-badge)](https://pmmathias.github.io/krr-chat/)
[![Paper](https://img.shields.io/badge/📄_Paper-Zenodo_(DOI)-blue?style=for-the-badge)](https://doi.org/10.5281/zenodo.19595642)
[![Blog Post](https://img.shields.io/badge/📖_Deep_Dive-How_It_Works-f59e0b?style=for-the-badge)](https://ki-mathias.de/en/krr-chat-explained.html)
[![Eigenvalues & AI](https://img.shields.io/badge/🔬_Foundation-Eigenvalues_&_AI-34d399?style=for-the-badge)](https://ki-mathias.de/en/eigenvalues.html)

![No Neural Network](https://img.shields.io/badge/neural_network-NONE-ff6b6b?style=flat-square)
![Vanilla JS](https://img.shields.io/badge/vanilla_JS-~80_lines_matching-34d399?style=flat-square)
![Browser GPU](https://img.shields.io/badge/runs_in-browser_(WebGL_GPU)-22d3ee?style=flat-square)
![Float64](https://img.shields.io/badge/trained-Float64_offline-f59e0b?style=flat-square)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19595642.svg)](https://doi.org/10.5281/zenodo.19595642)
![MIT](https://img.shields.io/badge/license-MIT-gray?style=flat-square)

<br>

**The same task as GPT-4 — predict the next word — solved with eigenvalues instead of backpropagation.**

### [**→ Launch Kalle in your browser**](https://pmmathias.github.io/krr-chat/)

<br>
</div>

---

## What is this?

Kalle is a bilingual chatbot (German + English) that runs **entirely in your browser** using **Kernel Ridge Regression** with **Random Fourier Features**. No neural networks, no backpropagation, no gradient descent, no server — just matrix multiplication, a Gaussian kernel, and eigenvalues. TensorFlow.js provides WebGL GPU acceleration; the model weights are embedded directly in the HTML.

Kalle chats about food, hobbies, music, feelings, weather, and simple math — with multi-turn context awareness, honest scope boundaries, and **RAG** (Retrieval-Augmented Generation) over the [Eigenvalues & AI](https://ki-mathias.de/en/eigenvalues.html) blog post. Ask "What are eigenvalues?" or "How does PageRank work?" and Kalle retrieves the relevant blog section to answer.

## How the build works

Everything is bundled into a **single self-contained HTML file** (`index.html`, ~33 MB). No server, no API, no external dependencies at runtime.

```
OFFLINE BUILD (Python, Float64, ~3 min on CPU)
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  data/corpus.md (2113 dialog pairs)                     │
│       │                                                 │
│       ├──→ Word2Vec (gensim, 32-dim) → embeddings       │
│       ├──→ Token sequence (5× repeat) → RFF features    │
│       │         └──→ W = (Z^TZ + λI)⁻¹ Z^TY   [KRR]   │
│       ├──→ IDF weights + BoW pair embeddings             │
│       └──→ data/chunk_index.json → RAG chunk embeddings  │
│                                                         │
│  All tensors → Float16 + gzip + base64                  │
│       └──→ Injected into data/template.html             │
│            └──→ index.html (self-contained, ~33 MB)     │
└─────────────────────────────────────────────────────────┘

ONLINE (Browser, WebGL GPU)
┌─────────────────────────────────────────────────────────┐
│  1. Decompress base64 → Float16 → GPU tensors           │
│  2. User query → chunk retrieval (RAG) → pair matching   │
│  3. Word-by-word rendering with prediction comparison    │
│  4. TensorFlow.js WebGL for matrix ops (<1ms/word)      │
└─────────────────────────────────────────────────────────┘
```

The template (`data/template.html`) contains ~80 lines of vanilla JavaScript for the matching and rendering logic. The build script (`src/build.py`) trains the model offline in Float64 for numerical precision, then quantizes to Float16 and packs everything into the template.

## The math (three equations)

```python
# 1. Random Fourier Features (Rahimi & Recht, 2007)
z(x) = sqrt(2/D) * cos(x @ omega + bias)       # ω ~ N(0, 1/σ²)

# 2. Kernel Ridge Regression (closed-form, no gradient descent)
W = solve(Z.T @ Z + lambda * I, Z.T @ Y)        # one matrix solve

# 3. Prediction (single matrix-vector multiply)
next_word = argmax(z(context) @ W)               # <1ms on WebGL GPU
```

No epochs. No learning rate. No convergence monitoring. `W` is the only learned parameter (6144 × 2952 ≈ 18.1M values).

## Key numbers

| Parameter | Value |
|---|---|
| Corpus | 2174 curated dialog pairs (DE + EN) |
| Vocabulary | 2952 words (Word2Vec, 32-dim) |
| Context window | 24 words |
| RFF dimension (D) | 6144 |
| Kernel bandwidth (σ) | 1.5 |
| Regularization (λ) | 10⁻⁶ |
| RAG chunks | 29 (from Eigenvalues & AI blog) |
| File size | ~56 MB (self-contained HTML) |
| Runtime | WebGL GPU via TensorFlow.js |

## Architectural properties

Deterministic consequences of the BoW+IDF design — not programmed, but also not magical:

- **Math validation**: Kalle asks "what is 3+5?", user says "8" → "correct!" — pure pattern matching on `plus 3 5 8`, no code checks the math
- **Insult immunity**: Profanity is out-of-vocabulary → invisible to the model → no filter needed
- **Typo robustness**: OOV words from typos are silently ignored, remaining words still match
- **Bilingual routing**: English/German words have different IDF weights → queries route to language-appropriate pairs without language detection code
- **Honest limits**: Low confidence → "I can only talk about food, hobbies, music, feelings, weather and math"

## The journey: less is more

| Iteration | Pairs | Encoding | Top-1 | Lesson |
|---|---|---|---|---|
| V1 (original) | 57 | Hash (128 buckets) | 99.8% | Perfect but narrow |
| MEGA (mass) | 4301 | Word2Vec + hacks | 34.9% | More data = worse! |
| **FINAL (curated)** | **2174** | **Word2Vec (32-dim)** | **65.4%** | **Curated > generated** |

This mirrors what OpenAI, Anthropic and Google learned: **data quality beats data quantity**.

## Engineering

### Build from source

```bash
pip install numpy gensim

# Generate corpus (optional — data/corpus.md already included)
python3 src/gen_corpus.py

# v1 build: direct Gaussian elimination (fastest on CPU at current scale)
python3 src/build.py
# → produces index.html (~56 MB, all weights embedded, runs in any browser)

# v2 build: pluggable solver (v1-compatible + new iterative paths)
python3 src/build_v2.py --solver=direct                           # v1-equivalent
python3 src/build_v2.py --solver=cg                               # Block-PCG (default, GPU-friendly)
python3 src/build_v2.py --solver=cg --cg-tol=1e-8 --cg-maxiter=2000  # tighter tolerance
# → produces kalle-chat-v2.html
```

The v2 solver implements the absorber-stochasticization described in the
[v2 paper](https://doi.org/10.5281/zenodo.19595642) — same result as direct solve,
but using only matrix-vector products (GPU-ideal, scales to D ≫ 10,000).
See [`benchmarks/README.md`](benchmarks/README.md) for detailed comparisons.

### Run tests

```bash
pip install playwright && playwright install chromium

# Full regression suite (34 scenarios)
python3 tests/test_regression.py index.html

# Filter by category
python3 tests/test_regression.py index.html --filter math
python3 tests/test_regression.py index.html --filter emotion
```

### Run locally

```bash
git clone https://github.com/pmmathias/krr-chat.git
cd krr-chat
open index.html   # opens in default browser, GPU accelerated
```

No build step needed to run — the HTML file is self-contained with embedded model weights.

### Project structure

```
krr-chat/
├── index.html              # The chatbot (self-contained, ~56 MB)
├── README.md
├── ARCHITECTURE.md         # Full technical deep-dive
├── src/
│   ├── build.py            # Build pipeline: corpus → Word2Vec → KRR → HTML
│   ├── gen_corpus.py       # Curated corpus generator (~2100 pairs)
│   └── gen_rag_qa.py       # RAG Q&A pair generator from blog chunks
├── tests/
│   └── test_regression.py  # Playwright regression suite (34 scenarios)
└── data/
    ├── corpus.md           # 2174 curated dialog pairs (training data)
    ├── chunk_index.json    # 29 blog chunks for RAG retrieval
    └── template.html       # HTML/JS template (matching + rendering logic)
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full technical architecture, design decisions, and mathematical foundations.

## Further reading

- **[How KRR Chat Works](https://ki-mathias.de/en/krr-chat-explained.html)** — deep-dive blog post covering every component: from RFF kernel approximation to the eigenvalue spectrum of the training matrix
- **[Eigenvalues & AI](https://ki-mathias.de/en/eigenvalues.html)** — the foundation: why eigenvalues connect Google PageRank, regularization, quantum mechanics, and language models
- **[KI-Mathias Blog](https://ki-mathias.de/en/)** — all posts on the mathematical foundations of AI

## License

MIT

## Author

[Mathias Leonhardt](https://ki-mathias.de/en/about.html) — CTO at [pmagentur.com](https://pmagentur.com), writing about the mathematical foundations of AI at [ki-mathias.de](https://ki-mathias.de/en/)

Built in collaboration with [Claude Code](https://claude.ai/code) (Anthropic)

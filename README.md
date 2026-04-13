<div align="center">

# λ KRR Chat + 👋 Kalle

### Language models with no neural network.
### Just eigenvalues, kernel ridge regression, and honest corpus engineering.

[![KRR Chat](https://img.shields.io/badge/🔴_KRR_Chat-Eigenvalues_(EN)-22d3ee?style=for-the-badge)](https://ki-mathias.de/krr-chat.html)
[![Kalle](https://img.shields.io/badge/👋_Kalle-Buddy_Chat_(DE)-34d399?style=for-the-badge)](https://ki-mathias.de/kalle-chat.html)
[![Blog Post](https://img.shields.io/badge/📖_Deep_Dive-Under_the_Hood-f59e0b?style=for-the-badge)](https://ki-mathias.de/en/krr-chat-explained.html)

![No Neural Network](https://img.shields.io/badge/neural_network-NONE-ff6b6b?style=flat-square)
![Vanilla JS](https://img.shields.io/badge/vanilla_JS-~80_lines_matching-34d399?style=flat-square)
![Browser](https://img.shields.io/badge/runs_in-browser_(GPU)-22d3ee?style=flat-square)
![Float64](https://img.shields.io/badge/trained-Float64_offline-f59e0b?style=flat-square)
![MIT](https://img.shields.io/badge/license-MIT-gray?style=flat-square)

<br>

**The same task as GPT-4 — predict the next word — solved with eigenvalues instead of backpropagation.**

<br>
</div>

---

## Two chatbots, one principle

| | λ KRR Chat (EN) | 👋 Kalle (DE) |
|---|---|---|
| **Language** | English (technical) | German (casual) |
| **Corpus** | 104 sentences on eigenvalues & ML | **2113 curated dialogue pairs** |
| **Vocabulary** | 505 words (hash encoding) | **1445 words (Word2Vec, 32-dim)** |
| **Context** | 5 words | **24 words** |
| **RFF dimension** | D=1536 | **D=6144** |
| **Multi-turn** | Keyword memory | **lastBotTurn concatenation + `<that>`-pairs** |
| **Scope handling** | — | **Honest fallback ("I can only talk about X")** |
| **File size** | 5.8 MB | **33 MB** |

[**Try KRR Chat (EN)**](https://ki-mathias.de/krr-chat.html) · [**Try Kalle (DE)**](https://ki-mathias.de/kalle-chat.html)

## What is this?

Two chatbots that run entirely in your browser using **Kernel Ridge Regression** with **Random Fourier Features**. No neural networks, no backpropagation, no gradient descent — just matrix multiplication, a Gaussian kernel, and eigenvalues.

**KRR Chat** answers questions about eigenvalues, kernel methods, and machine learning in English.

**Kalle** is a German buddy chatbot that chats about food, hobbies, music, feelings, weather, and simple math — with multi-turn context awareness and honest scope boundaries. It also answers questions about the [Eigenvalues & AI blog](https://ki-mathias.de/eigenwerte.html) via RAG (Retrieval-Augmented Generation) — without an LLM.

## Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full technical deep-dive. The core pipeline:

```
OFFLINE (Python, Float64):
  Corpus → Word2Vec → Token contexts → RFF projection → KRR solve → Pack into HTML

ONLINE (Browser, WebGL GPU):
  User query → Chunk retrieval (RAG) → BoW+IDF pair matching → Word-by-word rendering
```

Three equations power the entire system:

```python
z(x) = sqrt(2/D) * cos(x @ omega + bias)           # Random Fourier Features
W = solve(Z.T @ Z + lambda * I, Z.T @ Y)            # Kernel Ridge Regression
next_word = argmax(z(context) @ W)                   # Prediction
```

No gradient descent. No epochs. One matrix solve.

## Kalle's matching pipeline

```
User: "Fisch" (after Kalle talked about pizza)

1. queryText = lastBotTurn + " " + userInput
   → "mein lieblingsessen ist pizza ... fisch"

2. BoW+IDF embedding (32-dim Word2Vec)
   → semantic query vector

3. Combined scoring: 0.65 × keyword + 0.35 × semantic
   → best matching dialogue pair

4. Response from matched pair (word-by-word animation)

5. Prediction Comparison: KRR's top-3 prediction per word
   → green (agreed) / yellow (would have differed)
```

**No stemming. No substring expansion. No keyword hacks.** Just full-text BoW+IDF matching against curated dialogue pairs. The IDF weighting naturally down-weights function words.

## Architectural properties

Things that emerge **deterministically** from the BoW+IDF design — not programmed, but also not magical:

- **Math validation**: Kalle asks "what is 3+5?", user says "8" → "correct! 3+5=8" — no code checks the math, it's pure pair matching on `plus 3 5 8`
- **Insult immunity**: Profanity is out-of-vocabulary → invisible to the model → no filter needed
- **Typo robustness**: OOV words from typos are silently ignored, remaining words still match
- **Honest limits**: Low confidence → "I can only talk about food, hobbies, music, feelings, weather and math"
- **Bilingual routing**: English and German words have different IDF weights → queries automatically route to language-appropriate pairs without language detection code

## The journey: Less is more

| Iteration | Pairs | Encoding | Top-1 | Lesson |
|---|---|---|---|---|
| V1 (original) | 57 | Hash (128 buckets) | 99.8% | Perfect but narrow |
| MEGA (mass) | 4301 | Word2Vec + hacks | 34.9% | More data = worse! |
| **FINAL (curated)** | **2113** | **Word2Vec (32-dim)** | **65.4%** | **Curated > generated** |

This mirrors what OpenAI, Anthropic and Google learned: **data quality beats data quantity**. Instruction-tuning datasets are carefully curated — just like Kalle's corpus.

## Engineering

### Build from source

```bash
# Prerequisites
pip install numpy gensim

# Generate corpus (optional — data/corpus.md already included)
python3 src/gen_corpus.py

# Build HTML (trains Word2Vec, solves KRR, packs into self-contained HTML)
python3 src/build.py
# → produces kalle-chat.html (~33 MB, all weights embedded)
```

### Run tests

```bash
# Prerequisites
pip install playwright && playwright install chromium

# Full regression suite (34 scenarios: single-turn, multi-turn, edge cases)
python3 tests/test_regression.py kalle-chat.html

# Filter by category
python3 tests/test_regression.py kalle-chat.html --filter math
python3 tests/test_regression.py kalle-chat.html --filter emotion

# Custom pass threshold
python3 tests/test_regression.py kalle-chat.html --threshold 0.9
```

### Run locally

```bash
open index.html       # KRR Chat (EN)
open kalle-chat.html  # Kalle (DE)
```

No build step needed to run — self-contained HTML files with embedded model weights.

### Project structure

```
krr-chat/
├── index.html              # KRR Chat (EN) — eigenvalue Q&A, 505 vocab
├── kalle-chat.html         # Kalle (DE+EN) — dialog chatbot, 1445 vocab, RAG
├── README.md
├── ARCHITECTURE.md         # Full technical architecture documentation
├── src/
│   ├── build.py            # Build pipeline: corpus → Word2Vec → KRR → HTML
│   ├── gen_corpus.py       # Curated corpus generator (6 categories, ~2100 pairs)
│   └── gen_rag_qa.py       # RAG Q&A pair generator from blog chunks
├── tests/
│   └── test_regression.py  # Playwright regression suite (34 scenarios)
└── data/
    ├── corpus.md           # 2113 curated dialog pairs (the training data)
    ├── chunk_index.json    # 29 blog chunks for RAG retrieval
    └── template.html       # HTML/JS template (matching + rendering logic)
```

## Connection to the blog

Interactive companion to [Eigenwerte & KI](https://ki-mathias.de/eigenwerte.html) ([English](https://ki-mathias.de/en/eigenvalues.html)) on [ki-mathias.de](https://ki-mathias.de). The [deep-dive blog post](https://ki-mathias.de/en/krr-chat-explained.html) explains every component with mathematical rigor — from RFF kernel approximation to the eigenvalue spectrum of the training matrix.

## License

MIT

## Author

[KI-Mathias](https://ki-mathias.de) (Mathias Leonhardt) — in collaboration with Claude Code (Anthropic)

<div align="center">

# λ KRR Chat

### A language model with no neural network.
### Just eigenvalues, kernel ridge regression, and 120 lines of JavaScript.

[![Live Demo](https://img.shields.io/badge/🔴_Live_Demo-ki--mathias.de-22d3ee?style=for-the-badge)](https://ki-mathias.de/krr-chat.html)
[![Blog Post](https://img.shields.io/badge/📖_Blog_Post-Under_the_Hood-f59e0b?style=for-the-badge)](https://ki-mathias.de/en/krr-chat-explained.html)

![No Neural Network](https://img.shields.io/badge/neural_network-NONE-ff6b6b?style=flat-square)
![Vanilla JS](https://img.shields.io/badge/vanilla_JS-120_lines-34d399?style=flat-square)
![Browser](https://img.shields.io/badge/runs_in-browser-22d3ee?style=flat-square)
![Float64](https://img.shields.io/badge/trained-Float64-f59e0b?style=flat-square)
![MIT](https://img.shields.io/badge/license-MIT-gray?style=flat-square)

<br>

**The same task as GPT-4 — predict the next word — solved with eigenvalues instead of backpropagation.**

<br>
</div>

---

[**Try it live**](https://ki-mathias.de/krr-chat.html)

## What is this?

A dual-model language chatbot that runs entirely in your browser. It answers questions about eigenvalues, kernel methods, quantum mechanics, and machine learning using Kernel Ridge Regression with Random Fourier Features — the same mathematical chain described in the [Eigenvalues & AI](https://ki-mathias.de/eigenwerte.html) blog post.

**No neural networks.** No backpropagation. No gradient descent. Just matrix multiplication, a Gaussian kernel, and eigenvalues.

## Architecture: Dual Model (RAG-style)

```
User question → Keyword extraction → Retrieve best matching sentence (invisible)
→ Use as seed context → KRR generation (word by word) → Color-coded output
```

### System 1 — Retrieval (805 sentences)
Keyword search over a corpus of ArXiv ML abstracts + curated eigenvector content. Finds the most relevant sentence to use as generation context. **Not shown directly** — acts as invisible seed (like RAG).

### System 2 — Generation (KRR language model)
Trained offline (Float64, NumPy) on 104 curated sentences (505-word vocabulary, 1427 tokens). Predicts the next word given the last 5 words. The model weights are loaded as compressed Float16 in the browser.

### Conversation Memory
Follow-up questions ("and then?", "what about...?") inherit keywords from previous turns and continue generating from where the last answer ended — inspired by LangChain's ConversationBufferWindowMemory.

## Color-coded output

Every generated word is color-coded by source:

| Color | Meaning |
|-------|---------|
| **Green** | KRR-generated, appears as 3/4-gram in training data (verbatim / memorization) |
| **Orange** | KRR-generated, NOT in training data (novel / generalization) |

Typically ~60-70% verbatim, ~30-40% novel. The coloring makes the boundary between memorization and generalization literally visible.

## The math behind it

```
Word hashing          → Feature map φ(x)
Random Fourier Features → Kernel approximation k(x,x') ≈ z(x)ᵀz(x')
Gaussian elimination   → Solve (ZᵀZ + λI)W = ZᵀY
Ridge parameter λ      → Regularization = early stopping
Hash collisions        → Condition number of the matrix
Float64 vs Float32     → Numerical stability (why offline training matters)
```

- **Kernel Ridge Regression** replaces neural networks
- **Random Fourier Features** (Rahimi & Recht, 2007) break the O(n³) scaling barrier
- **Eigenvalues** of the kernel matrix determine what the model learns (signal) and ignores (noise)
- **The Representer Theorem** guarantees: the training data IS the model

## Why offline training?

Training directly in the browser (Float32 WebGL) fails for larger vocabularies — Gaussian elimination needs ~15 significant digits, but Float32 only provides 7. Result: "learning learning learning learning..."

Solution: Train offline with Float64 (NumPy), export weights as gzipped Float16. Full-precision training, reduced-precision inference — same principle as LLM quantization.

## Run locally

Just open `index.html` in any modern browser. No build step, no server needed. The 5.8MB file contains everything: model weights, corpus, and UI.

```bash
git clone https://github.com/pmmathias/krr-chat.git
cd krr-chat
open index.html
```

## Performance

| Metric | Value |
|--------|-------|
| Retrieval corpus | 805 sentences (ArXiv + curated) |
| Generation corpus | 104 sentences, 1427 tokens, 505 unique words |
| Context window | 5 words |
| Features | 1024 dimensions (HASH=128 × 5 pos + 3 bigram) |
| RFF dimension | D=1536 |
| Training | Offline (Float64, ~1 second) |
| Top-1 accuracy | 99.8% on training data |
| File size | ~5.8 MB (self-contained HTML) |

## Connection to the blog

Interactive companion to [Eigenwerte & KI](https://ki-mathias.de/eigenwerte.html) ([English](https://ki-mathias.de/en/eigenvalues.html)) on [ki-mathias.de](https://ki-mathias.de). The blog post includes a detailed section explaining how the chatbot works.

## License

MIT

## Author

[KI-Mathias](https://ki-mathias.de) (Mathias Leonhardt) — in collaboration with Claude Code (Anthropic)

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

**Kalle** is a German buddy chatbot that chats about food, hobbies, music, feelings, weather, and simple math — with multi-turn context awareness, emergent math validation, and honest scope boundaries.

## Kalle's architecture (new)

```
User: "Fisch" (after Kalle talked about pizza)

1. queryText = lastBotTurn + " " + userInput
   → "mein lieblingsessen ist pizza ... fisch"

2. BoW+IDF embedding (32-dim Word2Vec)
   → semantic query vector

3. Combined scoring: α × keyword + (1-α) × semantic
   → best matching dialogue pair

4. Response from matched pair (word-by-word animation)

5. Prediction Comparison: KRR's top-3 prediction per word
   → green (agreed) / yellow (would have differed)
```

**No stemming. No substring expansion. No keyword hacks.** Just full-text BoW+IDF matching against curated dialogue pairs. The IDF weighting naturally down-weights function words.

## Emergent behavior

Things **nobody programmed** that emerge from pure pattern matching:

- **Math validation**: Kalle asks "what is 3+5?", user says "8" → "correct! 3+5=8" — no code checks the math, it's pure pair matching
- **Insult immunity**: Profanity is out-of-vocabulary → invisible to the model → no filter needed
- **Typo robustness**: OOV words from typos are silently ignored, remaining words still match
- **Honest limits**: Low confidence → "I can only talk about food, hobbies, music, feelings, weather and math"

## The journey: Less is more

| Iteration | Pairs | Encoding | Top-1 | Lesson |
|---|---|---|---|---|
| V1 (original) | 57 | Hash (128 buckets) | 99.8% | Perfect but narrow |
| MEGA (mass) | 4301 | Word2Vec + hacks | 34.9% | More data = worse! |
| **FINAL (curated)** | **2113** | **Word2Vec (32-dim)** | **65.4%** | **Curated > generated** |

This mirrors what OpenAI, Anthropic and Google learned: **data quality beats data quantity**. Instruction-tuning datasets are carefully curated — just like Kalle's corpus.

## KRR Chat architecture (original)

```
User question → Keyword extraction → Retrieve best matching sentence
→ Use as seed context → KRR generation (word by word) → Color-coded output
```

- **System 1 — Retrieval**: 805 sentences (ArXiv + curated eigenvector content)
- **System 2 — Generation**: KRR language model trained on 104 sentences (Float64 offline)

## The math behind it

```
Word2Vec embeddings    → Feature map φ(x) (replaces hash encoding)
Random Fourier Features → Kernel approximation k(x,x') ≈ z(x)ᵀz(x')
Gaussian elimination   → Solve (ZᵀZ + λI)W = ZᵀY
Ridge parameter λ      → Regularization = early stopping
Eigenvalues of ZᵀZ    → What the model learns (signal) vs ignores (noise)
```

## Run locally

```bash
git clone https://github.com/pmmathias/krr-chat.git
cd krr-chat
open index.html       # KRR Chat (EN)
open kalle-chat.html  # Kalle (DE)
```

No build step, no server needed. Self-contained HTML files with embedded model weights.

## Connection to the blog

Interactive companion to [Eigenwerte & KI](https://ki-mathias.de/eigenwerte.html) ([English](https://ki-mathias.de/en/eigenvalues.html)) on [ki-mathias.de](https://ki-mathias.de). The [Deep-Dive blog post](https://ki-mathias.de/krr-chat-erklaert.html) explains every component in detail.

## License

MIT

## Author

[KI-Mathias](https://ki-mathias.de) (Mathias Leonhardt) — in collaboration with Claude Code (Anthropic)

# KRR Chat

**A mini language model built on Kernel Ridge Regression — no neural networks, just eigenvalues.**

[**Try it live**](https://ki-mathias.de/krr-chat.html)

## What is this?

A word-level language model that runs entirely in your browser. It predicts the next word using Kernel Ridge Regression with Random Fourier Features — the same mathematical chain described in the [Eigenvalues & AI](https://ki-mathias.de/eigenwerte.html) blog post.

**No neural networks.** No backpropagation. No gradient descent. Just matrix multiplication, a Gaussian kernel, and eigenvalues.

## How it works

```
User input → Word tokenization → Hash-based positional features
→ Random Fourier Features (approximate Gaussian kernel)
→ Solve (Z^TZ + λI) W = Z^TY
→ Predict next word → Repeat
```

1. **Corpus** (~2500 word tokens) is split into 5-word context windows
2. **Features**: Positional word hashes + bigram hashes (1024 dimensions)
3. **Random Fourier Features** (D=2048) approximate the Gaussian kernel, reducing O(n³) to O(n·D²)
4. **GPU acceleration**: RFF projection and Z^TZ computed on GPU via TensorFlow.js WebGL — works on any GPU (Intel, AMD, Apple, NVIDIA)
5. **Linear solve**: 2048×2048 Gaussian elimination on CPU
6. **Generation**: Top-K sampling with repetition penalty, stops at sentence boundaries

## The math behind it

This is the chain from the blog post:

```
Projection → Iterative Residual Correction → Eigenvalues → Regularization
→ Kernel Trick (X^TX → K) → Random Fourier Features → Word Prediction
```

- **Kernel Ridge Regression** replaces neural networks
- **Random Fourier Features** (Rahimi & Recht, 2007) break the O(n³) scaling barrier
- **Early stopping ≈ Ridge Regression** (λ ≈ 1/n) — regularization without a penalty term
- **Eigenvalues** of the kernel matrix determine what the model learns (signal) and ignores (noise)

The Representer Theorem guarantees: the optimal KRR solution is a weighted sum of kernel evaluations at the training points. **The training data IS the model.**

## Run locally

Just open `index.html` in any modern browser. No build step, no dependencies beyond TensorFlow.js (loaded from CDN).

```bash
git clone https://github.com/pmmathias/krr-chat.git
cd krr-chat
open index.html    # or python3 -m http.server 8080
```

Training takes 3-15 seconds depending on your GPU.

## Performance

| Metric | Value |
|--------|-------|
| Corpus | ~2500 word tokens, ~500 unique words |
| Context | 5 words |
| Features | 1024 dimensions (hash-based) |
| RFF | D=2048 random Fourier features |
| Training | 3-15s (GPU via WebGL) |
| Top-1 accuracy | ~35-45% (on held-out test set) |
| Top-3 accuracy | ~55-65% |
| Vocab | ~500 words (eigenvalues, ML, quantum mechanics) |

For comparison: random baseline would be ~0.2% top-1 accuracy on 500 classes.

## Why not a neural network?

This project is a proof of concept for the thesis of [Eigen-KI](https://ki-mathias.de/eigenwerte.html): **Kernel Ridge Regression and neural networks are equally powerful in theory** (Representer Theorem + Universal Approximation Theorem). The difference is computational efficiency, not mathematical power.

A Transformer with the same training data would be faster and more accurate — but it would hide the eigenvalue structure that makes everything work. This model makes it visible.

## Connection to the blog

This is the interactive companion to the blog post [Eigenwerte & KI](https://ki-mathias.de/eigenwerte.html) ([English](https://ki-mathias.de/en/eigenvalues.html)) on [ki-mathias.de](https://ki-mathias.de).

## License

MIT

## Author

[KI-Mathias](https://ki-mathias.de) (Mathias Leonhardt) — in collaboration with Claude Code (Anthropic)

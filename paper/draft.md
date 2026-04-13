# Kalle: A Transparent Language Model Using Kernel Ridge Regression with Random Fourier Features

**Mathias Leonhardt**
KI-Mathias / pmagentur.com, Hamburg, Germany
mathias@ki-mathias.de

---

## Abstract

We present Kalle, a fully client-side bilingual chatbot that performs next-word prediction and retrieval-augmented generation (RAG) using Kernel Ridge Regression (KRR) with Random Fourier Features — without neural networks, backpropagation, or gradient descent. The system makes the mathematical foundations of modern language models *transparent*: every component (feature maps, kernel approximation, regularization, eigenvalue spectra) is inspectable and maps to a concept from linear algebra. Kalle serves 2,178 curated Q&A pairs in German and English, runs entirely in the browser via WebGL GPU acceleration (TensorFlow.js), and achieves functional conversational ability including multi-turn context, bilingual routing without language detection, and RAG over blog content — all through corpus design rather than architectural complexity. We demonstrate that the mathematical structure underlying large language models can be made visible, interactive, and pedagogically useful at a scale of ~18M parameters and a single closed-form matrix solve. The system, source code, and training data are publicly available.

**Keywords:** kernel ridge regression, random fourier features, language model, educational NLP, browser-based ML, retrieval-augmented generation

---

## 1. Introduction

Modern large language models (LLMs) are powerful but opaque. GPT-4, Claude, and Gemini predict the next token given context — but the mechanism (attention layers, backpropagation over billions of parameters, RLHF) is inaccessible to anyone without specialized ML infrastructure. This opacity is a pedagogical problem: students learn *that* LLMs work, not *why*.

We ask: **Can the core task of language modeling — predict the next word given context — be solved with transparent mathematics at a scale where every component remains inspectable?**

We present Kalle, a system that answers this question affirmatively. Kalle is a bilingual (German/English) chatbot that combines:

1. **Kernel Ridge Regression** (KRR) with a closed-form solution — no gradient descent, no epochs, no convergence monitoring
2. **Random Fourier Features** (RFF; Rahimi & Recht, 2007) — approximating the Gaussian kernel without computing the full kernel matrix
3. **Word2Vec embeddings** (Mikolov et al., 2013) — providing collision-free word representations
4. **BoW+IDF matching** — retrieval over curated Q&A pairs using weighted bag-of-words with inverse document frequency
5. **RAG** — retrieval-augmented generation over blog content, without an LLM

The entire system runs in the browser (WebGL GPU via TensorFlow.js), requires no server, and is deployed as a single self-contained HTML file (~56 MB). Every prediction is accompanied by a **color-coded comparison** showing whether the KRR model agrees with the corpus (green) or would have chosen differently (yellow) — making the boundary between memorization and generalization *visible*.

Kalle is not a competitor to production LLMs. It is a **didactic instrument** that makes the mathematical structure behind language models transparent — eigenvalues, kernel functions, regularization, feature maps — on a scale where a student can follow every step.

---

## 2. System Architecture

### 2.1 Training Pipeline (Offline, Python, Float64)

Training proceeds in five steps, all computed once in closed form:

**Step 1: Corpus preparation.** The training corpus consists of 2,178 curated dialog pairs in the format `du: {user} . bot: {response} .` The full token sequence (5× repeated for training signal) contains 322,660 tokens over a vocabulary of 2,977 words.

**Step 2: Word2Vec embeddings.** We train 32-dimensional Word2Vec embeddings (Mikolov et al., 2013) on the corpus using CBOW with window size 8 and 20 epochs (gensim). Each word receives a unique vector; similar words (e.g., "pizza" / "pasta") have similar vectors. This eliminates hash collisions that plagued an earlier version using modular hashing.

**Step 3: Context encoding.** Each training sample encodes a context of CTX=24 consecutive words as a feature vector φ(x) ∈ ℝ^768 by concatenating position-weighted Word2Vec embeddings:

$$\phi(x) = [\alpha_1 \cdot e_{w_1}, \alpha_2 \cdot e_{w_2}, \ldots, \alpha_{24} \cdot e_{w_{24}}]$$

where $e_{w_i}$ is the 32-dim embedding of word $w_i$ and $\alpha_i = 0.4 + 0.6 \cdot i/23$ assigns linearly increasing weight to more recent context positions.

**Step 4: Random Fourier Features.** Following Rahimi & Recht (2007), we approximate the Gaussian kernel $k(x,x') = \exp(-\|x-x'\|^2/2\sigma^2)$ with $\sigma=1.5$ using D=6,144 random features:

$$z(x) = \sqrt{2/D} \cdot \cos(\phi(x) \cdot \omega + b)$$

where $\omega \sim \mathcal{N}(0, 1/\sigma^2)^{768 \times 6144}$ and $b \sim \text{Uniform}(0, 2\pi)^{6144}$ are drawn once with a fixed random seed and never modified.

**Step 5: KRR solve.** We solve the ridge regression system in closed form:

$$W = (Z^\top Z + \lambda I)^{-1} Z^\top Y$$

where $Z$ is the RFF matrix (computed in streaming chunks of 10,000 to avoid OOM), $Y$ is the one-hot target matrix, and $\lambda = 10^{-6}$. $W \in \mathbb{R}^{6144 \times 2977}$ (~18.1M parameters) is the only learned parameter. Total solve time: ~2 seconds on a single CPU core.

### 2.2 Inference Pipeline (Online, Browser, WebGL GPU)

At inference time, only three operations execute per word:

$$\text{next\_word} = \text{argmax}(z(\text{context})^\top \cdot W)$$

This is a single matrix-vector multiplication, running in <1ms on WebGL GPU.

### 2.3 Answer Retrieval (BoW+IDF Matching)

Kalle does not generate free-form text from the KRR model. Instead, it **retrieves** the best-matching Q&A pair from the curated corpus using a combined scoring function:

$$\text{score} = 0.65 \cdot \text{kw}(q, p) + 0.35 \cdot \text{sem}(q, p)$$

where $\text{kw}(q, p)$ is the IDF-weighted keyword overlap between query $q$ and pair user-side $p$, and $\text{sem}(q, p)$ is the cosine similarity between their 32-dimensional BoW+IDF sentence embeddings.

The KRR model then provides a **word-by-word prediction comparison**: for each word in the retrieved answer, the KRR model predicts its top-3 candidates. If the corpus word appears in the KRR top-3, it is rendered green (model agrees — verbatim knowledge); otherwise yellow (model would have chosen differently — the corpus answer diverges from what the model would generate).

### 2.4 Retrieval-Augmented Generation (RAG)

For domain-specific questions about blog content, Kalle implements a lightweight RAG pipeline:

1. **Chunk retrieval:** The blog is pre-segmented into 29 chunks (~58 words each). User query keywords are matched against chunk keywords (including English translations).
2. **Prompt construction:** The best chunk is injected as context: `kontext {chunk_text} frage {user_query}`.
3. **Pair matching:** The augmented prompt is matched against Q&A pairs that were trained on this format, using the same BoW+IDF scoring.

This achieves RAG functionality without any LLM — the "generation" is retrieval over context-conditioned pairs.

### 2.5 Multi-Turn Context

Following AIML's `<that>` mechanism (Wallace, 2009), Kalle concatenates the full previous bot response (`lastBotTurn`) with the current user input before matching. This provides multi-turn context without explicit state tracking:

```
Kalle: "my favorite food is pizza. what about you?"
User:  "Fish"
Query: "my favorite food is pizza ... fish"
Match: pair "favorite food pizza fish" → contextual response
```

The corpus includes explicit follow-up pairs whose user-sides contain context keywords from expected preceding turns.

---

## 3. Design Decisions and Lessons Learned

### 3.1 Data Quality > Data Quantity

The most instructive finding was the relationship between corpus size and quality:

| Iteration | Pairs | Encoding | Top-1 Accuracy |
|-----------|-------|----------|---------------|
| V1 | 57 | Hash (128 buckets) | 99.8% |
| V2 (mass expansion) | 4,301 | Word2Vec + heuristics | 34.9% |
| V3 (curated) | 2,178 | Word2Vec (32-dim) | 63.5% |

Blind corpus expansion *destroyed* quality because similar patterns competed in the feature space. This mirrors findings in instruction tuning for large models (Wei et al., 2022; Köpf et al., 2023), where curated datasets consistently outperform larger but noisier ones.

### 3.2 Hash Encoding → Word2Vec

The original system used `hash(word) % 128`, mapping 505 words onto 128 buckets (~4 collisions/bucket). Replacing this with 32-dimensional Word2Vec embeddings eliminated all collisions and reduced the feature dimension from 3,072 to 768 while providing semantic similarity between related words.

### 3.3 Hyperparameter Choices

- **σ = 1.5** (kernel bandwidth): Controls the trade-off between memorization (small σ, only very similar contexts match) and generalization (large σ, distant contexts influence each other). Chosen empirically to maximize Top-1 accuracy on the training set.
- **λ = 10⁻⁶** (regularization): Deliberately small — the training set is dense enough that stronger regularization hurts prediction quality.
- **D = 6,144** (RFF dimension): 8× oversampling of the 768-dimensional feature space, providing sufficient kernel approximation quality.

### 3.4 Architectural Properties (Not "Emergent")

Several properties arise deterministically from the BoW+IDF design:

- **Bilingual routing:** English and German words are distinct tokens with distinct IDF weights. "eigenvalue" (high IDF) matches English pairs; "Eigenwert" (high IDF) matches German pairs. No language detection code needed.
- **Typo robustness:** Out-of-vocabulary words (including typos) are silently ignored. Remaining in-vocabulary words suffice for matching.
- **Insult immunity:** Profanity not in the vocabulary is invisible to the model.
- **Math validation (illusory):** The pair `plus 3 5 8` matches when the user says "8" after a "3+5" question — pure pattern matching, not arithmetic.

---

## 4. Evaluation

### 4.1 Automated Regression Testing

We developed a Playwright-based test suite with 34 scenarios across 6 categories:

| Category | Scenarios | Description |
|----------|-----------|-------------|
| Greeting | 3 | Opening/closing phrases |
| Food | 3 | Topic-specific responses |
| Emotion | 3 | Empathetic responses |
| Math | 4 | Arithmetic validation |
| Meta | 3 | Self-description, scope |
| Multi-turn | 14 | Context preservation across turns |
| Edge cases | 4 | Typos, insults, long input |

### 4.2 RAG Evaluation

We tested the RAG pipeline with a 4-turn multi-topic conversation:

| Turn | Query | Language | Retrieved Chunk | kwRaw | Result |
|------|-------|----------|----------------|-------|--------|
| 1 | "What are eigenvalues?" | EN | Das Glasperlenspiel | 36.9 | ✓ Correct, English |
| 2 | "und was hat das mit pagerank zu tun?" | DE | Der Surfer | 44.3 | ✓ Correct, cross-topic |
| 3 | "und mit radiosity? auf deutsch bitte" | DE | Licht, das von Wänden springt | 42.0 | ✓ Correct, language switch |
| 4 | "wie hängt das mit KRR zusammen?" | DE | Was bedeutet das konkret? | 8.5 | △ Acceptable but weak |

The system correctly retrieved relevant chunks in 3/4 cases and maintained conversational context across language switches and topic transitions. The 4th turn shows a known limitation: abstract meta-questions dilute the keyword signal.

### 4.3 Prediction Comparison Statistics

Across the test conversations, the KRR model's Top-3 prediction matched the corpus word 39–50% of the time (average ~44%). This means the model has learned enough structure to agree with roughly half of all words — while transparently showing where it would diverge.

---

## 5. Related Work

**Kernel methods for NLP.** String kernels (Lodhi et al., 2002) and spectrum kernels (Leslie et al., 2002) have been used for text classification, but not for autoregressive language modeling. Our work applies KRR to next-word prediction, a task typically reserved for neural models.

**Random Fourier Features.** Rahimi & Recht (2007) introduced RFF for scalable kernel approximation, receiving the NeurIPS Test of Time Award in 2017. We apply RFF to language modeling, demonstrating that the kernel approximation is sufficient for a functional chatbot at moderate scale.

**Educational NLP systems.** The Teaching NLP workshop series (Jurgens et al., 2024) has highlighted the need for interactive tools that make NLP concepts tangible. Kalle contributes by providing a complete, inspectable language model where every component maps to a mathematical concept.

**Browser-based ML.** Ma et al. (2019) evaluated TensorFlow.js for deep learning in the browser. Kalle demonstrates that non-neural ML can also run effectively in the browser, with the advantage of mathematical transparency.

**Retrieval-augmented generation.** Lewis et al. (2020) introduced RAG for knowledge-grounded generation with neural retrievers and generators. Kalle implements a RAG-like pipeline using keyword matching and curated Q&A pairs — demonstrating that the *pattern* of RAG (retrieve context, condition generation) is separable from neural architecture.

**AIML and pattern-based chatbots.** Wallace (2009) described AIML's pattern matching with `<that>` tags for context. Kalle's `lastBotTurn` concatenation is a continuous-space analog of this mechanism.

---

## 6. Limitations

- **No free-form generation.** Kalle retrieves pre-written answers; it cannot compose novel sentences.
- **Vocabulary-bound.** Words not in the 2,977-word vocabulary are invisible. This includes many proper nouns and technical terms.
- **Scaling ceiling.** KRR's closed-form solution requires O(D²) memory and O(D³) compute for the solve. At D=6,144, this is ~288 MB and ~2 seconds. Scaling to D=100,000 (needed for significantly larger vocabularies) would require iterative methods, losing the closed-form advantage.
- **File size.** The self-contained HTML file is ~56 MB, at the edge of practical web deployment.
- **Evaluation scope.** Our evaluation is primarily qualitative and scenario-based. Formal metrics (perplexity, BLEU, human preference) are left for future work.

---

## 7. Conclusion

Kalle demonstrates that the mathematical foundations of language models — kernel functions, eigenvalues, regularization, feature maps — can be made transparent, interactive, and pedagogically useful. The system achieves functional conversational ability, bilingual routing, multi-turn context, and retrieval-augmented generation using three equations (RFF, KRR solve, matrix-vector prediction) and a curated corpus of 2,178 pairs.

The key finding is not that KRR competes with neural language models — it does not. The key finding is that the *mathematical structure* is the same: feature maps, regularization, eigenvalue spectra controlling what is learned versus ignored. Making this structure visible is the contribution.

**Demo:** https://pmmathias.github.io/krr-chat/
**Source:** https://github.com/pmmathias/krr-chat
**Blog:** https://ki-mathias.de/en/krr-chat-explained.html

---

## References

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.

Lodhi, H., Saunders, C., Shawe-Taylor, J., Cristianini, N., & Watkins, C. (2002). Text Classification using String Kernels. *JMLR*, 3, 419–444.

Jurgens, D., et al. (2024). Proceedings of the Sixth Workshop on Teaching NLP. *ACL 2024*.

Köpf, A., Kilcher, Y., von Rütte, D., et al. (2023). OpenAssistant Conversations — Democratizing Large Language Model Alignment. *NeurIPS Datasets and Benchmarks*.

Leslie, C., Eskin, E., & Noble, W. S. (2002). The Spectrum Kernel: A String Kernel for SVM Protein Classification. *PSB*.

Ma, Y., Xiang, D., Zheng, S., Tian, D., & Liu, Z. (2019). Moving Deep Learning into Web Browser: How Far Can We Go? *WWW*.

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR Workshop*.

Rahimi, A. & Recht, B. (2007). Random Features for Large-Scale Kernel Machines. *NeurIPS*.

Wallace, R. (2009). The Anatomy of A.L.I.C.E. In *Parsing the Turing Test*, Springer.

Wei, J., et al. (2022). Finetuned Language Models Are Zero-Shot Learners. *ICLR*.

---

## Appendix A: Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| CTX | 24 | Long enough for multi-turn context |
| EMB_DIM | 32 | Word2Vec dimension |
| FEAT | 768 (24 × 32) | Concatenated context embeddings |
| D | 6,144 | RFF dimension (8× oversampling) |
| σ | 1.5 | Kernel bandwidth (empirical) |
| λ | 10⁻⁶ | Regularization (small, dense training set) |
| REPEAT | 5 | Corpus repetition for training signal |
| α | 0.65 | Keyword weight in combined scoring |

## Appendix B: Corpus Statistics

| Metric | Value |
|--------|-------|
| Total pairs | 2,178 |
| German pairs | ~1,086 |
| English pairs | ~1,092 |
| RAG context pairs | 55 |
| Vocabulary | 2,977 words |
| Avg. response length | 20.2 words |
| RAG chunks | 29 |
| Avg. chunk length | 58 words |
| Training tokens (5× repeat) | 322,660 |
| Training samples (N) | 322,636 |
| Model parameters (W) | 18.1M |
| File size (deployed) | 56 MB |

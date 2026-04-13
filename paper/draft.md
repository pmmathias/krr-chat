# Kalle: A Transparent Language Model Using Kernel Ridge Regression with Random Fourier Features

**Mathias Leonhardt**
KI-Mathias / pmagentur.com, Hamburg, Germany
mathias-leonhardt@gmx.de

---

## Abstract

We present Kalle, a fully client-side bilingual chatbot that performs next-word prediction and retrieval-augmented generation (RAG) using Kernel Ridge Regression (KRR) with Random Fourier Features — without neural networks, backpropagation, or gradient descent. The system makes the mathematical foundations of modern language models *transparent*: the same eigenvalue structure that governs convergence of the Neumann series, stability of PageRank, light transport in radiosity, and learning dynamics of neural networks is exposed in an interactive, inspectable system. Kalle serves 2,178 curated Q&A pairs bilingually, runs entirely in the browser via WebGL GPU (TensorFlow.js), and achieves functional conversational ability including multi-turn context and RAG — all through a single closed-form matrix solve. We derive the complete mathematical chain from projection to kernel methods and show that the Neural Tangent Kernel theorem (Jacot et al., 2018) makes this connection rigorous: in the infinite-width limit, a neural network *becomes* KRR.

---

## 1. Introduction

Modern large language models solve a deceptively simple task: given a sequence of words, predict the next one. The mechanism — transformer attention over billions of parameters, trained via backpropagation with RLHF — is powerful but opaque. A student can learn *that* GPT-4 works; understanding *why* requires infrastructure and expertise that most curricula cannot provide.

We ask: **Can next-word prediction be solved with mathematics transparent enough that every component maps to a textbook concept — while remaining functional enough to demonstrate the principles at work?**

We present Kalle, a system that answers affirmatively. The mathematical structure is not a simplification of neural language models — it is the *same* structure, made visible. The unifying thread is **eigenvalues**: they control convergence of iterative methods (§2), determine what a model learns versus ignores (§3), govern the stability of PageRank and radiosity (§4), and — through the Neural Tangent Kernel — connect KRR to neural networks rigorously (§5).

---

## 2. Mathematical Foundations

### 2.1 From Projection to the Normal Equation

The simplest prediction problem: given data matrix $X$ and targets $\mathbf{y}$, find the coefficient vector $\mathbf{c}$ that minimizes $\|X\mathbf{c} - \mathbf{y}\|^2$. The solution is the orthogonal projection of $\mathbf{y}$ onto the column space of $X$:

$$\hat{\mathbf{y}} = X(X^\top X)^{-1} X^\top \mathbf{y}$$

This requires solving the **normal equation**:

$$X^\top X\,\mathbf{c} = X^\top \mathbf{y} \quad\implies\quad \mathbf{c} = (X^\top X)^{-1} X^\top \mathbf{y}$$

### 2.2 Iterative Solution and the Neumann Series

Instead of inverting $X^\top X$ directly, we can iterate. Starting from $\hat{\mathbf{y}}_0 = \mathbf{0}$:

$$\hat{\mathbf{y}}_{n+1} = \hat{\mathbf{y}}_n + X^\top(\mathbf{y} - X\hat{\mathbf{y}}_n)$$

The residual after $n$ steps satisfies:

$$\mathbf{r}_n = (I - X^\top X)^n \cdot \mathbf{y}$$

This converges when all eigenvalues of $(I - X^\top X)$ have magnitude less than 1. The limit is the **Neumann series** — the matrix generalization of the geometric series $\sum q^k = 1/(1-q)$:

$$\sum_{k=0}^{\infty} A^k = (I - A)^{-1} \quad\text{when } \rho(A) < 1$$

where $\rho(A)$ denotes the spectral radius. The iteration converges exactly to the closed-form solution (1). This is not coincidence — it is the same equation approached from two directions.

### 2.3 Eigenvalues Control Everything

Let $\mu_1, \ldots, \mu_n$ be the eigenvalues of $X^\top X$ with corresponding eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$. The spectral decomposition gives:

$$A^n = V \Lambda^n V^{-1}$$

The component of the solution along the $i$-th eigenvector is retained after $n$ iterations by the factor:

$$f_i^{\,\mathrm{iter}}(n) = 1 - (1 - \mu_i)^n$$

This has profound consequences:
- **Large $\mu_i$** (strong signal): $f_i \to 1$ quickly — learned in few iterations
- **Small $\mu_i$** (weak signal/noise): $f_i \approx n\mu_i$ — learned slowly
- **Early stopping at step $n$**: components with $\mu_i \ll 1/n$ are suppressed

Early stopping is not a heuristic — it is a *spectral filter* on the eigenvalue decomposition.

### 2.4 Ridge Regression: The Closed-Form Spectral Filter

Instead of stopping early, we can add an explicit penalty. Ridge regression solves:

$$\mathbf{c}_\lambda = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}$$

The filter factor for the $i$-th eigencomponent becomes:

$$f_i^{\,\mathrm{ridge}}(\lambda) = \frac{\mu_i}{\mu_i + \lambda}$$

**Theorem (Early Stopping ≈ Ridge Regression).** The iterative filter (5) and the ridge filter (7) produce equivalent spectral filtering with the correspondence $\lambda \approx 1/n$. More iterations = less regularization. Fewer iterations = stronger smoothing. Both separate signal (large $\mu_i$) from noise (small $\mu_i$) using the eigenvalue spectrum.

| Eigenvalue $\mu_i$ | Ridge factor $\frac{\mu_i}{\mu_i + \lambda}$ | Iter. factor $1-(1-\mu_i)^n$ | Interpretation |
|---|---|---|---|
| 1.0 | 0.999 | 1.000 | Signal: passes through |
| 0.1 | 0.909 | 0.651 | Moderate: partially filtered |
| 0.01 | 0.500 | 0.096 | Noise: heavily suppressed |
| 0.001 | 0.091 | 0.010 | Noise: nearly eliminated |

*(Table computed with $\lambda = 0.01$, $n = 10$.)*

### 2.5 The Kernel Trick: From Finite to Infinite Dimensions

Real-world data is rarely linear. The kernel trick lifts data into a higher-dimensional feature space $\phi: \mathbb{R}^d \to \mathcal{H}$ where nonlinear patterns become linear, without ever computing $\phi$ explicitly. A **kernel function** computes the inner product directly:

$$k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$$

The **Gaussian (RBF) kernel**

$$k(x, z) = \exp\!\left(-\frac{\|x - z\|^2}{2\sigma^2}\right)$$

has an *infinite-dimensional* feature space, yet the kernel matrix $K \in \mathbb{R}^{n \times n}$ remains finite. The iterative update generalizes to:

$$\hat{\mathbf{y}}_{n+1} = \hat{\mathbf{y}}_n + K(\mathbf{y} - \hat{\mathbf{y}}_n)$$

Replace $X^\top X$ with $K$. Same algorithm, same convergence analysis, same eigenvalue structure — but now with infinite expressive power.

**Kernel Ridge Regression** combines (6) and (8):

$$\boldsymbol{\alpha} = (K + \lambda I)^{-1} \mathbf{y}$$

Prediction for a new point uses the **Representer Theorem** (Kimeldorf & Wahba, 1970): the optimal solution has the form

$$f(x_{\ast}) = \sum_{i=1}^{n} \alpha_i\, k(x_i, x_{\ast})$$

The training data *is* the model. Each training point "votes" for the prediction, weighted by kernel similarity.

### 2.6 Random Fourier Features: Making Kernels Scalable

The kernel matrix $K$ is $n \times n$. For $n = 322636$ training samples, this is infeasible. Rahimi & Recht (2007) showed that shift-invariant kernels can be approximated via random projection, using **Bochner's theorem**: a bounded continuous shift-invariant kernel is the Fourier transform of a non-negative measure.

For the Gaussian kernel, this yields:

$$z(x) = \sqrt{\frac{2}{D}} \cos(\omega^\top x + b)$$

where $\omega \sim \mathcal{N}(0, \sigma^{-2} I)$ and $b \sim \text{Uniform}(0, 2\pi)$. The key property:

$$z(x)^\top z(x') \approx k(x, x')$$

The approximation improves with $D$. With RFF, the kernel ridge regression solution (11) becomes a standard linear system in the feature space:

$$W = (Z^\top Z + \lambda I)^{-1} Z^\top Y$$

where $Z \in \mathbb{R}^{N \times D}$ is the RFF feature matrix. This is Equation (6) again — the normal equation with ridge regularization — but operating on random Fourier features instead of raw data. The eigenvalues of $Z^\top Z$ control convergence, regularization, and what the model learns. **The same equation, the same eigenvalue structure, at every level of the hierarchy.**

---

## 3. The Unified Chain: One Equation, Four Applications

The equation $(I - A)^{-1}\mathbf{b} = \sum_{k=0}^{\infty} A^k \mathbf{b}$ appears in four seemingly unrelated domains. In each case, convergence is governed by the eigenvalues of $A$.

### 3.1 Radiosity: Light Bouncing Between Walls

The radiosity equation describes global illumination in computer graphics:

$$\mathbf{B} = \mathbf{E} + \rho F \mathbf{B} \quad\implies\quad \mathbf{B} = (I - \rho F)^{-1} \mathbf{E} = \sum_{k=0}^{\infty} (\rho F)^k \mathbf{E}$$

where $\mathbf{E}$ is direct emission, $\rho$ is reflectivity, and $F$ is the form factor matrix. Each term $(\rho F)^k \mathbf{E}$ represents one additional light bounce. The matrix $\rho F$ is **substochastic** (column sums $< 1$) because surfaces absorb energy. Convergence is guaranteed when $\rho({\rho F}) < 1$.

To make $\rho F$ stochastic (column sums $= 1$), we add an **absorber dimension** — a virtual surface that captures all unabsorbed light. The resulting stochastic matrix describes a Markov chain whose stationary distribution is the dominant eigenvector with eigenvalue 1.

### 3.2 PageRank: Importance Spreading Through Links

Google's PageRank (Page et al., 1998) applies the same structure to the web. The **Google matrix** is:

$$G = d \cdot M + \frac{1-d}{n}\,\mathbf{1}\mathbf{1}^\top$$

where $M$ is the column-stochastic link matrix, $d \approx 0.85$ is the damping factor (analogous to reflectivity $\rho$ in radiosity), and $(1-d)/n$ is the teleportation probability (analogous to the absorber).

The PageRank vector is the dominant eigenvector of $G$:

$$\mathbf{r}^{(k+1)} = G\,\mathbf{r}^{(k)} \longrightarrow \mathbf{r}^{\ast}$$

with $G\mathbf{r}^{\ast} = \mathbf{r}^{\ast}$ at convergence ($k \to \infty$).

The damping factor ensures $G$ is stochastic and primitive, guaranteeing a unique dominant eigenvector (Perron-Frobenius theorem). The **spectral gap** $1 - |\lambda_2|$ determines convergence speed — larger gap means faster convergence. With $d = 0.85$, $|\lambda_2| \le 0.85$, giving reliable convergence in $\sim$50–100 iterations even for billions of pages.

### 3.3 KRR Language Model: Predictions Spreading Through Similarities

Our language model has the same structure. The kernel matrix $K$ measures pairwise context similarities. The regularized prediction:

$$\boldsymbol{\alpha} = (K + \lambda I)^{-1} \mathbf{y}$$

can be rewritten as a Neumann series when $\rho(K) < 1 + \lambda$:

$$\boldsymbol{\alpha} = \frac{1}{\lambda}\sum_{k=0}^{\infty} \left(-\frac{K}{\lambda}\right)^k \mathbf{y}$$

Each term represents one "bounce" of prediction influence through the kernel similarity graph — analogous to light bouncing between walls (radiosity) or importance spreading through links (PageRank).

### 3.4 Summary: Same Equation, Same Eigenvalues

| Domain | Matrix $A$ | "Source" $\mathbf{b}$ | Eigenvalue role |
|--------|-----------|----------------------|-----------------|
| Regression | $X^\top X$ | $X^\top \mathbf{y}$ | Convergence speed |
| Radiosity | $\rho F$ | $\mathbf{E}$ (emission) | Light distribution |
| PageRank | $G$ | uniform start | Webpage ranking |
| KRR | $K/\lambda$ | $\mathbf{y}/\lambda$ | What model learns |

All four solve $(I - A)\mathbf{x} = \mathbf{b}$ via the Neumann series. All four converge when $\rho(A) < 1$. All four are controlled by eigenvalues.

---

## 4. The Neural Network Connection

### 4.1 Universal Approximation

Three results establish the power of our approach:

1. **Representer Theorem** (Kimeldorf & Wahba, 1970): The optimal KRR solution has the form $f^*(x) = \sum_i \alpha_i k(x_i, x)$ — finitely many coefficients regardless of the kernel space dimension.

2. **Universal Approximation** (Cybenko, 1989): A neural network with one hidden layer can approximate any continuous function to arbitrary precision.

3. **Gaussian Kernel Universality** (Micchelli et al., 2006): KRR with the Gaussian kernel can approximate any continuous function on a compact set.

Both KRR and neural networks are **dense in $C(K)$** — the space of continuous functions on a compact set. They are equally powerful in principle.

### 4.2 Neural Tangent Kernel: From Metaphor to Theorem

The connection is not merely analogical. Jacot et al. (2018) proved that in the limit of infinite width with suitable initialization, a neural network **becomes** kernel ridge regression. The **Neural Tangent Kernel** (NTK) is:

$$\Theta(x, x') = \left\langle \frac{\partial f(x; \theta)}{\partial \theta}, \frac{\partial f(x'; \theta)}{\partial \theta} \right\rangle$$

In the infinite-width limit, $\Theta$ converges to a deterministic kernel, and gradient descent on the neural network converges to the KRR solution with this kernel. The eigenvalues of $\Theta$ determine what the network learns first (large eigenvalues) and what it learns last or ignores (small eigenvalues).

This is not a metaphor. Kalle's KRR system and a neural language model are governed by the **same eigenvalue structure** — Kalle simply makes it visible.

---

## 5. System Implementation

### 5.1 Training Pipeline

| Step | Operation | Output |
|------|-----------|--------|
| 1 | Parse corpus (2,178 pairs) | Token sequence (64,532 tokens) |
| 2 | Word2Vec (gensim, 32-dim, CBOW) | Embedding matrix $E \in \mathbb{R}^{2977 \times 32}$ |
| 3 | Context encoding: $\phi(x) = [\alpha_1 e_{w_1}, \ldots, \alpha_{24} e_{w_{24}}]$ | Feature vectors $\in \mathbb{R}^{768}$ |
| 4 | RFF: $z(x) = \sqrt{2/D}\cos(\phi(x)\omega + b)$, $D=6144$ | $Z \in \mathbb{R}^{N \times 6144}$ (streaming) |
| 5 | KRR: $W = (Z^\top Z + \lambda I)^{-1} Z^\top Y$ | $W \in \mathbb{R}^{6144 \times 2977}$ (18.1M params) |
| 6 | IDF + BoW pair embeddings (32-dim) | Sentence vectors for retrieval |
| 7 | Float16 + gzip + base64 → inject into HTML | Self-contained ~56 MB file |

Total training time: ~3 minutes on a single CPU core. Solve time for step 5: ~2 seconds.

### 5.2 Inference

Prediction requires one matrix-vector multiplication per word:

$$\hat{w} = \operatorname{argmax}(z(c)^\top \cdot W)$$

where $c$ is the current context window.

Running in <1ms per word on WebGL GPU via TensorFlow.js.

### 5.3 Answer Retrieval and RAG

Kalle retrieves answers from curated Q&A pairs using combined scoring:

$$s(q, p) = 0.65 \cdot s_{\mathrm{kw}}(q, p) + 0.35 \cdot s_{\mathrm{sem}}(q, p)$$

where $s_{\mathrm{kw}}$ is IDF-weighted keyword overlap and $s_{\mathrm{sem}}$ is cosine similarity of 32-dim BoW+IDF embeddings. For domain questions, a RAG pipeline injects blog context: the query is augmented to `kontext {chunk} frage {query}` and matched against context-conditioned pairs.

### 5.4 Prediction Comparison: Making Memorization Visible

For each retrieved answer word, the KRR model predicts its top-3 candidates via Eq. (15). Words where the corpus agrees with KRR's top-3 are rendered **green** (the model has learned this pattern); disagreements are rendered **yellow** (the corpus answer diverges from what the kernel-space interpolation would produce). This makes the boundary between memorization and generalization *visible* — something production LLMs cannot offer.

### 5.5 Multi-Turn Context

Following AIML's `<that>` mechanism (Wallace, 2009), the full previous bot response is concatenated with the current input before matching. The corpus includes follow-up pairs whose user-sides contain context keywords from expected preceding turns, enabling coherent multi-turn conversation through corpus design rather than architectural state.

---

## 6. Evaluation

### 6.1 Data Quality vs. Quantity

| Iteration | Pairs | Encoding | Top-1 |
|-----------|-------|----------|-------|
| V1 (original) | 57 | Hash (128 buckets) | 99.8% |
| V2 (mass) | 4,301 | Word2Vec + heuristics | 34.9% |
| V3 (curated) | 2,178 | Word2Vec (32-dim) | 63.5% |

The curve 99.8% → 34.9% → 63.5% demonstrates that blind corpus expansion destroys quality in KRR: similar patterns compete in the feature space, averaging out to noise. This mirrors findings in LLM instruction tuning (Wei et al., 2022; Köpf et al., 2023).

### 6.2 Multi-Turn RAG Evaluation

| Turn | Query | Chunk retrieved | kwRaw | Language |
|------|-------|----------------|-------|----------|
| 1 | "What are eigenvalues?" | Das Glasperlenspiel | 36.9 | EN→EN ✓ |
| 2 | "und was hat das mit pagerank zu tun?" | Der Surfer | 44.3 | DE→EN ✓ |
| 3 | "und mit radiosity? auf deutsch bitte" | Licht, das von Wänden springt | 42.0 | DE→DE ✓ |
| 4 | "wie hängt das mit KRR zusammen?" | Was bedeutet das konkret? | 8.5 | DE→DE △ |

3/4 turns retrieve the correct chunk; the 4th (abstract meta-question) shows a known limitation of keyword-based retrieval.

### 6.3 Regression Test Suite

34 Playwright scenarios across greeting, food, emotion, math, meta, multi-turn, and edge cases. The system is tested automatically on every build.

---

## 7. Related Work

**Kernel methods for NLP.** String kernels (Lodhi et al., 2002) and spectrum kernels (Leslie et al., 2002) have been used for text classification. Our work applies KRR to autoregressive next-word prediction.

**Random Fourier Features.** Rahimi & Recht (2007) — NeurIPS Test of Time Award 2017 — introduced RFF for scalable kernel approximation via Bochner's theorem. We apply RFF to language modeling.

**Neural Tangent Kernel.** Jacot et al. (2018) proved that infinitely wide neural networks converge to kernel regression, establishing the theoretical bridge our system makes pedagogically visible.

**Browser-based ML.** Ma et al. (2019) evaluated TensorFlow.js for in-browser deep learning. Kalle demonstrates that non-neural ML also runs effectively in the browser, with the advantage of full mathematical transparency.

**RAG.** Lewis et al. (2020) introduced RAG with neural retrievers and generators. Kalle implements the RAG pattern using keyword matching and curated pairs — demonstrating that retrieval-augmented answering is separable from neural architecture.

---

## 8. Limitations

**No free-form generation.** Kalle retrieves pre-written answers; novel sentences are not composed. **Vocabulary-bound.** The 2,977-word vocabulary excludes most proper nouns and rare terms. **Scaling ceiling.** The KRR solve requires $O(D^2)$ memory ($\sim$288 MB at $D=6144$) and $O(D^3)$ compute; significantly larger models would need iterative solvers. **Evaluation scope.** Our evaluation is qualitative and scenario-based; formal metrics (perplexity, BLEU) are deferred to future work.

---

## 9. Conclusion

Kalle demonstrates that the mathematical structure underlying language models — eigenvalues, kernel functions, the Neumann series, regularization as spectral filtering — can be made transparent, interactive, and pedagogically useful. The same equation $(I - A)^{-1}\mathbf{b}$ that governs light transport (radiosity), web ranking (PageRank), and prediction (KRR) is made visible in a system where every component is inspectable. The Neural Tangent Kernel theorem establishes that this is not a simplification of neural language models — it is the same mathematics, made visible at a scale where a student can follow every step.

**Demo:** https://pmmathias.github.io/krr-chat/
**Code:** https://github.com/pmmathias/krr-chat
**Blog:** https://ki-mathias.de/en/krr-chat-explained.html

---

## References

Cybenko, G. (1989). Approximation by Superpositions of a Sigmoidal Function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.

Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural Tangent Kernel: Convergence and Generalization in Neural Networks. *NeurIPS*.

Kimeldorf, G. & Wahba, G. (1970). A Correspondence Between Bayesian Estimation on Stochastic Processes and Smoothing by Splines. *Annals of Mathematical Statistics*, 41(2), 495–502.

Köpf, A., Kilcher, Y., von Rütte, D., et al. (2023). OpenAssistant Conversations — Democratizing Large Language Model Alignment. *NeurIPS Datasets and Benchmarks*.

Leslie, C., Eskin, E., & Noble, W. S. (2002). The Spectrum Kernel: A String Kernel for SVM Protein Classification. *PSB*.

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.

Lodhi, H., Saunders, C., Shawe-Taylor, J., Cristianini, N., & Watkins, C. (2002). Text Classification using String Kernels. *JMLR*, 3, 419–444.

Ma, Y., Xiang, D., Zheng, S., Tian, D., & Liu, Z. (2019). Moving Deep Learning into Web Browser: How Far Can We Go? *WWW*.

Micchelli, C. A., Xu, Y., & Zhang, H. (2006). Universal Kernels. *JMLR*, 7, 2651–2667.

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR Workshop*.

Page, L., Brin, S., Motwani, R., & Winograd, T. (1998). The PageRank Citation Ranking: Bringing Order to the Web. *Stanford Technical Report*.

Rahimi, A. & Recht, B. (2007). Random Features for Large-Scale Kernel Machines. *NeurIPS*.

Wallace, R. (2009). The Anatomy of A.L.I.C.E. In *Parsing the Turing Test*, Springer.

Wei, J., et al. (2022). Finetuned Language Models Are Zero-Shot Learners. *ICLR*.

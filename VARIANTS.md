# Kalle Variants

There are currently two variants of Kalle, with very different design philosophies and very different quality profiles.

## Kalle (production, `index.html`)

The deployed model at [pmmathias.github.io/krr-chat](https://pmmathias.github.io/krr-chat/). Trained on **2,174 hand-curated dialogue pairs** plus 29 RAG chunks from the eigenvalues blog post.

| Metric | Value |
|---|---|
| Pairs | 2,174 |
| Vocabulary | 2,977 words |
| RAG chunks | 29 (eigenvalues post only) |
| RFF dimension D | 6,144 |
| File size | 56 MB |
| Top-1 accuracy | 63.5% |
| Deployment | GitHub Pages (live) |

**Strengths:** dialogue quality, multi-turn coherence, scope-appropriate responses for eigenvalue questions.

## Kalle XL (experimental, local only)

An attempt to train Kalle on the **entire DE+EN blog corpus** (30 articles, ~197K extra tokens). Built by ticket T036.

| Metric | Value |
|---|---|
| Pairs | 7,302 (dialogue + programmatic Q&A from blog) |
| Vocabulary | 13,059 words |
| RAG chunks | 1,025 (all blog posts, H2/H3 sections) |
| RFF dimension D | 4,096 |
| File size | 144 MB |
| Top-1 accuracy | 32.1% |
| CG iterations | 7 (converged fast) |
| Deployment | local only (not live) |

### Honest evaluation: why Kalle XL is not a successor to Kalle

We ran a 20-query smoke test across 6+ blog topic areas (eigenvalues, quantum mechanics, music, psychology, logic, mindfulness, deepfakes, emergence, PageRank, Euler). Pass rate: **7/20 = 35%**. The failures were revealing:

- `"Was sind Emotionen?"` → received a paragraph about Steven Hayes and ACT therapy, not about emotions
- `"What is consciousness?"` → received a text about emergence taxonomy, adjacent but off-topic
- `"Who was Euler?"` → received the $e^{i\pi}$ section rather than biography
- `"ich mag pizza"` (a Kalle dialogue staple) → received a text about hash encoding, because the blog RAG-pairs overwhelmed Kalle's original pizza pair in the retrieval ranking

### Root cause: programmatic Q&A generation is too weak

The blog-to-pairs conversion in `src/gen_blog_rag_pairs.py` generates 5 pair variants per chunk using heuristics (heading-as-question, keyword-as-question, "explain X" template, first-sentence-as-answer). These pairs are **lexically plausible** but lack:

- **Question-answer alignment:** the answer is the first few sentences of the chunk, which is often introductory prose rather than a direct answer.
- **Distinctive keywords:** with 13K words of vocabulary, many heuristic questions produce ambiguous BoW embeddings that match multiple chunks.
- **Topic discrimination:** the blog covers 30 topics with shared vocabulary (matrix, vector, function...); simple retrieval can't distinguish "math-in-music" from "math-in-AI".

The Kalle dialogue pairs are overwhelmed in retrieval because the 5,125 blog pairs dominate the BoW+IDF embedding landscape. Dialogue queries like "ich mag pizza" no longer reliably match the original Kalle pair.

### What would actually work

Three options, in increasing effort:

1. **Retrieval-weight the Kalle pairs** — during retrieval, give Kalle's 2,174 curated pairs a score multiplier (e.g., 1.5×) so that dialogue queries still prefer them over blog pairs. A one-line change in the template.

2. **LLM-based Q&A generation** — use Claude or GPT-5 to generate actual Q&A from each chunk, with real question-answer alignment. Budget: ~1,000 chunks × 10 pairs × ~$0.001 = ~$10 in API costs. Estimated quality: 60-75% pass rate.

3. **Hybrid retrieval** — first attempt to match against Kalle dialogue; if confidence is high, return that; otherwise, route to blog RAG. This is the standard production RAG architecture.

### Conclusion

**Kalle XL confirms the v1 paper's "less is more" finding empirically.** We went from 2,174 → 7,302 pairs and Top-1 fell from 63.5% → 32.1%, and the downstream task quality (smoke test) fell accordingly. The training pipeline scales fine — 2M samples trained in 5 minutes — but scaling the *data* without scaling the *curation quality* degrades the system.

The production Kalle remains the recommended deployment. Kalle XL is preserved locally as a reference for what the brute-force scaling approach produces and is documented in [`projektmanagement/T036_kalle_xl_full_blog.md`](projektmanagement/T036_kalle_xl_full_blog.md).

## Reproducing Kalle XL (if you want to explore)

```bash
# 1. Generate the combined Kalle + blog corpus
python3.11 src/gen_blog_rag_pairs.py
# → writes data/corpus-xl.md (7302 pairs) and data/chunk_index_xl.json (1025 chunks)

# 2. Build (with D=4096 to stay under 200 MB)
python3.11 src/build_v2.py \
    --corpus data/corpus-xl.md \
    --chunks data/chunk_index_xl.json \
    --solver cg \
    --D 4096 \
    --output kalle-xl.html

# 3. Test (20 topic queries)
python3.11 tests/test_kalle_xl.py kalle-xl.html
```

Training time: ~5 minutes. Peak memory: ~2.6 GB. Resulting file: 144 MB (not committed to git; listed in `.gitignore`).

#!/usr/bin/env python3
"""
Phase 1: Extract templates from blog/corpus by masking high-IDF content words.

A template is a sentence with [MASK] slots where important content words
were removed. The surrounding structure (grammar, function words) stays intact.

Example:
  Original: "The eigenvalue tells you how much a vector is scaled."
  Template: "The [MASK] tells you how much a [MASK] is [MASK]."
  Expected fills: "eigenvalue", "vector", "scaled"

Masking strategy: words with high IDF (rare, content-bearing) are masked.
Function words (the, is, a, and, ...) are never masked.

Usage:
  python template_extractor.py --corpus corpus_15m.txt --tokenizer bpe_50m.json
"""
import os, sys, re, math, argparse, pickle, json
import numpy as np
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', default='data/autoregressive/corpus_large.txt')
    p.add_argument('--tokenizer', default='data/autoregressive/bpe_large.json')
    p.add_argument('--output', default='data/autoregressive/templates.pkl')
    p.add_argument('--min-sentence-len', type=int, default=8, help='Min tokens per sentence')
    p.add_argument('--max-sentence-len', type=int, default=40, help='Max tokens per sentence')
    p.add_argument('--mask-ratio', type=float, default=0.2, help='Fraction of tokens to mask')
    p.add_argument('--max-templates', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    V = tokenizer.get_vocab_size()

    with open(args.corpus) as f:
        text = f.read()

    # Tokenize full corpus
    all_ids = tokenizer.encode(text).ids
    print(f"Corpus: {len(all_ids):,} tokens, V={V}")

    # Compute IDF per token
    print("Computing IDF...")
    # Treat each 100-token window as a "document"
    doc_freq = Counter()
    n_docs = 0
    for start in range(0, len(all_ids) - 100, 50):
        doc = set(all_ids[start:start+100])
        for tid in doc:
            doc_freq[tid] += 1
        n_docs += 1
    idf = {}
    for tid in range(V):
        df = doc_freq.get(tid, 0) + 1
        idf[tid] = math.log(n_docs / df)
    print(f"  IDF computed for {V} tokens, n_docs={n_docs}")

    # Find sentence boundaries (tokens that are "." or "\n")
    period_tokens = set()
    for tid in range(V):
        decoded = tokenizer.decode([tid]).strip()
        if decoded in ['.', '!', '?', '\n', '\n\n']:
            period_tokens.add(tid)

    # Extract sentences
    print("Extracting sentences...")
    sentences = []
    current = []
    for tid in all_ids:
        current.append(tid)
        if tid in period_tokens:
            if args.min_sentence_len <= len(current) <= args.max_sentence_len:
                sentences.append(current.copy())
            current = []
    print(f"  {len(sentences):,} sentences (len {args.min_sentence_len}-{args.max_sentence_len})")

    # Identify function words (low IDF = very common = not content-bearing)
    idf_threshold = np.percentile(list(idf.values()), 40)  # bottom 40% = function words
    print(f"  IDF threshold for masking: {idf_threshold:.2f} (tokens above this are maskable)")

    # Create templates by masking high-IDF tokens
    print("Creating templates...")
    templates = []
    for sent in sentences:
        if len(templates) >= args.max_templates:
            break

        # Identify maskable positions (high IDF content words)
        maskable = []
        for pos, tid in enumerate(sent):
            if idf.get(tid, 0) > idf_threshold:
                maskable.append(pos)

        if len(maskable) < 2:
            continue  # need at least 2 maskable positions

        # Mask a random subset
        n_mask = max(2, min(len(maskable), int(len(sent) * args.mask_ratio)))
        mask_positions = sorted(np.random.choice(maskable, n_mask, replace=False))

        # Build template
        template_tokens = list(sent)
        original_tokens = {}
        for pos in mask_positions:
            original_tokens[pos] = sent[pos]
            template_tokens[pos] = -1  # sentinel for [MASK]

        # Decode for human readability
        original_text = tokenizer.decode(sent)
        template_parts = []
        for tid in template_tokens:
            if tid == -1:
                template_parts.append('[MASK]')
            else:
                template_parts.append(tokenizer.decode([tid]))
        template_text = ''.join(template_parts)

        templates.append({
            'token_ids': sent,
            'template_tokens': template_tokens,
            'mask_positions': mask_positions.tolist(),
            'original_at_masks': original_tokens,
            'original_text': original_text,
            'template_text': template_text,
            'n_masks': len(mask_positions),
        })

    print(f"  Created {len(templates):,} templates")
    print(f"\n--- Sample templates ---")
    for t in templates[:10]:
        print(f"  Original: {t['original_text'][:100]}")
        print(f"  Template: {t['template_text'][:100]}")
        fills = [tokenizer.decode([t['original_at_masks'][p]]) for p in sorted(t['original_at_masks'])]
        print(f"  Answers:  {fills}")
        print()

    # Save
    with open(args.output, 'wb') as f:
        pickle.dump({
            'templates': templates,
            'idf': idf,
            'idf_threshold': idf_threshold,
            'V': V,
        }, f)
    print(f"Saved {len(templates)} templates to {args.output}")

    # Stats
    n_masks_per = [t['n_masks'] for t in templates]
    print(f"\nMasks per template: mean={np.mean(n_masks_per):.1f}, "
          f"min={min(n_masks_per)}, max={max(n_masks_per)}")


if __name__ == '__main__':
    main()

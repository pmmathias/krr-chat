"""
T042 — Strip LaTeX, math, and noisy fragments from the blog corpus.

Goals:
  - Remove inline math \( ... \) and display math $$ ... $$
  - Remove LaTeX commands (\frac, \mathbf, \cdot, etc.)
  - Remove code-like fragments in backticks
  - Collapse residual whitespace / noise
  - Keep pure prose so BPE can learn meaningful subword patterns

Input:  data/autoregressive/corpus.txt
Output: data/autoregressive/corpus_clean.txt + bpe_tokenizer_clean.json
"""
import os, re, sys

DATA_DIR = '/Users/mathiasleonhardt/Dev/krr-chat/data/autoregressive'
INPUT = f'{DATA_DIR}/corpus.txt'
OUTPUT = f'{DATA_DIR}/corpus_clean.txt'
TOKENIZER_OUT = f'{DATA_DIR}/bpe_tokenizer_clean.json'

VOCAB_SIZE = 8192


def clean_text(txt):
    """Aggressive cleaning for prose-only language modeling."""
    before_chars = len(txt)

    # --- Math blocks ---
    # Display math: $$ ... $$  or  \[ ... \]
    txt = re.sub(r'\$\$[\s\S]*?\$\$', ' ', txt)
    txt = re.sub(r'\\\[[\s\S]*?\\\]', ' ', txt)

    # Inline math: \( ... \)  or  $ ... $
    txt = re.sub(r'\\\([\s\S]*?\\\)', ' ', txt)
    txt = re.sub(r'\$[^$\n]{1,200}\$', ' ', txt)

    # --- LaTeX commands that slipped through ---
    # \command or \command{arg}
    txt = re.sub(r'\\[a-zA-Z]+\{[^{}]{0,200}\}', ' ', txt)
    txt = re.sub(r'\\[a-zA-Z]+', ' ', txt)

    # Stray curly braces, hats, underscores common in formula leftovers
    # but only when adjacent to math-like characters
    txt = re.sub(r'[{}]', ' ', txt)
    txt = re.sub(r'\^[+-]?[a-zA-Z0-9]{0,3}', ' ', txt)  # super/subscripts
    txt = re.sub(r'_\{[^}]{0,30}\}', ' ', txt)

    # --- Code-like fragments ---
    # Inline code: `foo`
    txt = re.sub(r'`[^`\n]{1,100}`', ' ', txt)
    # Triple-backtick blocks
    txt = re.sub(r'```[\s\S]*?```', ' ', txt)

    # --- Residual noise ---
    # URL-like patterns (http, www.)
    txt = re.sub(r'https?://\S+', ' ', txt)
    txt = re.sub(r'www\.\S+', ' ', txt)

    # File paths / Python-ish references
    txt = re.sub(r'[a-zA-Z0-9_/]+\.(py|html|json|md|csv|pkl)\b', ' ', txt)

    # Numbers with exponents or decimals that are clearly not prose
    # Keep plain integers and simple decimals — they can be legitimate prose
    txt = re.sub(r'\b\d+\.\d+e[+-]?\d+\b', ' ', txt, flags=re.I)
    txt = re.sub(r'\b\d{5,}\b', ' ', txt)  # very long numbers

    # Strange Unicode math symbols
    txt = re.sub(r'[⊤⊥±∓∞∑∏∂∇∫∮∝≈≠≤≥≺≻⟨⟩→←↔⇒⇐⇔]', ' ', txt)

    # Collapse whitespace
    txt = re.sub(r'[ \t]+', ' ', txt)
    txt = re.sub(r'\n{3,}', '\n\n', txt)
    txt = re.sub(r'[^\S\n]+\n', '\n', txt)
    txt = re.sub(r'\n[^\S\n]+', '\n', txt)
    txt = txt.strip()

    after_chars = len(txt)
    removed_pct = 100 * (before_chars - after_chars) / max(before_chars, 1)
    return txt, removed_pct


def main():
    with open(INPUT) as f:
        raw = f.read()
    print(f"Input:  {len(raw):,} chars")

    clean, pct = clean_text(raw)
    print(f"Output: {len(clean):,} chars  ({pct:.1f}% removed)")

    with open(OUTPUT, 'w') as f:
        f.write(clean)
    print(f"Wrote {OUTPUT}")

    # Spot-check samples
    print("\n--- Sample before (first 500 chars of raw) ---")
    print(raw[:500])
    print("\n--- Sample after (first 500 chars of clean) ---")
    print(clean[:500])

    print("\n--- LaTeX-residue check (should be ~0 hits) ---")
    for pat in [r'\\frac', r'\\math', r'\\cdot', r'\$\$', r'\\\(']:
        hits = len(re.findall(pat, clean))
        print(f"  '{pat}': {hits} hits in clean")

    # Re-train BPE on clean corpus
    print(f"\n--- Training new BPE tokenizer (V={VOCAB_SIZE}) ---")
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=['<pad>', '<unk>', '<s>', '</s>'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train([OUTPUT], trainer)
    tok.save(TOKENIZER_OUT)
    print(f"  Saved {TOKENIZER_OUT}")

    # Tokenize clean corpus and print stats
    ids = tok.encode(clean).ids
    vocab = tok.get_vocab_size()
    print(f"  Clean corpus tokenized: {len(ids):,} tokens")
    print(f"  Vocab size: {vocab}")

    # Peek at most common tokens (cheap via Counter)
    from collections import Counter
    cnt = Counter(ids)
    top10 = cnt.most_common(10)
    print(f"\n  Top 10 most common tokens:")
    for tid, freq in top10:
        print(f"    id={tid:>5} freq={freq:>5}  {tok.decode([tid])!r}")


if __name__ == '__main__':
    main()

"""
T037 — Prepare raw text corpus for autoregressive KRR training.

Extracts plain prose from the ki-mathias.de blog (DE+EN) and saves it as a
single flat text file, then trains a BPE tokenizer on it.
"""
import os, re, html, glob, sys

BLOG = '/Users/mathiasleonhardt/Dev/QuantenBlog'
OUT_TEXT = '/Users/mathiasleonhardt/Dev/krr-chat/data/autoregressive/corpus.txt'
OUT_TOKENIZER = '/Users/mathiasleonhardt/Dev/krr-chat/data/autoregressive/bpe_tokenizer.json'

SKIP = {'404', 'datenschutz', 'impressum', 'imprint', 'privacy',
        'index', 'sitemap', 'about'}


def strip_html(txt):
    """Extract plain prose text from HTML."""
    # Remove scripts, styles, SVGs entirely
    for tag in ['script', 'style', 'svg', 'nav', 'header', 'footer']:
        txt = re.sub(rf'<{tag}\b[^>]*>.*?</{tag}>', ' ', txt, flags=re.DOTALL|re.I)
    # Remove all remaining tags
    txt = re.sub(r'<[^>]+>', ' ', txt)
    # Decode HTML entities (&amp; -> & etc.)
    txt = html.unescape(txt)
    # Collapse whitespace
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt


def main():
    os.makedirs(os.path.dirname(OUT_TEXT), exist_ok=True)

    # Collect text from all blog articles (DE + EN)
    files = sorted(set(glob.glob(f'{BLOG}/*.html')) | set(glob.glob(f'{BLOG}/en/*.html')))
    all_text = []
    counts = []
    for f in files:
        base = os.path.basename(f).replace('.html', '')
        if base in SKIP:
            continue
        with open(f) as fp:
            raw = fp.read()
        clean = strip_html(raw)
        if len(clean.split()) < 200:
            continue
        all_text.append(clean)
        counts.append((f, len(clean)))
    print(f"Articles: {len(counts)}")
    total_chars = sum(len(t) for t in all_text)
    print(f"Total characters: {total_chars:,}")

    # One big text stream with double newlines between articles
    corpus_text = '\n\n'.join(all_text)
    with open(OUT_TEXT, 'w') as f:
        f.write(corpus_text)
    print(f"Wrote corpus: {OUT_TEXT} ({len(corpus_text):,} chars)")

    # Train BPE tokenizer
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=8192,
        min_frequency=2,
        special_tokens=['<pad>', '<unk>', '<s>', '</s>'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"\nTraining BPE tokenizer (vocab_size=8192)...")
    tokenizer.train([OUT_TEXT], trainer)
    tokenizer.save(OUT_TOKENIZER)
    print(f"Wrote tokenizer: {OUT_TOKENIZER}")

    # Quick sanity check
    print(f"\n--- Sanity check ---")
    samples = [
        "Heute scheint die Sonne über Hamburg.",
        "Kernel ridge regression solves a linear system.",
        "Im Anfang war das Wort.",
        "The eigenvalue of a matrix tells you how much it scales along that direction.",
    ]
    for s in samples:
        ids = tokenizer.encode(s).ids
        decoded = tokenizer.decode(ids)
        print(f"  {s!r}")
        print(f"    tokens ({len(ids)}): {ids[:15]}{'...' if len(ids) > 15 else ''}")
        print(f"    decoded: {decoded!r}")

    # Token count of full corpus
    full_ids = tokenizer.encode(corpus_text).ids
    print(f"\nFull corpus tokenized: {len(full_ids):,} tokens")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == '__main__':
    main()

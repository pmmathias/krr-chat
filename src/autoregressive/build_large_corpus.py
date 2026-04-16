"""
T043 — Build a larger (~5M token) DE+EN corpus for autoregressive KRR.

Sources:
  - WikiText-103 (EN): curated English Wikipedia prose, public license
  - German Wikipedia subset (via HuggingFace datasets, streaming)
  - Our existing blog corpus (bonus)

Target: ~2-3M tokens of each language, mixed randomly at the paragraph level.
Output:
  - data/autoregressive/corpus_large.txt    (raw text, cleaned)
  - data/autoregressive/bpe_large.json      (V=16384 BPE)
"""
import os, re, random
from datasets import load_dataset

DATA_DIR   = '/Users/mathiasleonhardt/Dev/krr-chat/data/autoregressive'
OUT_TEXT   = f'{DATA_DIR}/corpus_large.txt'
OUT_BPE    = f'{DATA_DIR}/bpe_large.json'
BLOG_TXT   = f'{DATA_DIR}/corpus_clean.txt'
VOCAB_SIZE = 16384

# Rough token budget (we'll cap once we hit these)
TARGET_EN_CHARS = 6_000_000   # ~1.2M tokens in BPE-dense prose
TARGET_DE_CHARS = 6_000_000   # ~1.2M tokens

random.seed(42)


def clean_para(txt):
    """Light cleanup for Wikipedia paragraphs."""
    # Remove headers like '= Title =', citations [1], curly tags {{foo}}
    txt = re.sub(r'=+\s*[^=]+\s*=+', ' ', txt)
    txt = re.sub(r'\[\d+\]', ' ', txt)
    txt = re.sub(r'\{\{[^}]{0,200}\}\}', ' ', txt)
    # Remove URLs
    txt = re.sub(r'https?://\S+', ' ', txt)
    # Remove @,<,> tags
    txt = re.sub(r'<[^>]{0,200}>', ' ', txt)
    # Collapse whitespace
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt


def collect_en_wikipedia(target_chars):
    """Get English Wikipedia prose via WikiText-103."""
    print("Loading WikiText-103 (EN)...")
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train', streaming=True)
    total = 0
    chunks = []
    for row in ds:
        txt = clean_para(row['text'])
        if len(txt) < 100:
            continue
        chunks.append(txt)
        total += len(txt)
        if total >= target_chars:
            break
    print(f"  Collected {len(chunks):,} EN paragraphs ({total:,} chars)")
    return chunks


def collect_de_wikipedia(target_chars):
    """Get German Wikipedia prose. HF has 'wikipedia' + '20220301.de' config."""
    print("Loading German Wikipedia (streaming)...")
    try:
        ds = load_dataset('wikimedia/wikipedia', '20231101.de',
                          split='train', streaming=True)
    except Exception as e:
        print(f"  wikimedia/wikipedia failed ({e}), falling back to graelo/wikipedia")
        ds = load_dataset('graelo/wikipedia', '20230901.de',
                          split='train', streaming=True)

    total = 0
    chunks = []
    for row in ds:
        txt = clean_para(row.get('text', ''))
        if len(txt) < 200:
            continue
        # Take first ~1000 chars of each article to get diverse topics
        chunks.append(txt[:1500])
        total += min(len(txt), 1500)
        if total >= target_chars:
            break
    print(f"  Collected {len(chunks):,} DE articles ({total:,} chars)")
    return chunks


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Our own blog as flavor ---
    blog_chunks = []
    if os.path.exists(BLOG_TXT):
        with open(BLOG_TXT) as f:
            blog_txt = f.read()
        # Split by double-newline for paragraphs
        blog_chunks = [p.strip() for p in blog_txt.split('\n\n') if len(p.strip()) > 100]
        print(f"Blog: {len(blog_chunks):,} paragraphs, {len(blog_txt):,} chars")

    # --- Wikipedia samples ---
    try:
        en_chunks = collect_en_wikipedia(TARGET_EN_CHARS)
    except Exception as e:
        print(f"EN Wikipedia failed: {e}")
        en_chunks = []

    try:
        de_chunks = collect_de_wikipedia(TARGET_DE_CHARS)
    except Exception as e:
        print(f"DE Wikipedia failed: {e}")
        de_chunks = []

    # --- Mix at paragraph level ---
    all_chunks = blog_chunks + en_chunks + de_chunks
    random.shuffle(all_chunks)
    combined = '\n\n'.join(all_chunks)
    print(f"\nCombined corpus: {len(all_chunks):,} paragraphs, {len(combined):,} chars")

    with open(OUT_TEXT, 'w') as f:
        f.write(combined)
    print(f"Wrote {OUT_TEXT}")

    # --- Train BPE ---
    print(f"\nTraining BPE (V={VOCAB_SIZE})...")
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=3,
        special_tokens=['<pad>', '<unk>', '<s>', '</s>'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train([OUT_TEXT], trainer)
    tok.save(OUT_BPE)
    print(f"Wrote {OUT_BPE}")

    # Tokenize + stats
    ids = tok.encode(combined).ids
    print(f"\n--- Corpus stats ---")
    print(f"Chars:        {len(combined):,}")
    print(f"BPE tokens:   {len(ids):,}")
    print(f"Vocab size:   {tok.get_vocab_size()}")
    print(f"Compression:  {len(combined)/len(ids):.2f} chars/token")

    # Top tokens sanity check
    from collections import Counter
    cnt = Counter(ids)
    print(f"\n--- Top 10 most common tokens ---")
    for tid, freq in cnt.most_common(10):
        print(f"  id={tid:>5} freq={freq:>7}  {tok.decode([tid])!r}")


if __name__ == '__main__':
    main()

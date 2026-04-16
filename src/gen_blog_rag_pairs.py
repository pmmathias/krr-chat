"""
Generate Kalle XL corpus: Kalle dialogue pairs + Blog Q&A pairs.

Parses all DE/EN blog posts, splits each into H2/H3 chunks, and generates
multiple Q&A pairs per chunk. Output formats:

  - data/corpus-xl.md        : Kalle pairs + Blog Q&A pairs in Kalle dialog format
  - data/chunk_index_xl.json : chunk index extended to all blog posts for RAG

Pair generation is programmatic (not LLM-based) — we use heading text as
the natural question, and the first sentence(s) of the chunk as the answer.
Additional variants are created by rewording.
"""
import os, re, html, glob, json, sys

BLOG = '/Users/mathiasleonhardt/Dev/QuantenBlog'
KALLE_CORPUS = '/Users/mathiasleonhardt/Dev/krr-chat/data/corpus.md'
OUT_CORPUS = '/Users/mathiasleonhardt/Dev/krr-chat/data/corpus-xl.md'
OUT_CHUNKS = '/Users/mathiasleonhardt/Dev/krr-chat/data/chunk_index_xl.json'

# Skip meta pages that are not content
SKIP = {'404', 'datenschutz', 'impressum', 'imprint', 'privacy',
        'index', 'sitemap', 'about'}

# Keyword boost list for the RAG chunk-match (same as existing chunk_index.json)
EN_KEYWORDS = {
    'eigenvalue','eigenvalues','eigenvector','eigenvectors','kernel','pagerank',
    'projection','regularization','convergence','overfitting','residual','iteration',
    'fourier','radiosity','quantum','neural','matrix','vector','regression','ridge',
    'stochastic','polynomial','diagonal','subspace','linear','nonlinear','feature',
    'emergence','entropy','probability','bayes','logic','incompleteness',
    'mindfulness','emotion','perception','euler','music','harmony','consciousness',
    'intelligence','artificial','attention','transformer','embedding',
}


# ----------------------------------------------------------------------
# HTML → plain text per section
# ----------------------------------------------------------------------
def strip_inline_tags(txt):
    """Remove inline HTML tags, keep text."""
    txt = re.sub(r'<script\b[^>]*>.*?</script>', ' ', txt, flags=re.DOTALL|re.I)
    txt = re.sub(r'<style\b[^>]*>.*?</style>', ' ', txt, flags=re.DOTALL|re.I)
    txt = re.sub(r'<svg\b[^>]*>.*?</svg>', ' ', txt, flags=re.DOTALL|re.I)
    txt = re.sub(r'<[^>]+>', ' ', txt)
    txt = html.unescape(txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt


def parse_blog_sections(html_path):
    """Split a blog HTML into (heading_text, section_text) pairs by H2/H3."""
    with open(html_path) as f:
        raw = f.read()

    # Remove nav, header, footer globally
    for tag in ['nav', 'header', 'footer']:
        raw = re.sub(rf'<{tag}\b[^>]*>.*?</{tag}>', ' ', raw, flags=re.DOTALL|re.I)

    # Find main content area (everything after first </h1> typically)
    h1_end = raw.find('</h1>')
    if h1_end >= 0:
        raw = raw[h1_end+5:]

    # Split by H2 or H3
    pattern = re.compile(r'<(h[23])[^>]*>(.*?)</\1>', re.DOTALL|re.I)
    sections = []
    last_end = 0
    last_heading = None
    last_start = 0

    for m in pattern.finditer(raw):
        # Text between previous heading and this one = previous section
        if last_heading is not None:
            section_html = raw[last_start:m.start()]
            section_text = strip_inline_tags(section_html)
            if len(section_text.split()) > 30:  # skip near-empty sections
                sections.append((last_heading, section_text))
        last_heading = strip_inline_tags(m.group(2))
        last_start = m.end()

    # Last section
    if last_heading is not None:
        section_html = raw[last_start:]
        section_text = strip_inline_tags(section_html)
        if len(section_text.split()) > 30:
            sections.append((last_heading, section_text))

    return sections


# ----------------------------------------------------------------------
# Kalle tokenization
# ----------------------------------------------------------------------
def kalle_tokenize(txt, max_words=None):
    """Apply Kalle's tokenization conventions (lowercase, umlauts→ae/oe/ue)."""
    umlauts = {'ä':'ae','ö':'oe','ü':'ue','Ä':'ae','Ö':'oe','Ü':'ue','ß':'ss'}
    for u,r in umlauts.items():
        txt = txt.replace(u,r)
    txt = txt.lower()
    # keep digits, letters, basic punctuation
    txt = re.sub(r"[^a-z0-9.,!?;:'\-\s]", ' ', txt)
    # tokenize punctuation
    txt = re.sub(r'([.,!?;:])', r' \1 ', txt)
    tokens = [t for t in txt.split() if t]
    if max_words:
        tokens = tokens[:max_words]
    return tokens


def truncate_to_chunk(text, max_words=80):
    """Take first N meaningful words for a chunk summary."""
    toks = kalle_tokenize(text, max_words=max_words)
    return ' '.join(toks)


def first_sentences(text, n=3, max_words=60):
    """First n sentences from the text (for generating answers)."""
    toks = kalle_tokenize(text)
    # collect up to n '.'-terminated sentences
    sentences = []
    cur = []
    for t in toks:
        cur.append(t)
        if t == '.':
            sentences.append(' '.join(cur))
            cur = []
            if len(sentences) >= n:
                break
    if cur and not sentences:
        sentences.append(' '.join(cur))
    result = ' '.join(sentences)
    # trim to max_words
    rtoks = result.split()
    if len(rtoks) > max_words:
        result = ' '.join(rtoks[:max_words]) + ' .'
    if not result.endswith('.'):
        result = result + ' .'
    return result


# ----------------------------------------------------------------------
# Question generation from heading
# ----------------------------------------------------------------------
def heading_to_question(heading, lang):
    """Turn a heading into a natural question."""
    h = heading.strip()
    # If already a question, keep
    if h.endswith('?'):
        return kalle_tokenize(h)

    # Common heuristics
    toks = kalle_tokenize(h)
    if not toks:
        return []

    if lang == 'de':
        # "Was ist X" style
        if not toks[0] in ('was', 'wie', 'warum', 'wo', 'wer', 'wann', 'welches'):
            # Natural question: "was ist <heading>?"
            return ['was', 'bedeutet'] + toks + ['?']
        return toks + ['?']
    else:
        if not toks[0] in ('what', 'how', 'why', 'where', 'who', 'when', 'which'):
            return ['what', 'is'] + toks + ['?']
        return toks + ['?']


def detect_lang(path):
    return 'en' if '/en/' in path else 'de'


def keywords_from_text(text, max_kw=15):
    """Extract likely topic keywords (frequent content words, EN-keyword-boosted)."""
    toks = kalle_tokenize(text)
    # filter stopwords (very common tokens)
    stop_de = {'der','die','das','ein','eine','einen','und','oder','aber','ist','sind',
               'war','waren','zu','in','im','auf','mit','bei','von','vom','zum','zur',
               'fuer','dass','nicht','auch','so','als','wie','wenn','wir','sie','er',
               'es','du','ich','mich','dich','sich','man','aus','an','am','sein','seine',
               'ihr','ihre','diese','dieser','dieses','noch','nur','mehr','ueber','um'}
    stop_en = {'the','a','an','and','or','but','is','are','was','were','to','of','in',
               'on','at','with','for','from','that','this','these','those','not','as',
               'by','be','been','being','have','has','had','do','does','did','we','you',
               'they','he','she','it','i','me','him','her','us','them','so','if','then',
               'than','which','who','what','how','why'}
    stop = stop_de | stop_en | {'.','?','!',',',':',';',"'",'-'}
    content = [t for t in toks if t not in stop and len(t) >= 3]

    # Rank by frequency, boost domain keywords
    from collections import Counter
    cnt = Counter(content)
    def score(word, freq):
        return freq + (3.0 if word in EN_KEYWORDS else 0.0)
    ranked = sorted(cnt.items(), key=lambda kv: -score(kv[0], kv[1]))
    return [w for w, _ in ranked[:max_kw]]


# ----------------------------------------------------------------------
# Pair generation from a (heading, text, lang) chunk
# ----------------------------------------------------------------------
def pairs_from_chunk(heading, text, lang, chunk_snippet):
    """Produce a list of (user_line, bot_line) pairs for a blog chunk."""
    pairs = []

    # Truncate answer to 3 sentences max
    answer = first_sentences(text, n=3, max_words=50)

    # 1) Direct question from heading (no context — quick-fire fact)
    q_toks = heading_to_question(heading, lang)
    if q_toks and len(q_toks) >= 2:
        q_str = ' '.join(q_toks).rstrip(' ?').strip()
        if q_str:
            pairs.append((q_str, answer))

    # 2) RAG-style context-anchored pair ("kontext <snippet> frage <q>")
    if q_toks and len(q_toks) >= 2:
        q_str = ' '.join(q_toks).rstrip(' ?').strip()
        if q_str:
            kontext_prefix = 'kontext' if lang == 'de' else 'context'
            frage_prefix = 'frage' if lang == 'de' else 'question'
            user = f'{kontext_prefix} {chunk_snippet} {frage_prefix} {q_str}'
            pairs.append((user, answer))

    # 3) "Tell me about <heading>" variant
    if lang == 'de':
        tell_prefixes = ['erklaere', 'erzaehl mir ueber', 'was sagt der artikel ueber']
    else:
        tell_prefixes = ['explain', 'tell me about', 'describe']
    heading_clean = ' '.join(kalle_tokenize(heading))
    for prefix in tell_prefixes[:1]:  # use just one to avoid too much duplication
        if heading_clean:
            user = f'{prefix} {heading_clean}'
            pairs.append((user, answer))

    # 4) Keyword-based question (for top keywords)
    kws = keywords_from_text(text, max_kw=3)
    for kw in kws[:2]:  # limit to avoid corpus bloat
        if lang == 'de':
            user = f'was ist {kw}'
        else:
            user = f'what is {kw}'
        pairs.append((user, answer))

    return pairs


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    files = sorted(set(glob.glob(f'{BLOG}/*.html')) | set(glob.glob(f'{BLOG}/en/*.html')))

    all_chunks = []      # for chunk_index.json
    all_pairs = []       # for corpus.md

    for f in files:
        base = os.path.basename(f).replace('.html', '')
        if base in SKIP:
            continue
        lang = detect_lang(f)
        sections = parse_blog_sections(f)
        if not sections:
            continue
        print(f"  {os.path.relpath(f, BLOG)} [{lang}]: {len(sections)} sections")

        for heading, section in sections:
            if not heading.strip() or len(section.split()) < 30:
                continue

            chunk_snippet = truncate_to_chunk(section, max_words=80)
            chunk_keywords = keywords_from_text(section, max_kw=10)

            # Chunk index entry
            all_chunks.append({
                'heading': heading.strip(),
                'text': chunk_snippet,
                'keywords': chunk_keywords,
                'source': os.path.relpath(f, BLOG),
                'lang': lang,
            })

            # Pairs from this chunk
            pairs = pairs_from_chunk(heading, section, lang, chunk_snippet)
            all_pairs.extend(pairs)

    # Load existing Kalle pairs
    print()
    kalle_pairs_raw = []
    with open(KALLE_CORPUS) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('du:'): continue
            if ' bot: ' not in line: continue
            u, b = line.split(' bot: ', 1)
            u = u.replace('du:', '').strip().rstrip(' .').strip()
            b = b.strip().rstrip(' .').strip()
            if u and b:
                kalle_pairs_raw.append((u, b))
    print(f"Kalle dialogue pairs:    {len(kalle_pairs_raw):>6}")
    print(f"Blog Q&A pairs:          {len(all_pairs):>6}")
    print(f"Blog chunks:             {len(all_chunks):>6}")

    # Combine (Kalle first, then blog pairs)
    combined = kalle_pairs_raw + all_pairs

    # Dedupe by (user, bot)
    seen = set()
    dedup = []
    for u, b in combined:
        key = (u, b)
        if key in seen: continue
        seen.add(key)
        dedup.append((u, b))
    print(f"Total after dedupe:      {len(dedup):>6}")

    # Write corpus.md
    with open(OUT_CORPUS, 'w') as f:
        f.write('# Kalle XL corpus (dialogue + full blog Q&A)\n\n')
        for u, b in dedup:
            # Make sure pairs end with period
            u_clean = u.rstrip(' .').strip()
            b_clean = b.rstrip(' .').strip()
            if not b_clean.endswith(('.', '?', '!')):
                b_clean = b_clean + ' .'
            f.write(f'du: {u_clean} . bot: {b_clean}\n\n')
    print(f"\nWrote corpus: {OUT_CORPUS}")

    # Write chunk index
    with open(OUT_CHUNKS, 'w') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"Wrote chunks: {OUT_CHUNKS}")

    # Vocab stats
    all_tokens = []
    for u, b in dedup:
        all_tokens.extend(kalle_tokenize(u) + kalle_tokenize(b))
    vocab_size = len(set(all_tokens))
    print(f"\nEstimated corpus stats:")
    print(f"  Pairs:        {len(dedup):,}")
    print(f"  Tokens:       {len(all_tokens):,}")
    print(f"  Vocab:        {vocab_size:,}")
    print(f"  Chunks (RAG): {len(all_chunks):,}")


if __name__ == '__main__':
    main()

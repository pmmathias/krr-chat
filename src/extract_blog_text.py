"""
Extract plain text from all blog HTML files (DE + EN content only — skips
navigation, footers, meta pages like imprint/privacy).
"""
import os, re, html, glob

BLOG = '/Users/mathiasleonhardt/Dev/QuantenBlog'
SKIP = {'404', 'datenschutz', 'impressum', 'about', 'imprint', 'privacy',
        'index', 'sitemap'}

def strip_html(txt):
    # Remove <script>, <style>, <nav>, <header>, <footer> blocks with content
    txt = re.sub(r'<script\b[^>]*>.*?</script>', ' ', txt, flags=re.DOTALL|re.I)
    txt = re.sub(r'<style\b[^>]*>.*?</style>', ' ', txt, flags=re.DOTALL|re.I)
    txt = re.sub(r'<nav\b[^>]*>.*?</nav>', ' ', txt, flags=re.DOTALL|re.I)
    txt = re.sub(r'<header\b[^>]*>.*?</header>', ' ', txt, flags=re.DOTALL|re.I)
    txt = re.sub(r'<footer\b[^>]*>.*?</footer>', ' ', txt, flags=re.DOTALL|re.I)
    # Remove SVGs (figures with inline graphics)
    txt = re.sub(r'<svg\b[^>]*>.*?</svg>', ' ', txt, flags=re.DOTALL|re.I)
    # Remove all other tags
    txt = re.sub(r'<[^>]+>', ' ', txt)
    # Decode HTML entities
    txt = html.unescape(txt)
    # Normalize whitespace
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt


def tokenize(txt):
    # Kalle-style lowercasing, only ascii-ish alphanum + digits + basic punct
    # Replace umlauts the way the existing corpus does
    umlaut_map = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'Ä': 'ae', 'Ö': 'oe',
                  'Ü': 'ue', 'ß': 'ss'}
    for u, r in umlaut_map.items():
        txt = txt.replace(u, r)
    txt = txt.lower()
    # Split off punctuation as separate tokens (like the corpus)
    txt = re.sub(r'([.,!?;:])', r' \1 ', txt)
    # Collapse everything except letters/digits/basic punct into space
    txt = re.sub(r"[^a-z0-9äöüß.,!?;:'\-\s]", ' ', txt)
    tokens = [t for t in txt.split() if t]
    # Drop periods in the middle of numbers, keep standalone
    return tokens


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
    toks = tokenize(clean)
    if len(toks) < 200:
        continue  # skip near-empty pages
    counts.append((f, len(toks)))
    all_text.extend(toks)
    # Sentence separator between articles
    all_text.append('.')

print(f"Articles: {len(counts)}")
print(f"Total tokens: {len(all_text):,}")
print(f"Unique tokens: {len(set(all_text)):,}")
print(f"\nTop 10 longest articles:")
for f, n in sorted(counts, key=lambda x: -x[1])[:10]:
    print(f"  {os.path.relpath(f, BLOG):<45} {n:>8,} tokens")
print(f"\nBottom 5 shortest (still >200):")
for f, n in sorted(counts, key=lambda x: x[1])[:5]:
    print(f"  {os.path.relpath(f, BLOG):<45} {n:>8,} tokens")

# Save tokenized text
with open('/tmp/blog_tokens.txt', 'w') as f:
    f.write(' '.join(all_text))
print(f"\nSaved to /tmp/blog_tokens.txt")

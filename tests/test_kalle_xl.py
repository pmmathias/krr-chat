"""
Quick smoke test for Kalle XL across blog topic breadth.

Tests 20 queries spanning 6+ topic areas (eigenvalues, quantum, music,
psychology, logic, mindfulness, KRR, emergence, Euler, etc.).

Usage:
    python3.11 tests/test_kalle_xl.py [html_file]
"""
import asyncio
import sys
from playwright.async_api import async_playwright

QUERIES = [
    # Eigenvalues / math
    ("What are eigenvalues?", "eigenvalue|eigenvector"),
    ("Was sind Eigenwerte?", "eigenwert|eigenvektor"),
    ("Explain the kernel trick", "kernel|feature|high"),
    # Quantum
    ("What is the Schrödinger equation?", "schroedinger|quantum|wave"),
    ("Was ist Quantenmechanik?", "quanten|welle|energie"),
    # Psychology
    ("Was sind Emotionen?", "gefuehl|emotion"),
    ("What is consciousness?", "conscious|awareness|mind"),
    # Music
    ("Was ist ein Akkord?", "akkord|note|harmonie"),
    ("What makes music harmonious?", "harmon|chord|note"),
    # Logic
    ("Was sagt Gödel?", "goedel|unvollstaendig|wahrheit"),
    ("Explain incompleteness", "goedel|incomplete|formal"),
    # Mindfulness
    ("Was ist Achtsamkeit?", "achtsam|meditation|gegenwart"),
    ("What is mindfulness?", "mindful|awareness|meditation"),
    # KRR / blog topic
    ("How does KRR work?", "kernel|ridge|regression"),
    ("Wie funktioniert PageRank?", "pagerank|eigenvektor|link"),
    # Emergence
    ("What is emergence?", "emergen|complex|self"),
    # Euler
    ("Who was Euler?", "euler|mathemati"),
    # Deepfakes
    ("What are deepfakes?", "deepfake|face|video"),
    # Casual (should still work — Kalle dialogue pairs preserved)
    ("hallo", "hallo|hey|schoen|kalle"),
    ("ich mag pizza", "pizza"),
]


async def run_query(page, query, timeout=30000):
    """Send a query, wait for response, return bot answer text."""
    await page.fill('#in', query)
    await page.click('#go')

    # Wait for bot bubble to appear with text
    try:
        await page.wait_for_function(
            """() => {
                const bubbles = document.querySelectorAll('.m.b');
                if (!bubbles.length) return false;
                const last = bubbles[bubbles.length - 1];
                return last.textContent.trim().length > 10;
            }""",
            timeout=timeout
        )
        # Small delay for word animation to settle
        await page.wait_for_timeout(2000)
        bubbles = await page.query_selector_all('.m.b')
        if bubbles:
            text = await bubbles[-1].inner_text()
            return text.strip()
    except Exception as e:
        return f"[TIMEOUT: {e}]"
    return "[NO RESPONSE]"


async def main():
    html_file = sys.argv[1] if len(sys.argv) > 1 else 'kalle-xl.html'
    from pathlib import Path
    url = f"file://{Path(html_file).resolve()}"
    print(f"Testing: {url}")
    print()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1200, 'height': 800})
        page = await context.new_page()
        print(f"Loading HTML (may take ~30s for large file)...")
        await page.goto(url)

        # Wait for "Ready" status
        await page.wait_for_function(
            """() => {
                const st = document.getElementById('st');
                return st && (st.textContent.includes('Ready') || st.textContent.includes('Bereit'));
            }""",
            timeout=180000,  # 3 min for 211 MB load
        )
        status = await page.inner_text('#st')
        print(f"Status: {status}")
        print()

        passed = 0
        for i, (q, expected_re) in enumerate(QUERIES, 1):
            import re
            ans = await run_query(page, q)
            # Strip legend/"Under the hood" etc - take first paragraph
            ans_first = ans.split('\n')[0][:200]
            match = bool(re.search(expected_re, ans.lower()))
            status = '✓' if match else '✗'
            if match: passed += 1
            print(f"{status} [{i:>2}/{len(QUERIES)}] {q!r}")
            print(f"     → {ans_first!r}")

        print()
        print(f"{'='*60}")
        print(f"PASSED: {passed}/{len(QUERIES)} ({passed*100//len(QUERIES)}%)")
        print(f"{'='*60}")

        await browser.close()
        sys.exit(0 if passed >= len(QUERIES) * 0.6 else 1)


if __name__ == '__main__':
    asyncio.run(main())

#!/usr/bin/env python3
"""
T003: Kalle KRR-Chatbot Regression Test Suite

Usage:
    python3 test_regression.py [html_file] [--filter category] [--threshold 0.8]

Examples:
    python3 test_regression.py /tmp/kalle-ernsthaft.html
    python3 test_regression.py /tmp/kalle-ernsthaft.html --filter math
    python3 test_regression.py /tmp/kalle-ernsthaft.html --threshold 0.9

Exit codes:
    0 = pass rate >= threshold (default 80%)
    1 = pass rate < threshold
"""
import asyncio, sys, re, time, argparse, json
from dataclasses import dataclass, asdict
from typing import Optional
from playwright.async_api import async_playwright

# ============================================================
# Test Result Data Structure
# ============================================================
@dataclass
class TestResult:
    scenario_id: str
    category: str
    query: str
    response_text: str
    matched_pair: str
    kw_raw: float
    top3_pct: float
    word_count: int
    is_fallback: bool
    response_time_ms: int
    passed: bool
    fail_reason: Optional[str] = None

# ============================================================
# Scenario Definitions
# ============================================================
# Each scenario: (id, category, query, assertions)
# Assertions: dict with keys:
#   contains: list of words that MUST be in response (any of them)
#   not_contains: list of words that must NOT be in response
#   pair_prefix: expected pair user-side starts with this
#   expect_fallback: True if we EXPECT fallback
#   min_words: minimum word count
#   min_kw_raw: minimum confidence score
#   check_last_bot_turn: expected substring in lastBotTurn (for multi-turn)

SINGLE_TURN = [
    ('greeting_hey', 'greeting', 'hey wie gehts', {
        'contains': ['kalle'],
        'pair_prefix': 'hey',
        'min_words': 15,
    }),
    ('greeting_hallo', 'greeting', 'hallo', {
        'contains': ['schoen', 'kalle'],
        'min_words': 10,
    }),
    ('greeting_tschuess', 'greeting', 'tschuess', {
        'contains': ['bald', 'schoen', 'tschuess', 'bye', 'mach'],
        'min_words': 5,
    }),
    ('food_pizza', 'food', 'ich mag pizza', {
        'contains': ['pizza'],
        'min_words': 10,
        'min_kw_raw': 5.0,
    }),
    ('food_lieblingsessen', 'food', 'was ist dein lieblingsessen', {
        'contains': ['pizza', 'lieblingsessen'],
        'pair_prefix': 'was ist dein lieblingsessen',
    }),
    ('food_magst_reis', 'food', 'magst du reis', {
        'contains': ['reis'],
        'min_kw_raw': 3.0,
    }),
    ('emotion_traurig', 'emotion', 'ich bin traurig', {
        'contains': ['leid', 'hoere', 'hier'],
        'min_words': 8,
    }),
    ('emotion_gluecklich', 'emotion', 'ich bin gluecklich', {
        'contains': ['freut', 'schoen', 'toll', 'gluecklich'],
        'min_words': 8,
    }),
    ('emotion_verzweifelt', 'emotion', 'ich bin verzweifelt', {
        'contains': ['leid', 'hoere', 'hier', 'los'],
        'min_words': 8,
    }),
    ('math_5plus3', 'math', 'was ist 5 plus 3', {
        'contains': ['8'],
        'pair_prefix': 'was ist 5 plus 3',
    }),
    ('math_7mal7', 'math', 'was ist 7 mal 7', {
        'contains': ['49'],
    }),
    ('math_9minus4', 'math', 'was ist 9 minus 4', {
        'contains': ['5'],
    }),
    ('math_aufgabe', 'math', 'stell mir eine aufgabe', {
        'contains': ['plus', 'minus', 'mal'],
        'min_words': 4,
    }),
    ('meta_werbistdu', 'meta', 'wer bist du', {
        'contains': ['kalle'],
        'min_words': 10,
    }),
    ('scope_english', 'meta', 'in english please', {
        'contains': ['deutsch'],
        'not_contains': ['english', 'please'],
    }),
    ('scope_offtopic', 'meta', 'was ist quantenphysik', {
        'contains': ['kann', 'nicht', 'sagen', 'einfacher', 'themen'],
    }),
    ('scope_garbage', 'meta', 'blabla asdf 12345', {
        'expect_fallback': True,
    }),
]

# Multi-turn: list of (chain_id, category, turns)
# Each turn: (query, assertions, check_state)
MULTI_TURN = [
    ('mt_food_followup', 'mt_food', [
        ('ich habe hunger', {
            'contains': ['essen', 'einkaufen', 'hunger'],
            'min_words': 5,
        }, None),
        ('toast', {
            'contains': ['toast'],
            'not_contains': ['reden', 'thema'],  # should NOT derail to generic chat
        }, {'lastBotTurn_contains': 'essen'}),
    ]),
    ('mt_food_lieblingsessen', 'mt_food', [
        ('was ist dein lieblingsessen', {
            'contains': ['pizza'],
        }, None),
        ('fisch', {
            'contains': ['fisch'],
            'pair_prefix': 'mein lieblingsessen',  # should match context-aware pair
        }, {'lastBotTurn_contains': 'pizza'}),
    ]),
    ('mt_hobby_followup', 'mt_hobby', [
        ('was ist dein lieblingshobby', {
            'contains': ['lesen'],
        }, None),
        ('schwimmen', {
            'contains': ['schwimmen'],
            'pair_prefix': 'mein lieblingshobby',
        }, {'lastBotTurn_contains': 'lesen'}),
    ]),
    ('mt_math_validation', 'mt_math', [
        ('was ist 4 plus 6', {
            'contains': ['10'],
        }, None),
        ('10', {
            'contains': ['richtig', '10'],
            'pair_prefix': 'plus 4 6 10',
        }, {'lastBotTurn_contains': 'plus'}),
    ]),
    ('mt_math_7mal7_49', 'mt_math', [
        ('was ist 7 mal 7', {
            'contains': ['49'],
        }, None),
        ('49', {
            'contains': ['richtig', '49'],
        }, {'lastBotTurn_contains': 'mal'}),
    ]),
    ('mt_emotion_chain', 'mt_emotion', [
        ('ich bin traurig', {
            'contains': ['leid', 'hoere', 'hier'],
        }, None),
        ('alles ist schlecht heute', {
            'contains': ['leid', 'traurig', 'hier', 'hoere', 'okay'],
            'not_contains': ['pizza', 'rechnen', 'aufgabe'],  # must NOT derail
        }, None),
    ]),
    ('mt_music_genre', 'mt_food', [  # using mt_food category as general mt
        ('magst du musik', {
            'contains': ['musik'],
        }, None),
        ('welches genre', {
            'not_contains': ['pizza', 'essen', 'rechnen'],  # must stay music-related
        }, {'lastBotTurn_contains': 'musik'}),
    ]),
]

# Edge cases
EDGE_CASES = [
    ('edge_typo', 'edge', 'stell ir ne aufgbe', {
        'contains': ['plus', 'minus', 'mal', 'aufgabe', 'rechne', 'ist'],  # should still match math-ish
        'min_words': 3,
    }),
    ('edge_insult', 'edge', 'du bist total bloed und doof', {
        'min_words': 3,  # just shouldn't crash; OOV insults are ignored
    }),
    ('edge_long', 'edge',
     'ich habe heute morgen ganz frueh aufgestanden und bin dann in die schule gegangen und da war es total langweilig und ich habe mich gelangweilt und dann bin ich nach hause gegangen und habe pizza gegessen', {
        'min_words': 3,  # shouldn't crash on long input
    }),
]

# ============================================================
# Test Runner
# ============================================================
async def extract_response(page, timeout_ms=10000):
    """Extract the last bot message text and metadata from the page."""
    t0 = time.time()
    await asyncio.sleep(0.5)  # initial wait for response to start
    # Wait for response to complete (no more typing animation)
    for _ in range(timeout_ms // 200):
        await asyncio.sleep(0.2)
        msgs = await page.locator('.m.b').all()
        if msgs:
            # Check if generation is done (finish() sets the .leg element)
            legs = await msgs[-1].locator('.leg').all()
            if legs:
                break
    elapsed_ms = int((time.time() - t0) * 1000)

    msgs = await page.locator('.m.b').all()
    if not msgs:
        return '', '', 0.0, 0.0, 0, True, elapsed_ms

    last = msgs[-1]
    full_text = await last.inner_text()

    # Parse response text (first line before metrics)
    lines = full_text.strip().split('\n')
    response_text = lines[0] if lines else ''

    # Parse ragInfo line: "Kalle · Xw · match: "pair" (kw=X sem=Y kwRaw=Z)"
    matched_pair = ''
    kw_raw = 0.0
    top3_pct = 0.0
    word_count = 0
    is_fallback = False

    for line in lines:
        # Match info
        m = re.search(r'match:\s*"([^"]*)"', line)
        if m:
            matched_pair = m.group(1)
        m = re.search(r'kwRaw=([0-9.]+)', line)
        if m:
            kw_raw = float(m.group(1))
        # Fallback
        if 'fallback' in line.lower():
            is_fallback = True
            m = re.search(r'kwMax=([0-9.]+)', line)
            if m:
                kw_raw = float(m.group(1))
        # Top-3
        m = re.search(r'(\d+)%\s*Top-3', line)
        if m:
            top3_pct = float(m.group(1))
        # Word count
        m = re.search(r'(\d+)w\s*·', line)
        if m:
            word_count = int(m.group(1))
        # Greeting
        m = re.search(r'greeting:\s*"([^"]*)"', line)
        if m:
            matched_pair = m.group(1)
            kw_raw = 99.0

    return response_text, matched_pair, kw_raw, top3_pct, word_count, is_fallback, elapsed_ms

def check_assertions(response_text, matched_pair, kw_raw, word_count, is_fallback, assertions):
    """Check assertions against response. Returns (passed, fail_reason)."""
    resp_lower = response_text.lower()

    # Contains check (ANY of the words)
    if 'contains' in assertions:
        found = any(w in resp_lower for w in assertions['contains'])
        if not found:
            return False, f'expected one of {assertions["contains"]} in response'

    # Not-contains check (NONE of the words)
    if 'not_contains' in assertions:
        found_bad = [w for w in assertions['not_contains'] if w in resp_lower]
        if found_bad:
            return False, f'unexpected words in response: {found_bad}'

    # Pair prefix check
    if 'pair_prefix' in assertions:
        if not matched_pair.lower().startswith(assertions['pair_prefix'].lower()):
            return False, f'pair "{matched_pair}" does not start with "{assertions["pair_prefix"]}"'

    # Fallback check
    if assertions.get('expect_fallback'):
        if not is_fallback:
            return False, f'expected fallback but got match: "{matched_pair}"'
    elif 'expect_fallback' not in assertions:
        pass  # no opinion on fallback

    # Min words
    if 'min_words' in assertions:
        if word_count < assertions['min_words']:
            return False, f'word count {word_count} < minimum {assertions["min_words"]}'

    # Min confidence
    if 'min_kw_raw' in assertions:
        if kw_raw < assertions['min_kw_raw']:
            return False, f'kwRaw {kw_raw:.1f} < minimum {assertions["min_kw_raw"]}'

    return True, None

async def run_tests(html_file, category_filter=None):
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context()
        page = await ctx.new_page()

        # Suppress console noise
        page.on('pageerror', lambda e: None)

        # --- SINGLE-TURN TESTS ---
        for sid, cat, query, assertions in SINGLE_TURN:
            if category_filter and cat != category_filter:
                continue

            await page.goto(f'file://{html_file}', wait_until='domcontentloaded')
            for _ in range(60):
                await asyncio.sleep(0.5)
                try:
                    s = (await page.locator('#st').inner_text()).lower()
                    if 'ready' in s: break
                except: pass

            await page.fill('#in', query)
            await page.click('#go')
            resp, pair, kw, top3, wc, fb, ms = await extract_response(page)
            passed, reason = check_assertions(resp, pair, kw, wc, fb, assertions)

            results.append(TestResult(
                scenario_id=sid, category=cat, query=query,
                response_text=resp[:200], matched_pair=pair,
                kw_raw=kw, top3_pct=top3, word_count=wc,
                is_fallback=fb, response_time_ms=ms,
                passed=passed, fail_reason=reason
            ))

        # --- MULTI-TURN TESTS ---
        for chain_id, cat, turns in MULTI_TURN:
            if category_filter and cat != category_filter:
                continue

            await page.goto(f'file://{html_file}', wait_until='domcontentloaded')
            for _ in range(60):
                await asyncio.sleep(0.5)
                try:
                    s = (await page.locator('#st').inner_text()).lower()
                    if 'ready' in s: break
                except: pass

            for turn_idx, (query, assertions, state_check) in enumerate(turns):
                # Check state BEFORE sending (if specified)
                if state_check and 'lastBotTurn_contains' in state_check:
                    try:
                        lbt = await page.evaluate('lastBotTurn || ""')
                        expected = state_check['lastBotTurn_contains']
                        if expected not in lbt.lower():
                            results.append(TestResult(
                                scenario_id=f'{chain_id}_t{turn_idx}', category=cat, query=query,
                                response_text=f'[STATE CHECK] lastBotTurn missing "{expected}"',
                                matched_pair='', kw_raw=0, top3_pct=0, word_count=0,
                                is_fallback=False, response_time_ms=0,
                                passed=False, fail_reason=f'lastBotTurn missing "{expected}": got "{lbt[:100]}"'
                            ))
                            continue
                    except Exception as e:
                        pass  # state check optional, don't block

                await page.fill('#in', query)
                await page.click('#go')
                resp, pair, kw, top3, wc, fb, ms = await extract_response(page)
                passed, reason = check_assertions(resp, pair, kw, wc, fb, assertions)

                results.append(TestResult(
                    scenario_id=f'{chain_id}_t{turn_idx}', category=cat, query=query,
                    response_text=resp[:200], matched_pair=pair,
                    kw_raw=kw, top3_pct=top3, word_count=wc,
                    is_fallback=fb, response_time_ms=ms,
                    passed=passed, fail_reason=reason
                ))

        # --- EDGE CASE TESTS ---
        for sid, cat, query, assertions in EDGE_CASES:
            if category_filter and cat != category_filter:
                continue

            await page.goto(f'file://{html_file}', wait_until='domcontentloaded')
            for _ in range(60):
                await asyncio.sleep(0.5)
                try:
                    s = (await page.locator('#st').inner_text()).lower()
                    if 'ready' in s: break
                except: pass

            try:
                await page.fill('#in', query)
                await page.click('#go')
                resp, pair, kw, top3, wc, fb, ms = await extract_response(page)
                passed, reason = check_assertions(resp, pair, kw, wc, fb, assertions)
            except Exception as e:
                resp, pair, kw, top3, wc, fb, ms = str(e), '', 0, 0, 0, False, 0
                passed, reason = False, f'CRASH: {e}'

            results.append(TestResult(
                scenario_id=sid, category=cat, query=query,
                response_text=resp[:200], matched_pair=pair,
                kw_raw=kw, top3_pct=top3, word_count=wc,
                is_fallback=fb, response_time_ms=ms,
                passed=passed, fail_reason=reason
            ))

        await browser.close()
    return results

# ============================================================
# Report
# ============================================================
def print_report(results, html_file, threshold):
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    rate = passed / total if total > 0 else 0

    # Category breakdown
    cats = {}
    for r in results:
        if r.category not in cats:
            cats[r.category] = {'total': 0, 'passed': 0}
        cats[r.category]['total'] += 1
        if r.passed:
            cats[r.category]['passed'] += 1

    # Performance
    times = [r.response_time_ms for r in results if r.response_time_ms > 0]
    avg_time = sum(times) / len(times) if times else 0
    max_time = max(times) if times else 0

    print(f'\n{"="*70}')
    print(f'KALLE REGRESSION TEST')
    print(f'{"="*70}')
    print(f'Build: {html_file}')
    print(f'Tests: {total} | Threshold: {threshold*100:.0f}%')
    print()

    # Detailed results
    current_cat = None
    for r in results:
        if r.category != current_cat:
            current_cat = r.category
            print(f'\n  {current_cat.upper()}:')
        status = 'PASS' if r.passed else 'FAIL'
        pair_info = f'pair="{r.matched_pair[:35]}"' if r.matched_pair else 'fallback'
        print(f'    [{status}] {r.scenario_id:30s} | {r.query[:30]:30s} | {pair_info} | kwRaw={r.kw_raw:.1f} | {r.word_count}w | {r.response_time_ms}ms')
        if not r.passed:
            print(f'           REASON: {r.fail_reason}')

    # Summary
    print(f'\n{"="*70}')
    print(f'SUMMARY: {passed}/{total} PASS ({rate*100:.1f}%)')
    print(f'{"="*70}')
    print(f'  Categories:')
    for cat, info in sorted(cats.items()):
        cr = info['passed'] / info['total'] if info['total'] > 0 else 0
        bar = '█' * info['passed'] + '░' * (info['total'] - info['passed'])
        print(f'    {cat:15s}: {info["passed"]}/{info["total"]} {bar} ({cr*100:.0f}%)')
    print(f'\n  Performance: avg {avg_time:.0f}ms/query, max {max_time:.0f}ms')
    print(f'  Threshold: {threshold*100:.0f}% → {"PASS ✓" if rate >= threshold else "FAIL ✗"}')
    print()

    return rate >= threshold

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Kalle KRR-Chatbot Regression Tests')
    parser.add_argument('html_file', nargs='?', default='/tmp/kalle-ernsthaft.html',
                        help='Path to kalle HTML file')
    parser.add_argument('--filter', type=str, default=None,
                        help='Filter by category (greeting, food, emotion, math, meta, mt_food, mt_math, mt_emotion, mt_hobby, edge)')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Minimum pass rate (default: 0.8 = 80%%)')
    parser.add_argument('--json', type=str, default=None,
                        help='Write JSON report to file')
    args = parser.parse_args()

    results = asyncio.run(run_tests(args.html_file, args.filter))
    success = print_report(results, args.html_file, args.threshold)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
        print(f'JSON report: {args.json}')

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Ambiguity Spectrum Test: KRR accuracy vs. sequence determinism.

Tests the hypothesis: KRR achieves high accuracy on low-ambiguity data
and degrades as ambiguity increases. All tests use the SAME architecture
(1L attention, RFF, KRR solve) — only the data changes.

Test 1: Arithmetic (zero ambiguity)
Test 2: SCAN navigation (zero ambiguity, real benchmark)
Test 3: Reber Grammar (very low ambiguity)
Test 4: Simple patterns (zero ambiguity)
"""
import os, sys, time, math, pickle
import numpy as np

# We'll use a simplified version of the KRR pipeline — no GPU needed,
# small enough for CPU. Character/word-level tokenization (no BPE needed).

def train_and_eval_krr(train_tokens, val_tokens, vocab, ctx=16, D=2048, sigma=1.5, lam=1e-5, emb_dim=32):
    """Train a minimal KRR-LM and return Val Top-1/Top-5."""
    V = len(vocab)
    W2I = {w: i for i, w in enumerate(vocab)}

    # Random embeddings (small model, no W2V needed)
    np.random.seed(42)
    emb = np.random.randn(V, emb_dim).astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    FEAT = ctx * emb_dim
    omega = np.random.randn(FEAT, D).astype(np.float32) / sigma
    bias = np.random.rand(D).astype(np.float32) * 2 * np.pi
    scale = math.sqrt(2.0 / D)

    # Position weights
    pos_w = np.linspace(0.4, 1.0, ctx, dtype=np.float32)

    def encode(tokens, t):
        start = max(0, t - ctx)
        c = tokens[start:t]
        vec = np.zeros(FEAT, dtype=np.float32)
        for i, tok in enumerate(c):
            slot = ctx - len(c) + i
            if tok in W2I:
                vec[slot*emb_dim:(slot+1)*emb_dim] = pos_w[slot] * emb[W2I[tok]]
        return vec

    # Train
    N = len(train_tokens) - ctx
    print(f"    Training: N={N:,}, V={V}, D={D}, FEAT={FEAT}")
    t0 = time.time()

    ZtZ = np.zeros((D, D), dtype=np.float64)
    ZtY = np.zeros((D, V), dtype=np.float64)

    for i in range(N):
        f = encode(train_tokens, i + ctx)
        z = scale * np.cos(f @ omega + bias).astype(np.float64)
        ZtZ += np.outer(z, z)
        target = W2I.get(train_tokens[i + ctx], 0)
        ZtY[:, target] += z

    ZtZ += lam * np.eye(D)
    W = np.linalg.solve(ZtZ, ZtY).astype(np.float32)
    t_train = time.time() - t0
    print(f"    Trained in {t_train:.1f}s")

    # Eval
    N_val = len(val_tokens) - ctx
    n_eval = min(5000, N_val)
    sample = np.random.choice(N_val, n_eval, replace=False) if N_val > n_eval else range(N_val)
    top1 = top5 = 0
    for i in sample:
        f = encode(val_tokens, i + ctx)
        z = scale * np.cos(f @ omega + bias)
        scores = z @ W
        pred = np.argmax(scores)
        target = W2I.get(val_tokens[i + ctx], 0)
        if pred == target: top1 += 1
        if V <= 5:
            top5_idx = np.argsort(-scores)[:min(5,V)]
        else:
            top5_idx = np.argpartition(-scores, 5)[:5]
        if target in top5_idx: top5 += 1

    t1 = top1 / len(sample)
    t5 = top5 / len(sample)
    return t1, t5, t_train


# ================================================================
# TEST 1: ARITHMETIC (zero ambiguity)
# ================================================================
def generate_arithmetic(n_examples=5000):
    """Generate simple arithmetic: '3 + 5 = 8 .'"""
    np.random.seed(42)
    tokens = []
    for _ in range(n_examples):
        op = np.random.choice(['+', '-', '*'])
        if op == '+':
            a, b = np.random.randint(0, 10, 2)
            r = a + b
        elif op == '-':
            a, b = sorted(np.random.randint(0, 10, 2), reverse=True)
            r = a - b
        else:
            a, b = np.random.randint(0, 5, 2)
            r = a * b
        tokens.extend([str(a), op, str(b), '=', str(r), '.'])
    return tokens

print("=" * 60)
print("TEST 1: ARITHMETIC (zero ambiguity)")
print("=" * 60)
arith_tokens = generate_arithmetic(8000)
vocab_arith = sorted(set(arith_tokens))
split = int(len(arith_tokens) * 0.9)
t1, t5, tt = train_and_eval_krr(
    arith_tokens[:split], arith_tokens[split:],
    vocab_arith, ctx=8, D=1024, emb_dim=16
)
print(f"    >>> Val Top-1: {t1*100:.1f}%, Top-5: {t5*100:.1f}%")
print()


# ================================================================
# TEST 2: DETERMINISTIC PATTERNS (zero ambiguity)
# ================================================================
def generate_patterns(n_repeats=2000):
    """Generate deterministic repeating patterns."""
    patterns = [
        "die katze sitzt auf der matte .".split(),
        "der hund liegt auf dem boden .".split(),
        "die sonne scheint am himmel .".split(),
        "der mond leuchtet in der nacht .".split(),
        "das kind spielt im garten .".split(),
        "der vogel singt auf dem baum .".split(),
        "die blume wächst im feld .".split(),
        "der fisch schwimmt im wasser .".split(),
        "die maus läuft durch das haus .".split(),
        "der wind weht über das land .".split(),
    ]
    tokens = []
    for _ in range(n_repeats):
        for p in patterns:
            tokens.extend(p)
    return tokens

print("=" * 60)
print("TEST 2: SIMPLE PATTERNS (zero ambiguity, 10 sentences)")
print("=" * 60)
pattern_tokens = generate_patterns(1500)
vocab_pattern = sorted(set(pattern_tokens))
split = int(len(pattern_tokens) * 0.9)
t1, t5, tt = train_and_eval_krr(
    pattern_tokens[:split], pattern_tokens[split:],
    vocab_pattern, ctx=8, D=1024, emb_dim=16
)
print(f"    >>> Val Top-1: {t1*100:.1f}%, Top-5: {t5*100:.1f}%")
print()


# ================================================================
# TEST 3: REBER GRAMMAR (very low ambiguity)
# ================================================================
def generate_reber(n_sequences=5000):
    """Generate sequences from the Reber Grammar FSA."""
    # States: 0=start, 1, 2, 3, 4, 5=end
    # Transitions: state -> [(char, next_state), ...]
    transitions = {
        0: [('B', 1)],
        1: [('T', 2), ('P', 3)],
        2: [('S', 2), ('X', 4)],
        3: [('T', 3), ('V', 5)],
        4: [('S', 6), ('X', 3)],
        5: [('P', 6), ('V', 4)],
        6: [('E', -1)],
    }
    rng = np.random.default_rng(42)
    all_tokens = []
    for _ in range(n_sequences):
        state = 0
        seq = []
        while state != -1:
            options = transitions[state]
            char, next_state = options[rng.integers(len(options))]
            seq.append(char)
            state = next_state
        all_tokens.extend(seq)
        all_tokens.append('.')  # separator
    return all_tokens

print("=" * 60)
print("TEST 3: REBER GRAMMAR (very low ambiguity, 7 tokens)")
print("=" * 60)
reber_tokens = generate_reber(8000)
vocab_reber = sorted(set(reber_tokens))
split = int(len(reber_tokens) * 0.9)
t1, t5, tt = train_and_eval_krr(
    reber_tokens[:split], reber_tokens[split:],
    vocab_reber, ctx=12, D=1024, emb_dim=16
)
print(f"    >>> Val Top-1: {t1*100:.1f}%, Top-5: {t5*100:.1f}%")
print()


# ================================================================
# TEST 4: DYCK LANGUAGE (balanced brackets)
# ================================================================
def generate_dyck(n_sequences=5000, max_depth=4):
    """Generate balanced bracket sequences."""
    rng = np.random.default_rng(42)
    all_tokens = []
    for _ in range(n_sequences):
        seq = []
        depth = 0
        length = rng.integers(2, 12)
        for _ in range(length):
            if depth == 0:
                seq.append('(')
                depth += 1
            elif depth >= max_depth:
                seq.append(')')
                depth -= 1
            else:
                if rng.random() < 0.5:
                    seq.append('(')
                    depth += 1
                else:
                    seq.append(')')
                    depth -= 1
        while depth > 0:
            seq.append(')')
            depth -= 1
        all_tokens.extend(seq)
        all_tokens.append('.')
    return all_tokens

print("=" * 60)
print("TEST 4: DYCK-1 LANGUAGE (balanced brackets, 3 tokens)")
print("=" * 60)
dyck_tokens = generate_dyck(8000)
vocab_dyck = sorted(set(dyck_tokens))
split = int(len(dyck_tokens) * 0.9)
t1, t5, tt = train_and_eval_krr(
    dyck_tokens[:split], dyck_tokens[split:],
    vocab_dyck, ctx=16, D=1024, emb_dim=16
)
print(f"    >>> Val Top-1: {t1*100:.1f}%, Top-5: {t5*100:.1f}%")
print()


# ================================================================
# TEST 5: COUNTING (fully deterministic)
# ================================================================
def generate_counting(max_n=20, n_repeats=500):
    """Generate counting sequences: 1 2 3 4 5 . 1 2 3 4 5 ."""
    tokens = []
    for _ in range(n_repeats):
        for n in range(1, max_n + 1):
            tokens.append(str(n))
        tokens.append('.')
    return tokens

print("=" * 60)
print("TEST 5: COUNTING 1-20 (fully deterministic)")
print("=" * 60)
count_tokens = generate_counting(20, 500)
vocab_count = sorted(set(count_tokens), key=lambda x: (x != '.', int(x) if x != '.' else -1))
split = int(len(count_tokens) * 0.9)
t1, t5, tt = train_and_eval_krr(
    count_tokens[:split], count_tokens[split:],
    vocab_count, ctx=8, D=512, emb_dim=16
)
print(f"    >>> Val Top-1: {t1*100:.1f}%, Top-5: {t5*100:.1f}%")
print()


# ================================================================
# TEST 6: MIXED AMBIGUITY (same start, different continuations)
# ================================================================
def generate_mixed_ambiguity(n_repeats=1000):
    """Sentences that START the same but END differently — controlled ambiguity."""
    templates = [
        "der mann geht nach hause .".split(),
        "der mann geht zum markt .".split(),
        "der mann geht in den wald .".split(),
        "die frau geht nach hause .".split(),
        "die frau geht zum markt .".split(),
        "die frau geht in den wald .".split(),
        "das kind geht nach hause .".split(),
        "das kind geht zum markt .".split(),
        "das kind geht in den wald .".split(),
    ]
    rng = np.random.default_rng(42)
    tokens = []
    for _ in range(n_repeats):
        for t in templates:
            tokens.extend(t)
    return tokens

print("=" * 60)
print("TEST 6: CONTROLLED AMBIGUITY (same prefix, 3 possible endings)")
print("=" * 60)
mixed_tokens = generate_mixed_ambiguity(1500)
vocab_mixed = sorted(set(mixed_tokens))
split = int(len(mixed_tokens) * 0.9)
t1, t5, tt = train_and_eval_krr(
    mixed_tokens[:split], mixed_tokens[split:],
    vocab_mixed, ctx=8, D=1024, emb_dim=16
)
print(f"    >>> Val Top-1: {t1*100:.1f}%, Top-5: {t5*100:.1f}%")
print()


# ================================================================
# SUMMARY
# ================================================================
print("=" * 60)
print("AMBIGUITY SPECTRUM — ZUSAMMENFASSUNG")
print("=" * 60)
print()
print("Die These: KRR-Accuracy ist eine Funktion der Sequenz-Entropie.")
print()
print("(Bibel und Wikipedia aus früheren Experimenten ergänzt)")

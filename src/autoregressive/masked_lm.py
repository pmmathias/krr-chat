#!/usr/bin/env python3
"""
Phase 3: Bidirectional KRR Masked Language Model.

Combines forward (left→right) and backward (right→left) KRR models
to fill [MASK] slots in templates with contextually appropriate words.

For each [MASK]:
  - Forward model scores P_fwd(word | left context)
  - Backward model scores P_bwd(word | right context)
  - Combined: log P(word) = log P_fwd + log P_bwd → argmax

Usage:
  # Evaluate on template slot-filling accuracy
  python masked_lm.py eval --forward model_moe.pkl --backward model_backward.pkl

  # Fill specific template
  python masked_lm.py fill --forward model_moe.pkl --backward model_backward.pkl \
      --template "The [MASK] transforms data into a [MASK] space."

  # Interactive demo
  python masked_lm.py demo --forward model_moe.pkl --backward model_backward.pkl
"""
import os, sys, time, math, argparse, pickle
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description='Bidirectional KRR Masked LM')
    sub = p.add_subparsers(dest='command')

    pe = sub.add_parser('eval')
    pe.add_argument('--forward', required=True, help='Forward model (MoE)')
    pe.add_argument('--backward', required=True, help='Backward model')
    pe.add_argument('--templates', default='data/autoregressive/templates.pkl')
    pe.add_argument('--tokenizer', default='data/autoregressive/bpe_large.json')
    pe.add_argument('--n-eval', type=int, default=500)
    pe.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    pf = sub.add_parser('fill')
    pf.add_argument('--forward', required=True)
    pf.add_argument('--backward', required=True)
    pf.add_argument('--tokenizer', default='data/autoregressive/bpe_large.json')
    pf.add_argument('--template', required=True)
    pf.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    return p.parse_args()


class KRRPredictor:
    """Wrapper for a KRR model (forward or backward, with MoE)."""

    def __init__(self, model_path, device, direction='forward'):
        with open(model_path, 'rb') as f:
            m = pickle.load(f)
        self.cfg = m['config']
        self.direction = direction
        self.dev = device

        self.emb = torch.from_numpy(m['emb']).to(device)
        self.pe = torch.from_numpy(m['pe']).to(device)
        self.omega = torch.from_numpy(m['omega']).to(device)
        self.bias = torch.from_numpy(m['bias']).to(device)
        self.attn = {k: torch.from_numpy(v).to(device) for k, v in m['attn'].items()}

        # MoE components
        self.expert_W = [torch.from_numpy(w).to(device) for w in m['expert_W']]
        self.centers = torch.from_numpy(m['centers']).to(device)
        self.W_global = torch.from_numpy(m['W_global']).to(device)

        CTX = self.cfg['CTX']
        self.causal_mask = torch.triu(torch.ones(CTX, CTX, device=device), diagonal=1).bool()
        self.rff_scale = math.sqrt(2.0 / self.cfg['D'])

    def predict_scores(self, token_ids):
        """Given a sequence of token IDs, predict scores for the NEXT token.

        For forward model: predicts token after the sequence.
        For backward model: input should be REVERSED right-context.
        Returns: (V,) numpy array of log-probabilities.
        """
        cfg = self.cfg
        CTX, EMB, H, DK, DV = cfg['CTX'], cfg['EMB_DIM'], cfg['N_HEADS'], cfg['D_K'], cfg['D_V']

        ids_t = torch.tensor(token_ids, dtype=torch.long, device=self.dev)
        ctx = ids_t[-CTX:] if len(ids_t) >= CTX else ids_t
        n = len(ctx)

        E = self.emb[ctx].unsqueeze(0)
        X = E + self.pe[-n:].unsqueeze(0)

        Q_ = (X @ self.attn['W_Q']).view(1, n, H, DK).permute(0, 2, 1, 3)
        K_ = (X @ self.attn['W_K']).view(1, n, H, DK).permute(0, 2, 3, 1)
        Vv_ = (X @ self.attn['W_V']).view(1, n, H, DV).permute(0, 2, 1, 3)
        sc = (Q_ @ K_) / math.sqrt(DK)
        sc.masked_fill_(self.causal_mask[:n, :n].unsqueeze(0).unsqueeze(0), float('-inf'))
        ao = (torch.softmax(sc, -1) @ Vv_).permute(0, 2, 1, 3).reshape(1, n, H*cfg['D_V']) @ self.attn['W_O']
        X1 = F.layer_norm(X + ao, [EMB])
        h = torch.relu(X1 @ self.attn['W_in'])
        X2 = F.layer_norm(X1 + h @ self.attn['W_out'], [EMB])
        feat = torch.cat([X2[0, -1], E[0, -1]])

        with torch.no_grad():
            z = self.rff_scale * torch.cos(feat @ self.omega + self.bias)
            # MoE routing
            d = torch.cdist(feat.unsqueeze(0), self.centers)
            eidx = d.argmin().item()
            scores = (z @ self.expert_W[eidx]).squeeze()

        # Log-softmax
        log_probs = torch.log_softmax(scores, dim=0).cpu().numpy()
        return log_probs


def fill_template(fwd_model, bwd_model, template_tokens, mask_positions, tokenizer):
    """Fill [MASK] slots using bidirectional prediction.

    template_tokens: list of token IDs with -1 at mask positions
    mask_positions: list of positions that are masked
    Returns: filled token IDs, per-slot scores
    """
    filled = list(template_tokens)
    slot_results = []

    for pos in mask_positions:
        # Left context (for forward model)
        left_ctx = [t for t in template_tokens[:pos] if t != -1]

        # Right context (for backward model — reversed)
        right_ctx = [t for t in template_tokens[pos+1:] if t != -1]
        right_ctx_reversed = right_ctx[::-1]

        # Forward prediction
        with torch.no_grad():
            fwd_scores = fwd_model.predict_scores(left_ctx) if left_ctx else np.zeros(fwd_model.cfg['V'])

        # Backward prediction
        with torch.no_grad():
            bwd_scores = bwd_model.predict_scores(right_ctx_reversed) if right_ctx_reversed else np.zeros(bwd_model.cfg['V'])

        # Combined bidirectional score
        combined = fwd_scores + bwd_scores
        best_tid = int(np.argmax(combined))
        best_word = tokenizer.decode([best_tid])

        # Top-5 for analysis
        top5_idx = np.argpartition(-combined, 5)[:5]
        top5_idx = top5_idx[np.argsort(-combined[top5_idx])]
        top5_words = [(tokenizer.decode([int(i)]), combined[i]) for i in top5_idx]

        filled[pos] = best_tid
        slot_results.append({
            'position': pos,
            'predicted_tid': best_tid,
            'predicted_word': best_word,
            'fwd_score': float(fwd_scores[best_tid]),
            'bwd_score': float(bwd_scores[best_tid]),
            'combined_score': float(combined[best_tid]),
            'top5': top5_words,
        })

    return filled, slot_results


def eval_templates(args):
    """Evaluate slot-filling accuracy on extracted templates."""
    dev = torch.device(args.device)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)

    print("Loading forward model...")
    fwd = KRRPredictor(args.forward, dev, 'forward')
    print("Loading backward model...")
    bwd = KRRPredictor(args.backward, dev, 'backward')

    print("Loading templates...")
    with open(args.templates, 'rb') as f:
        data = pickle.load(f)
    templates = data['templates']
    print(f"  {len(templates)} templates")

    # Evaluate
    n_eval = min(args.n_eval, len(templates))
    sample = np.random.choice(len(templates), n_eval, replace=False)

    total_slots = 0
    correct_top1 = 0
    correct_top5 = 0
    correct_fwd_only = 0
    correct_bwd_only = 0

    print(f"\nEvaluating {n_eval} templates...")
    for idx, ti in enumerate(sample):
        t = templates[ti]
        filled, results = fill_template(
            fwd, bwd, t['template_tokens'], t['mask_positions'], tokenizer)

        for r in results:
            total_slots += 1
            original_tid = t['original_at_masks'][r['position']]
            if r['predicted_tid'] == original_tid:
                correct_top1 += 1
            top5_tids = [int(np.argmax(np.array([s for _, s in r['top5']]))) for _ in range(5)]
            # Actually check top5 properly
            combined_scores = np.zeros(fwd.cfg['V'])
            left_ctx = [tok for tok in t['template_tokens'][:r['position']] if tok != -1]
            right_ctx = [tok for tok in t['template_tokens'][r['position']+1:] if tok != -1]
            with torch.no_grad():
                fwd_sc = fwd.predict_scores(left_ctx) if left_ctx else np.zeros(fwd.cfg['V'])
                bwd_sc = bwd.predict_scores(right_ctx[::-1]) if right_ctx else np.zeros(fwd.cfg['V'])
            combined_scores = fwd_sc + bwd_sc
            top5_idx = np.argpartition(-combined_scores, 5)[:5]
            if original_tid in top5_idx:
                correct_top5 += 1
            # Forward-only baseline
            if int(np.argmax(fwd_sc)) == original_tid:
                correct_fwd_only += 1
            if int(np.argmax(bwd_sc)) == original_tid:
                correct_bwd_only += 1

        if idx % 50 == 0 and idx > 0:
            print(f"  {idx}/{n_eval}: Bidi Top-1={correct_top1/total_slots*100:.1f}%, "
                  f"Fwd-only={correct_fwd_only/total_slots*100:.1f}%, "
                  f"Bwd-only={correct_bwd_only/total_slots*100:.1f}%")

        # Show some examples
        if idx < 5:
            print(f"\n  Example {idx+1}:")
            print(f"    Original: {t['original_text'][:120]}")
            filled_text = tokenizer.decode([tok for tok in filled if tok != -1])
            print(f"    Filled:   {filled_text[:120]}")
            for r in results:
                orig_word = tokenizer.decode([t['original_at_masks'][r['position']]])
                print(f"    Slot @{r['position']}: pred={r['predicted_word']!r} "
                      f"orig={orig_word!r} "
                      f"{'✓' if r['predicted_tid'] == t['original_at_masks'][r['position']] else '✗'}")

    print(f"\n{'='*60}")
    print(f"SLOT-FILLING RESULTS ({n_eval} templates, {total_slots} slots)")
    print(f"  Bidirectional Top-1: {correct_top1/total_slots*100:.1f}%")
    print(f"  Bidirectional Top-5: {correct_top5/total_slots*100:.1f}%")
    print(f"  Forward-only Top-1:  {correct_fwd_only/total_slots*100:.1f}%")
    print(f"  Backward-only Top-1: {correct_bwd_only/total_slots*100:.1f}%")
    print(f"  Gain from bidi:      +{(correct_top1-max(correct_fwd_only,correct_bwd_only))/total_slots*100:.1f}pp")
    print(f"{'='*60}")


def main():
    args = parse_args()
    if args.command == 'eval':
        eval_templates(args)
    elif args.command == 'fill':
        print("Single-template filling — coming soon")
    else:
        print("Usage: python masked_lm.py {eval|fill|demo}")


if __name__ == '__main__':
    main()

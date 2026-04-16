# T040: Multi-Layer Attention (Stacked Transformer-style)

**Status:** TODO
**Priorität:** P0 (erwartete größte Einzelverbesserung)
**Parent:** T038
**Geschätzter Aufwand:** 4h

## Hypothese

**Das ist der große Unlock.** Eine einzige Attention-Schicht kann nur lineare Interaktionen zwischen Token-Embeddings modellieren. Zwei oder drei Schichten (jede mit eigenen fresh random Q/K/V und eigenem MLP-ähnlichem Zwischenschritt) erlauben **hierarchische Feature-Komposition** — Syntax auf Layer 1, Semantik auf Layer 2, topische Kohärenz auf Layer 3.

Standard-Transformer-Rezept: $x' = x + \text{Attention}(x)$, dann $x'' = x' + \text{FFN}(x')$. Wir halten den Residual-Connection, ersetzen aber die FFN durch eine fixed random 2-layer projection (ReLU zwischen zwei Random-Matrices). Kein Backprop, alles closed-form am Ende.

**Erwartet:** Val Top-1: 9.1% → **15–20%** (der große Sprung).

## Architektur-Skizze

```
x₀ = embedding(tokens) + positional_encoding
x₁ = LayerNorm(x₀ + MultiHeadAttn₁(x₀, W_Q₁, W_K₁, W_V₁))
x₁' = LayerNorm(x₁ + FixedFFN₁(x₁, W_in₁, W_out₁))
x₂ = LayerNorm(x₁' + MultiHeadAttn₂(x₁', W_Q₂, W_K₂, W_V₂))
x₂' = LayerNorm(x₂ + FixedFFN₂(x₂, W_in₂, W_out₂))
[... optional x₃ ...]
c_t = x_final[t-1]  (attention output at last position)
z_t = RFF(c_t)
predict via KRR(W·z_t)
```

Alle W_Q, W_K, W_V, W_in, W_out sind **random, fixed**. LayerNorm ist deterministisch (mean/variance per-token, keine learned affine).

## Schritte

1. Implementiere `FixedFFN`: $\text{FFN}(x) = W_{\text{out}} \cdot \text{ReLU}(W_{\text{in}} \cdot x)$ mit random init
2. Implementiere Layer-Stack in `train_ar_multilayer.py`
3. Starte mit 2 Layern, d_model=64, d_ff=256, 4 heads × d_k=32 × d_v=64
4. Messung (Train/Val Top-1/Top-5, Prompt-Suite)
5. Falls vielversprechend: 3 Layer testen

## Akzeptanzkriterien

- [ ] 2-Layer Variante trainiert und evaluiert
- [ ] 3-Layer Variante getestet
- [ ] Val Top-1 ≥ 12% (wenn nicht: Layer-Tiefe war nicht der Hebel)
- [ ] Soft-Score auf Prompt-Suite dokumentiert
- [ ] Entscheidung: 2 oder 3 Layer als Standard für folgende Experimente

## Risiken

- **Compute**: pro Layer verdoppelt sich die Accumulation-Zeit. 2 Layer = ~2 min, 3 Layer = ~3 min Training. Akzeptabel.
- **Feature-Dimension explodiert** nicht, weil wir immer nur den finalen State an Position $t$ nutzen (nicht die Konkatenation aller Layers).
- **LayerNorm ohne learned affine** kann zu Training-Instabilität führen. Mitigation: clip activations oder zusätzlichen random bias/scale.

## Referenzen

- Vaswani et al. 2017 "Attention is All You Need" — Transformer-Architektur
- Peng et al. 2021 "Random Feature Attention" — Random Q/K
- Sun et al. 2023 "RetNet" — retention als Alternative zu Attention

## Output

- `src/autoregressive/train_ar_multilayer.py`
- `src/autoregressive/generate_multilayer.py`
- Modell-Pickle `model_attention_multilayer.pkl` (gitignored)
- CSV-Eintrag

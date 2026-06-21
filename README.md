# ml-foundations

From-scratch implementations of core ML primitives — written by hand (forward **and**
backward passes), then gradient-checked against PyTorch autograd. The point is to
implement each piece cold, the way frontier-lab ML interviews test.

Each week is one primitive: a clean implementation + a gradient/correctness check that
verifies it matches a reference to ~1e-6.

## Progress

| # | Primitive | Status | Check |
|---|-----------|--------|-------|
| 1 | 3-layer MLP — forward + **manual backward** (no autograd) | ✅ | gradient-check vs autograd, 0 error |
| 2 | Softmax + cross-entropy (log-sum-exp), manual backward | — | |
| 3 | AdamW optimizer from scratch | — | |
| 4 | LayerNorm forward + manual backward | — | |
| 5 | Multi-head self-attention | — | |
| 6 | BPE tokenizer | — | |
| 7 | KV cache for autoregressive decoding | — | |
| 8 | INT4 round-to-nearest quantizer (group size 128) | — | |
| 9 | DPO loss | — | |
| 10 | FlashAttention online-softmax (math walkthrough) | — | |

## Running

```bash
python week01_mlp.py   # prints a per-parameter gradient check; expects "GRADIENT CHECK PASSED"
```

Requires `torch`.

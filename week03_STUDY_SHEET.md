# Week 3 — AdamW optimizer from scratch (study sheet)

**The interview prompt:** "Implement AdamW from scratch. No `torch.optim`. Show me it matches
the reference." That's it. AdamW is ~10 lines of real math; the whole skill is knowing the
update rule cold and getting the *weight-decay decoupling* right (that's the one thing people
miss).

## What you're actually writing

An optimizer holds two running averages **per parameter** and applies one update per step.
For each parameter tensor θ with gradient g at step t:

1. **First moment (mean of grads):** `m = β1·m + (1−β1)·g`
2. **Second moment (mean of squared grads):** `v = β2·v + (1−β2)·g²`
3. **Bias correction** (m and v start at 0, so they're biased toward 0 early):
   `m̂ = m / (1 − β1^t)`,  `v̂ = v / (1 − β2^t)`
4. **Update:** `θ ← θ − lr · m̂ / (√v̂ + ε)`

That's **Adam**. Intuition: `m̂` is the direction (smoothed gradient), `√v̂` is a per-parameter
scale — parameters with big/noisy gradients get smaller effective steps. It's per-coordinate
adaptive learning rates.

## The one thing that makes it AdamW, not Adam

**Weight decay is decoupled from the gradient.** Plain Adam-with-L2 would add `wd·θ` *into g*,
which then gets divided by `√v̂` — so decay gets scaled weirdly per-parameter. AdamW instead
shrinks the weights directly, outside the adaptive step:

```
θ ← θ − lr·wd·θ        # decoupled decay, applied to θ first
θ ← θ − lr · m̂/(√v̂+ε)  # then the normal Adam step
```

To match `torch.optim.AdamW` exactly, do the decay **first**, using θ before the Adam step,
scaled by `lr·wd`. Defaults: `lr=1e-3, β1=0.9, β2=0.999, ε=1e-8, wd=0.01`.

## State you must keep

- `t`  — global step count (starts at 0, increment to 1 on the first step; bias correction
  needs `t ≥ 1`).
- `m`, `v` — one tensor each, same shape as the parameter, initialized to zeros. If you have
  several parameters, you keep a separate (m, v) for each.

## What to write (the skeleton)

```
class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01)
    def zero_grad(self)      # set each p.grad to None/zero
    def step(self)           # the 4 lines above + decoupled decay, for every param
```

Run under `torch.no_grad()` inside `step()` and mutate params in place (`p -= ...` or
`p.add_(..., alpha=...)`), because you're editing the leaf tensors, not building graph.

## Gotchas (each one is a real bug people ship)

- Increment `t` **once per step**, before bias correction — not per parameter.
- `v̂` can be tiny early → `√v̂ + ε`, never `√(v̂ + ε)`. The ε is *outside* the sqrt in Adam.
- Decoupled decay uses `lr·wd·θ`, not `wd·θ`. Wrong scaling = won't match torch.
- Do it in place under `no_grad()`, and `zero_grad()` between steps or grads accumulate.
- `β1^t` uses the running step t, so step 1 → `1−β1`, which is large; correction matters most
  in the first ~50 steps.

## The three files & how to run

| File | What it is | When |
|---|---|---|
| `week03_STUDY_SHEET.md` | this file — concepts + how to run + the full solution | read first / check answer |
| `week03_adamw.py` | **skeleton**: the class shell with the `step()` logic as empty slots | the scaffold pass |
| `week03_BLANK_CHALLENGE.py` | **really blank**: prompt + the check, nothing else | the cold pass |

Both `.py` files end with a check that optimizes the same toy least-squares problem with YOUR
`AdamW` and with `torch.optim.AdamW` from identical init, and compares parameters after 200
steps. Max abs diff `< 1e-6` = correct.

```
python week03_adamw.py             # skeleton — fill the slots, then it self-checks
python week03_BLANK_CHALLENGE.py   # cold — write the whole class, then it self-checks
```

(If it drifts: you almost certainly have the decay order/scale or the bias-correction wrong.)

**Cold-pass criterion for the tracker:** you can write the `AdamW` class from the blank file,
AI off, and pass the check first or second try. Then hit "✓ AdamW done" on the tracker.

## Full solution (the answer key — try the skeleton before reading)

```python
import torch

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr, self.eps, self.wd = lr, eps, weight_decay
        self.b1, self.b2 = betas
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1                                  # once per step, before bias correction
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            p.mul_(1 - self.lr * self.wd)            # (1) decoupled decay FIRST
            self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)          # (2) m
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)   #     v
            m_hat = self.m[i] / bc1                  # (3) bias correction
            v_hat = self.v[i] / bc2
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)  # (4) eps outside sqrt
```

Line-by-line this is exactly the four-step rule above, with decoupled decay applied first. The
in-place ops (`mul_`, `add_`, `addcmul_`, `addcdiv_`) just avoid allocating new tensors; you
could write `self.m[i] = self.b1*self.m[i] + (1-self.b1)*g` and it'd match too.

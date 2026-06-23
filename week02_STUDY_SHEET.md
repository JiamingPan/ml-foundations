# Week 2 Study Sheet — Softmax + Cross-Entropy (log-sum-exp), manual backward

**Goal:** implement softmax and cross-entropy loss with the log-sum-exp trick for numerical
stability, plus the manual backward pass (no autograd). This is the loss at the end of every
classifier and LLM. Read this (~15 min), then fill the blanks in `week02_softmax_ce.py`.

---

## 1. What softmax does
Turns a vector of logits `z` into a probability distribution: `p_i = exp(z_i) / Σ_j exp(z_j)`.
Non-negative, sums to 1. The output is "how much probability the model puts on each class."

## 2. The numerical problem (and the log-sum-exp fix)
Naive `exp(z_i)` overflows for large logits: `exp(1000) = inf` → NaN everywhere. Fix: subtract
the max before exponentiating. Let `m = max(z)`. Then:
```
softmax(z)_i = exp(z_i − m) / Σ_j exp(z_j − m)
```
This is **mathematically identical** (the `exp(−m)` cancels top and bottom) but now the biggest
exponent is `exp(0) = 1` — no overflow. The largest underflow risk is also controlled.

For the loss you don't even need softmax explicitly — use **log-softmax** directly:
```
log_softmax(z)_i = z_i − logsumexp(z)      where logsumexp(z) = m + log Σ_j exp(z_j − m)
```
This is stabler than `log(softmax(z))` and is what you'll use for the loss.

## 3. Cross-entropy loss
For a true class `y` (an integer label), the loss for one example is the negative log-prob of
the correct class:
```
L = − log p_y = − log_softmax(z)_y = logsumexp(z) − z_y
```
Average over the batch. That `logsumexp(z) − z_y` form is the clean, stable way to compute it —
no explicit softmax, no log of a tiny number.

## 4. The backward pass (the famously clean gradient)
The gradient of cross-entropy-with-softmax w.r.t. the **logits** `z` is:
```
dL/dz_i = p_i − 1{i == y}          (predicted prob minus the one-hot target)
```
i.e. take the softmax probabilities, subtract 1 from the entry of the true class. Average over
the batch (divide by N). That's the entire backward. This simplicity is *why* softmax and
cross-entropy are always paired — the softmax Jacobian and the log's derivative cancel into
`p − y`.

Derivation sketch (so you can say it): L = logsumexp(z) − z_y.
- ∂/∂z_i of logsumexp(z) = exp(z_i)/Σexp(z_j) = p_i.
- ∂/∂z_i of (−z_y) = −1 if i==y else 0.
- Sum: p_i − 1{i==y}. ∎

## 5. The pass criteria (how you know it's right)
- **Forward (softmax):** softmax of `[1000, 1000, 1000]` → `[1/3, 1/3, 1/3]` (no NaN), and
  `[-1000, 0, 1000]` → `[0, 0, 1]` (no NaN). Matches `F.softmax` to ~1e-6.
- **Forward (loss):** your CE matches `F.cross_entropy` to ~1e-6.
- **Backward:** your `dz` matches autograd's `z.grad` to ~1e-6.

## 6. Your procedure
1. Read this sheet. Then close it.
2. Open `week02_softmax_ce.py`, fill the blanks from memory (the 4 things: softmax with max-
   subtraction, logsumexp, the CE loss, and the `p − onehot` backward).
3. Run `python week02_softmax_ce.py` → expects "ALL CHECKS PASSED".
4. Commit: `git add week02_softmax_ce.py && git commit -m "week 2: softmax + CE, manual backward"`.

The key skills this tests: numerical stability awareness (the max-subtraction), and knowing the
`p − y` gradient cold — both come up in real interviews.

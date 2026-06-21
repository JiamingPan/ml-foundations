# Week 1 Study Sheet — MLP backprop by hand (~20 min read, then code)

**The one-sentence goal:** build a tiny 3-layer neural net, and write the gradient math
yourself instead of calling `.backward()`. Read this, then open `week01_mlp.py` and fill
the blanks. Don't re-watch Karpathy — this sheet is the refresher.

---

## 0. What is even happening (the big picture)

A neural net is a function with knobs (weights). Training = nudging the knobs so the output
gets closer to the target. To nudge a knob the right direction, you need the **gradient** of
the loss with respect to that knob: "if I increase this weight a little, does the loss go up
or down, and how fast?"

**Forward pass** = run the input through the net, compute the loss.
**Backward pass** = compute the gradient of the loss w.r.t. every weight.

PyTorch's `.backward()` does the backward pass automatically. Your assignment is to do it by
hand, because interviews test exactly this.

---

## 1. The network (this exact one)

Input `x` → Linear → ReLU → Linear → ReLU → Linear → output.

Dimensions: `in_dim → 64 → 64 → out_dim`. Three weight matrices (W1, W2, W3) and three
bias vectors (b1, b2, b3).

### Forward pass (what you compute, in order)

```
z1 = x  @ W1 + b1      # linear layer 1
a1 = relu(z1)          # nonlinearity
z2 = a1 @ W2 + b2      # linear layer 2
a2 = relu(z2)          # nonlinearity
z3 = a2 @ W3 + b3      # linear layer 3 (output, no relu)
loss = mean((z3 - y)**2)   # mean squared error vs target y
```

`@` is matrix multiply. `x` has shape `(batch, in_dim)`. You **cache** z1,a1,z2,a2,z3 because
the backward pass needs them.

---

## 2. The four formulas that ARE backprop

Backprop = chain rule, applied backward, one layer at a time. At every step you're holding a
quantity called the **upstream gradient** (the gradient of the loss w.r.t. the current layer's
output), and you do two jobs: (a) peel off the gradients for this layer's weights, (b) pass a
new upstream gradient to the layer below.

You only need these four patterns. Memorize them; everything else is repetition.

### (A) Loss gradient — the starting point
For `loss = mean((z3 - y)**2)`:
```
dz3 = (2 / N) * (z3 - y)        # N = number of elements (batch * out_dim)
```
This is your first "upstream gradient." Everything flows from here.

### (B) Linear layer backward — the workhorse
For a layer `z = a @ W + b`, given the upstream gradient `dz` (grad of loss w.r.t. z):
```
dW = a.T @ dz          # gradient for THIS layer's weight
db = dz.sum(axis=0)    # gradient for THIS layer's bias
da = dz @ W.T          # NEW upstream gradient, to hand to the layer below
```
**These three lines are 80% of the whole assignment.** You apply them at layer 3, then 2,
then 1. (Shapes work out: if `a` is (N, in) and `dz` is (N, out), then `a.T @ dz` is (in, out)
— same shape as W. Good sanity check.)

### (C) ReLU backward
ReLU forward zeroed out the negatives. Backward, the gradient only flows where the input was
positive:
```
dz = da * (z > 0)      # kill gradient wherever the pre-activation z was <= 0
```
(`z > 0` is a boolean mask, multiplied elementwise.)

### (D) That's it. There is no (D). Chain (A)→(B)→(C)→(B)→(C)→(B) and you're done.

---

## 3. Backward pass, fully written out (the answer key for the chain)

Starting from the loss and walking backward:
```
dz3 = (2/N) * (z3 - y)         # (A) loss grad

dW3 = a2.T @ dz3               # (B) layer-3 weight grad
db3 = dz3.sum(0)              # (B) layer-3 bias grad
da2 = dz3 @ W3.T              # (B) upstream for layer 2

dz2 = da2 * (z2 > 0)          # (C) relu backward
dW2 = a1.T @ dz2              # (B)
db2 = dz2.sum(0)             # (B)
da1 = dz2 @ W2.T             # (B)

dz1 = da1 * (z1 > 0)          # (C) relu backward
dW1 = x.T  @ dz1              # (B)  (note: input to layer 1 is x, not an activation)
db1 = dz1.sum(0)            # (B)
# (no da0 needed — we don't compute gradients w.r.t. the input x)
```
Return `dW1, db1, dW2, db2, dW3, db3`. **Those six are the deliverable.**

Notice the pattern: relu-backward, then the three linear-backward lines, repeat. The only
special cases are the top (loss grad kicks it off) and the bottom (layer 1's input is `x`).

---

## 4. How you KNOW it's right (the gradient check)

You don't eyeball it. You let PyTorch compute the same gradients automatically and compare:

1. Build the identical forward pass with autograd ON (plain PyTorch, `requires_grad=True`).
2. Call `loss.backward()`.
3. Compare your hand-computed `dW1` to PyTorch's `W1.grad`, etc.
4. If the max difference is below ~`1e-6`, your math is correct. ✅

If a gradient is wrong, the check tells you *which* one — usually a transpose flipped or a
ReLU mask missing. That's the debugging loop.

---

## 5. Your actual procedure now

1. Read sections 1–4 above once (you just did). ~15–20 min.
2. Open `week01_mlp.py`. The forward and the gradient-check harness are scaffolded; the
   backward pass is blanked with `# TODO`.
3. **Close this sheet** and fill in the backward TODOs from memory, using section 2's four
   formulas. Peek back only if truly stuck.
4. Run it. Iterate until the gradient check prints PASS.
5. `git add foundations/week01_mlp.py && git commit -m "week 1: MLP manual backprop"`.

If step 3 is smooth, you're done in 30–45 min and you jump to Week 2 the same day. If it's
slow, that's the point — you found the rust, and now it's gone.

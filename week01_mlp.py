"""
Week 1 — 3-layer MLP with MANUAL backward pass (no .backward(), no autograd).

Goal: implement forward + backward by hand, then prove the gradients are correct by
gradient-checking against PyTorch autograd.

Network:  x -> Linear -> ReLU -> Linear -> ReLU -> Linear -> output
Dims:     in_dim -> 64 -> 64 -> out_dim
Loss:     mean squared error

HOW TO USE THIS FILE:
  1. Read week01_STUDY_SHEET.md first (sections 1-4).
  2. Fill in the `backward` function where it says TODO. The forward pass and the
     gradient-check harness are already written for you.
  3. Run:  python week01_mlp.py
  4. Iterate until it prints "GRADIENT CHECK PASSED".
  5. git commit this file.

The four formulas you need (from the study sheet):
  Loss grad:     dz3 = (2/N) * (z3 - y)
  Linear bwd:    dW = a.T @ dz ;  db = dz.sum(0) ;  da = dz @ W.T
  ReLU bwd:      dz = da * (z > 0)
"""

import torch

torch.manual_seed(0)


# ----------------------------------------------------------------------------
# Parameters: 3 weight matrices + 3 bias vectors. Small random init.
# ----------------------------------------------------------------------------
def init_params(in_dim, hidden, out_dim):
    g = torch.Generator().manual_seed(0)
    p = {
        "W1": torch.randn(in_dim, hidden, generator=g) * 0.5,
        "b1": torch.zeros(hidden),
        "W2": torch.randn(hidden, hidden, generator=g) * 0.5,
        "b2": torch.zeros(hidden),
        "W3": torch.randn(hidden, out_dim, generator=g) * 0.5,
        "b3": torch.zeros(out_dim),
    }
    return p


def relu(z):
    return torch.clamp(z, min=0.0)


# ----------------------------------------------------------------------------
# FORWARD PASS  (already written — study it, this is your reference for backward)
# Returns the output z3 and a cache of everything backward needs.
# ----------------------------------------------------------------------------
def forward(x, p):
    z1 = x @ p["W1"] + p["b1"]
    a1 = relu(z1)
    z2 = a1 @ p["W2"] + p["b2"]
    a2 = relu(z2)
    z3 = a2 @ p["W3"] + p["b3"]          # no relu on the output layer
    cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3}
    return z3, cache


def mse_loss(pred, y):
    return ((pred - y) ** 2).mean()


# ----------------------------------------------------------------------------
# BACKWARD PASS  <-- THIS IS YOUR JOB. Fill in every TODO.
# Inputs:  the cache from forward, the params p, and the target y.
# Output:  a dict of gradients {dW1, db1, dW2, db2, dW3, db3}.
# ----------------------------------------------------------------------------
def backward(cache, p, y):
    x, z1, a1, z2, a2, z3 = (
        cache["x"], cache["z1"], cache["a1"],
        cache["z2"], cache["a2"], cache["z3"],
    )
    N = y.numel()  # total number of elements, for the MSE gradient

    # Fill in each line from memory. No hints. The 4 patterns are:
    #   loss grad → linear backward (dW, db, da) → relu backward → repeat.
    # Return grads for W1,b1,W2,b2,W3,b3.

    # --- (A) loss gradient: grad of MSE loss w.r.t. the output z3 ---
    dz3 = 2/N*(z3-y)

    # --- (B) layer 3 (output) backward: z3 = a2 @ W3 + b3 ---
    dW3 = a2.T @ dz3
    db3 = dz3.sum(axis=0)
    da2 = dz3 @ p["W3"].T

    # --- (C) relu backward at layer 2: a2 = relu(z2) ---
    dz2 = da2 *(z2>0)

    # --- (B) layer 2 backward: z2 = a1 @ W2 + b2 ---
    dW2 =a1.T @ dz2
    db2 =dz2.sum(axis=0)
    da1 =dz2 @ p["W2"].T

    # --- (C) relu backward at layer 1: a1 = relu(z1) ---
    dz1 = da1 *(z1>0)

    # --- (B) layer 1 backward: z1 = x @ W1 + b1   (input is x, not an activation) ---
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0) #bias terms

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}


# ----------------------------------------------------------------------------
# GRADIENT CHECK  (already written — this is how you know your math is right)
# Rebuilds the identical forward with autograd ON, calls .backward(), and compares.
# ----------------------------------------------------------------------------
def gradient_check():
    in_dim, hidden, out_dim, batch = 3, 64, 2, 16
    p = init_params(in_dim, hidden, out_dim)
    x = torch.randn(batch, in_dim)
    y = torch.randn(batch, out_dim)

    # --- your manual version ---
    pred, cache = forward(x, p)
    grads = backward(cache, p, y)

    if any(v is None for v in grads.values()):
        print("Some gradients are still None — fill in the TODOs in backward().")
        return

    # --- autograd reference: same params, requires_grad=True ---
    ref = {k: v.clone().detach().requires_grad_(True) for k, v in p.items()}
    z1 = relu(x @ ref["W1"] + ref["b1"])
    z2 = relu(z1 @ ref["W2"] + ref["b2"])
    z3 = z2 @ ref["W3"] + ref["b3"]
    loss = mse_loss(z3, y)
    loss.backward()

    name_map = {"dW1": "W1", "db1": "b1", "dW2": "W2",
                "db2": "b2", "dW3": "W3", "db3": "b3"}
    print(f"{'param':<6}{'max abs diff':>16}   status")
    print("-" * 36)
    ok = True
    for gname, pname in name_map.items():
        diff = (grads[gname] - ref[pname].grad).abs().max().item()
        status = "ok" if diff < 1e-6 else "MISMATCH"
        if diff >= 1e-6:
            ok = False
        print(f"{pname:<6}{diff:>16.2e}   {status}")
    print("-" * 36)
    print("GRADIENT CHECK PASSED ✅" if ok else "gradient check failed — fix the mismatched grads above")


if __name__ == "__main__":
    gradient_check()

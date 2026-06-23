"""
Week 2 — Softmax + Cross-Entropy with the log-sum-exp trick, MANUAL backward (no autograd).

Goal: implement numerically-stable softmax, log-softmax, cross-entropy loss, and the manual
backward pass, then gradient-check against PyTorch.

HOW TO USE:
  1. Read week02_STUDY_SHEET.md first.
  2. Fill in every line that is left blank (look for the `=` with nothing after it).
  3. Run:  python week02_softmax_ce.py   → expects "ALL CHECKS PASSED"
  4. git commit.

The 4 things you implement (from the study sheet):
  softmax (subtract max) · logsumexp · cross-entropy loss (logsumexp(z) - z_y) · backward (p - onehot)
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)


# ----------------------------------------------------------------------------
# 1. SOFTMAX  (stable: subtract the max before exp).  z: (N, C) logits.
#    Return probabilities, same shape, each row sums to 1.
# ----------------------------------------------------------------------------
def softmax(z):
    # subtract the row max for stability, exp, normalize over the class dim
    m = z.max(dim=1, keepdim=True).values
    ex =torch.exp(z-m)
    return ex/ex.sum(dim=1,keepdim=True)


# ----------------------------------------------------------------------------
# 2. LOGSUMEXP  over the class dim. z: (N, C) -> (N,) . Stable version.
# ----------------------------------------------------------------------------
def logsumexp(z):
    m = z.max(dim=1,keepdim= True).values
    return m.squeeze(1)+ torch.log(torch.exp(z-m).sum(dim=1))


# ----------------------------------------------------------------------------
# 3. CROSS-ENTROPY LOSS.  z: (N, C) logits, y: (N,) integer class labels.
#    Per-example loss = logsumexp(z) - z[true class]; return the MEAN over the batch.
# ----------------------------------------------------------------------------
def cross_entropy(z, y):
    N = z.shape[0]
    z_y = z[torch.arange(N),y]
    per_example = logsumexp(z)-z_y
    return per_example.mean()


# ----------------------------------------------------------------------------
# 4. BACKWARD.  Gradient of the mean cross-entropy w.r.t. the logits z.
# ----------------------------------------------------------------------------
def cross_entropy_backward(z, y):
    N = z.shape[0]
    p = softmax(z)
    onehot = torch.zeros_like(z)
    onehot[torch.arange(N), y] = 1.0
    dz = p-onehot
    return dz/N


# ----------------------------------------------------------------------------
# CHECKS  (written for you)
# ----------------------------------------------------------------------------
def checks():
    ok = True

    # (a) stability: extreme logits must not NaN and must match F.softmax
    for row in [[1000., 1000., 1000.], [-1000., 0., 1000.]]:
        z = torch.tensor([row])
        mine = softmax(z)
        ref = F.softmax(z, dim=1)
        if mine is None or torch.isnan(mine).any() or (mine - ref).abs().max() > 1e-6:
            print(f"softmax FAILED on {row}: got {None if mine is None else mine.tolist()}")
            ok = False
        else:
            print(f"softmax ok on {row}: {mine.tolist()[0]}")

    # (b) loss matches F.cross_entropy
    z = torch.randn(16, 5)
    y = torch.randint(0, 5, (16,))
    mine_loss = cross_entropy(z, y)
    ref_loss = F.cross_entropy(z, y)
    if mine_loss is None or abs(float(mine_loss) - float(ref_loss)) > 1e-6:
        print(f"loss FAILED: mine={mine_loss}, ref={float(ref_loss):.6f}")
        ok = False
    else:
        print(f"loss ok: {float(mine_loss):.6f} vs F.cross_entropy {float(ref_loss):.6f}")

    # (c) backward matches autograd
    zz = z.clone().detach().requires_grad_(True)
    F.cross_entropy(zz, y).backward()
    mine_dz = cross_entropy_backward(z, y)
    if mine_dz is None or (mine_dz - zz.grad).abs().max() > 1e-6:
        diff = "None" if mine_dz is None else f"{(mine_dz - zz.grad).abs().max():.2e}"
        print(f"backward FAILED: max abs diff = {diff}")
        ok = False
    else:
        print(f"backward ok: max abs diff = {(mine_dz - zz.grad).abs().max():.2e}")

    print("-" * 40)
    print("ALL CHECKS PASSED ✅" if ok else "some checks failed — fix the blanks above")


if __name__ == "__main__":
    checks()

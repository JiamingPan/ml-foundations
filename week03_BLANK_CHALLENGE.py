"""
WEEK 3 — AdamW from scratch  [BLANK — cold pass, AI off]

Write the whole AdamW class from memory in the empty space below. No shell, no study sheet.

  class AdamW:
      __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01)
      zero_grad(self)
      step(self)

Then run:  python week03_BLANK_CHALLENGE.py
The check compares your params against torch.optim.AdamW after 200 steps. diff < 1e-6 = pass.
Mark "✓ AdamW done" on the tracker once you pass cold.
"""

import torch


# ---- write your AdamW class from here ----




# ============================================================================================
# CHECK (do NOT edit — uses the AdamW class you wrote above)
# ============================================================================================
def _check(steps=200, lr=1e-2, wd=0.05, seed=0):
    if "AdamW" not in globals():
        print("Not done yet — define your AdamW class above.")
        return
    torch.manual_seed(seed)
    A = torch.randn(32, 8); y = torch.randn(32, 1); w0 = torch.randn(8, 1)

    def run(make_opt):
        w = w0.clone().requires_grad_(True)
        opt = make_opt([w])
        for _ in range(steps):
            opt.zero_grad()
            (((A @ w - y) ** 2).mean()).backward()
            opt.step()
        return w.detach()

    try:
        mine = run(lambda ps: AdamW(ps, lr=lr, weight_decay=wd))
    except Exception as e:
        print("Your AdamW raised:", repr(e))
        return
    ref = run(lambda ps: torch.optim.AdamW(ps, lr=lr, weight_decay=wd))
    diff = (mine - ref).abs().max().item()
    print(f"max param diff after {steps} steps: {diff:.2e}")
    print("BLANK-FILE CHALLENGE PASSED — AdamW cold." if diff < 1e-6
          else "MISMATCH — usual culprits: decay order/scale, eps inside sqrt, t not incremented once/step")


if __name__ == "__main__":
    _check()

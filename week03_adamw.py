"""
WEEK 3 — AdamW optimizer from scratch.

Hand-written AdamW: per-parameter first/second moment estimates, bias correction, and
decoupled weight decay. Checked against torch.optim.AdamW on a small least-squares problem —
matches to < 1e-6.

Run:  python week03_adamw.py
"""

import torch


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr, self.eps, self.wd = lr, eps, weight_decay
        self.b1, self.b2 = betas
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]   # 1st moment (momentum)
        self.v = [torch.zeros_like(p) for p in self.params]   # 2nd moment (variance)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t        # bias-correction denominators
        bc2 = 1 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            p.mul_(1 - self.lr * self.wd)                             # decoupled weight decay
            self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)        # m = b1*m + (1-b1)*g
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2) # v = b2*v + (1-b2)*g^2
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            p.addcdiv_(m_hat, v_hat.sqrt().add(self.eps), value=-self.lr)  # eps OUTSIDE sqrt


# ============================================================================================
# CHECK — optimizes the same problem with this AdamW and torch.optim.AdamW; expects < 1e-6.
# ============================================================================================
def _check(steps=200, lr=1e-2, wd=0.05, seed=0):
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

    mine = run(lambda ps: AdamW(ps, lr=lr, weight_decay=wd))
    ref = run(lambda ps: torch.optim.AdamW(ps, lr=lr, weight_decay=wd))
    diff = (mine - ref).abs().max().item()
    print(f"max param diff vs torch.optim.AdamW after {steps} steps: {diff:.2e}   "
          f"{'GRADIENT CHECK PASSED' if diff < 1e-6 else 'MISMATCH'}")


if __name__ == "__main__":
    _check()

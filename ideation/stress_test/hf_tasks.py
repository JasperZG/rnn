"""Tasks with a declared update F. Outputs are the full task state z.

Convention: z_0 = 0. At step t the network consumes u_t and outputs y_t, whose
target is z_{t+1} = F(z_t, u_t).
"""
import math
import torch


class Task:
    name = ""
    k = 1          # task-state dimension
    n_in = 1
    eps = 0.25     # failure threshold (task units)
    T_train = 50

    def F(self, z, u):
        raise NotImplementedError

    def JF(self, z, u):
        """Task Jacobian dF/dz, shape (B, k, k)."""
        raise NotImplementedError

    def inputs(self, B, T, g, broad=False):
        raise NotImplementedError

    def eval_mask(self, u):
        """(B, T) mask of steps where error is scored."""
        return torch.ones(u.shape[0], u.shape[1])

    def targets(self, u):
        B, T, _ = u.shape
        z = torch.zeros(B, self.k)
        Y = []
        for t in range(T):
            z = self.F(z, u[:, t])
            Y.append(z)
        return torch.stack(Y, 1)

    # features for the defect fit: (discrete key per sample, continuous inputs)
    def split_u(self, u):
        return torch.zeros(u.shape[0], dtype=torch.long), u


class FlipFlop(Task):
    name, k, n_in, eps, T_train = "flipflop", 1, 1, 0.5, 80

    def F(self, z, u):
        return torch.where(u != 0, u, z)

    def JF(self, z, u):
        return (u == 0).float().view(-1, 1, 1)

    def inputs(self, B, T, g, broad=False):
        if broad:
            p = 0.02 + 0.18 * torch.rand(B, 1, 1, generator=g)
        else:
            p = torch.full((B, 1, 1), 0.06)
        hit = (torch.rand(B, T, 1, generator=g) < p).float()
        sgn = torch.where(torch.rand(B, T, 1, generator=g) < 0.5, -1.0, 1.0)
        return hit * sgn

    def eval_mask(self, u):
        hit = (u[..., 0] != 0).float()
        B, T = hit.shape
        m = torch.ones(B, T)
        seen = torch.zeros(B)
        since = torch.full((B,), 1e9)
        for t in range(T):
            since = torch.where(hit[:, t] > 0, torch.zeros(B), since + 1)
            seen = torch.maximum(seen, hit[:, t])
            m[:, t] = ((seen > 0) & (since >= 3)).float()
        return m

    def split_u(self, u):
        return (u[:, 0].round().long() + 1), u[:, :0]


class Accumulation(Task):
    name, k, n_in, eps, T_train = "accumulation", 1, 1, 0.25, 50
    scale = 0.15

    def F(self, z, u):
        return z + u

    def JF(self, z, u):
        return torch.ones(z.shape[0], 1, 1)

    def inputs(self, B, T, g, broad=False):
        s = self.scale
        if broad:
            s = s * (0.5 + 2.5 * torch.rand(B, 1, 1, generator=g))
            mu = (torch.rand(B, 1, 1, generator=g) * 2 - 1) * 0.1
        else:
            mu = 0.0
        return (torch.rand(B, T, 1, generator=g) * 2 - 1) * s + mu


class CtxInt(Task):
    """Context-dependent integration: integrate the stream selected by ctx."""
    name, k, n_in, eps, T_train = "ctxint", 1, 3, 0.25, 50
    scale = 0.15

    def F(self, z, u):
        sel = torch.where(u[:, 2:3] > 0.5, u[:, 1:2], u[:, 0:1])
        return z + sel

    def JF(self, z, u):
        return torch.ones(z.shape[0], 1, 1)

    def inputs(self, B, T, g, broad=False):
        s = self.scale
        if broad:
            s = s * (0.5 + 2.5 * torch.rand(B, 1, 1, generator=g))
            mu = (torch.rand(B, 1, 2, generator=g) * 2 - 1) * 0.1
        else:
            mu = torch.zeros(B, 1, 2)
        st = (torch.rand(B, T, 2, generator=g) * 2 - 1) * s + mu
        ctx = (torch.rand(B, 1, 1, generator=g) < 0.5).float().expand(B, T, 1)
        return torch.cat([st, ctx], -1)

    def split_u(self, u):
        ctx = (u[:, 2] > 0.5)
        rel = torch.where(ctx, u[:, 1], u[:, 0])
        irr = torch.where(ctx, u[:, 0], u[:, 1])
        return ctx.long(), torch.stack([rel, irr], 1)


class Oscillation(Task):
    name, k, n_in, eps, T_train = "oscillation", 2, 1, 0.25, 50
    omega = 2 * math.pi / 16

    def __init__(self):
        c, s = math.cos(self.omega), math.sin(self.omega)
        self.R = torch.tensor([[c, -s], [s, c]])

    def F(self, z, u):
        return z @ self.R.to(z.dtype).T + torch.cat([u, torch.zeros_like(u)], -1)

    def JF(self, z, u):
        return self.R.expand(z.shape[0], 2, 2)

    def inputs(self, B, T, g, broad=False):
        u = torch.zeros(B, T, 1)
        amp = 0.5 + torch.rand(B, generator=g) if broad else torch.ones(B)
        u[:, 0, 0] = amp
        return u

    def eval_mask(self, u):
        m = torch.ones(u.shape[0], u.shape[1])
        return m

    def split_u(self, u):
        return (u[:, 0] != 0).long(), u[:, :0]


TASKS = {c.name: c for c in (FlipFlop, Accumulation, CtxInt, Oscillation)}

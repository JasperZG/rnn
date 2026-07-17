"""
Recurrent architectures. The vanilla tanh RNN is the one in which the dynamical
objects appear directly in the unit activations; the gated architectures (GRU,
LSTM) test that the recovered structure is a property of the task, not the
architecture (Maheswaranathan et al., 2019).
"""
import torch, torch.nn as nn


class VanillaRNN(nn.Module):
    """h_{t+1} = tanh(W h_t + Win u_t + b). Optional rotation-biased (antisymmetric)
    init helps oscillation tasks escape the fixed-point-biased basin near identity."""
    def __init__(self, n_in, n_out, N=128, rotation_init=0.0, seed=0):
        super().__init__()
        self.N = N
        g = torch.Generator().manual_seed(seed)
        self.Win = nn.Linear(n_in, N, bias=False)
        self.W = nn.Linear(N, N, bias=True)
        self.Wout = nn.Linear(N, n_out, bias=False)
        with torch.no_grad():
            A = torch.randn(N, N, generator=g) * (0.9 / (N ** 0.5))
            if rotation_init > 0:
                A = rotation_init * (A - A.T) / 2 + (1 - rotation_init) * A  # bias toward rotation
            self.W.weight.copy_(A); self.W.bias.zero_()

    def step(self, h, u):
        return torch.tanh(self.W(h) + self.Win(u))

    def forward(self, x, h0=None):
        B = x.shape[0]
        h = torch.zeros(B, self.N, device=x.device) if h0 is None else h0
        H = []
        for t in range(x.shape[1]):
            h = self.step(h, x[:, t]); H.append(h)
        H = torch.stack(H, 1)
        return self.Wout(H), H


class GRUNet(nn.Module):
    def __init__(self, n_in, n_out, N=128, seed=0):
        super().__init__(); self.N = N
        torch.manual_seed(seed)
        self.rnn = nn.GRU(n_in, N, batch_first=True)
        self.Wout = nn.Linear(N, n_out, bias=False)

    def step(self, h, u):
        """One autonomous GRU-cell update on the hidden state, so the same
        slow-point search and Jacobian analysis used for the vanilla RNN run
        unchanged on the GRU (the hidden state IS the dynamical state here)."""
        out, _ = self.rnn(u.unsqueeze(1), h.unsqueeze(0).contiguous())
        return out.squeeze(1)

    def forward(self, x, h0=None):
        H, _ = self.rnn(x, None if h0 is None else h0.unsqueeze(0))
        return self.Wout(H), H


class LSTMNet(nn.Module):
    """The LSTM state is the pair (h, c). Fixed points must be sought in the
    JOINT state z = [h; c] (dimension 2N), and the Jacobian is 2N x 2N. The
    analysis helpers below expose that joint state so the same search/Jacobian
    machinery applies; forward() (training) still reads the N-dim hidden h."""
    def __init__(self, n_in, n_out, N=128, seed=0):
        super().__init__(); self.N = N; self._Nh = N
        torch.manual_seed(seed)
        self.rnn = nn.LSTM(n_in, N, batch_first=True)
        self.Wout = nn.Linear(N, n_out, bias=False)

    def set_joint(self, on):
        """Toggle .N between the hidden dim (training/forward) and 2*N (joint
        analysis). find_slow_points reshapes visited states with net.N, so this
        must be True while searching the joint state and False otherwise."""
        self.N = 2 * self._Nh if on else self._Nh

    def step(self, z, u):
        """One autonomous LSTM-cell update on the joint state z = [h; c]."""
        Nh = self._Nh
        h = z[:, :Nh].contiguous(); c = z[:, Nh:].contiguous()
        out, (hn, cn) = self.rnn(u.unsqueeze(1),
                                 (h.unsqueeze(0).contiguous(), c.unsqueeze(0).contiguous()))
        return torch.cat([hn.squeeze(0), cn.squeeze(0)], dim=1)

    def joint_states(self, x):
        """Joint (h, c) trajectory over the input, for seeding the search."""
        B = x.shape[0]; Nh = self._Nh
        h = torch.zeros(1, B, Nh, device=x.device); c = torch.zeros(1, B, Nh, device=x.device)
        Z = []
        with torch.no_grad():
            for t in range(x.shape[1]):
                _, (h, c) = self.rnn(x[:, t:t + 1], (h, c))
                Z.append(torch.cat([h.squeeze(0), c.squeeze(0)], dim=1))
        return torch.stack(Z, 1)

    def forward(self, x, h0=None):
        H, _ = self.rnn(x)
        return self.Wout(H), H


def build(arch, n_in, n_out, N=128, seed=0, rotation_init=0.0):
    if arch == "rnn": return VanillaRNN(n_in, n_out, N, rotation_init, seed)
    if arch == "gru": return GRUNet(n_in, n_out, N, seed)
    if arch == "lstm": return LSTMNet(n_in, n_out, N, seed)
    raise ValueError(arch)

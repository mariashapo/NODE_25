from typing import Sequence
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, widths: Sequence[int], out_dim: int, act: str = "tanh"):
        super().__init__()
        acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}
        A = acts.get(act, nn.Tanh)
        layers = []
        d = in_dim
        for w in widths:
            layers += [nn.Linear(d, w), A()]
            d = w
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # [..., in_dim]
        return self.net(x)

class NeuralODE(nn.Module):
    """Vector field dy/dt = f_theta(t, y)."""
    def __init__(self, widths=(32, 32), act="tanh", time_invariant=True, state_dim=2):
        super().__init__()
        self.time_invariant = time_invariant
        in_dim = state_dim if time_invariant else state_dim + 1
        self.f = MLP(in_dim, widths, state_dim, act)

    def rhs(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.time_invariant:
            x = y
        else:
            t_col = t.expand_as(y[..., :1])
            x = torch.cat([t_col, y], dim=-1)
        return self.f(x)
